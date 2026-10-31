"""LLM prompt/API-context and reducer-alias normalization tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aiop import ascend_ops
from aiop.agent.orchestrator import OperatorAgent
from aiop.checks.static import check_semantics
from aiop.config import load_config
from aiop.generation.generator import generate
from aiop.generation.llm import LLMClient, REDUCER_CONTRACT, build_system_prompt
from aiop.runtime.hardware import build_plan
from aiop.specs import TaskSpec, TensorSpec


def test_system_prompt_carries_allowlist_and_reducer_contract():
    text = build_system_prompt(load_config())
    assert "AUTHORITATIVE API ALLOWLIST" in text
    assert "alloc_shared" in text
    assert "vadd" in text
    assert "T.reducer_init" in text
    assert "T.finalize_reducer" in text
    # The forbidden aliases are named explicitly so the model stops guessing.
    assert "T.reduce_init" in text
    assert "T.reducer_finalize" in text


def test_repair_prompt_includes_knowledge_and_contract():
    client = LLMClient(load_config())
    captured: dict[str, str] = {}

    def fake_complete(prompt, system=None, timeout=None):
        captured["prompt"] = prompt
        captured["system"] = system or ""
        return "def build():\n    return None\n"

    client.complete = fake_complete  # type: ignore[method-assign]
    client.repair_kernel("def build():\n    pass\n", "API001", "design text", "KNOWLEDGE_TOKEN")
    assert "KNOWLEDGE_TOKEN" in captured["prompt"]
    assert "REDUCER CONTRACT" in captured["prompt"]
    assert REDUCER_CONTRACT.strip().splitlines()[0] in captured["prompt"]
    assert "AUTHORITATIVE API ALLOWLIST" in client.system


def test_known_aliases_are_normalized():
    source = (
        "acc = T.alloc_reduce((4,), 'float32')\n"
        "T.reduce_init(acc)\n"
        "T.reduce_update(acc[i], x[i])\n"
        "T.reducer_finalize(acc)\n"
        "y = T.reduce_max(x, out, dim=1)\n"
    )
    rewritten, fixes = ascend_ops.normalize_known_aliases(source)
    assert "T.alloc_reducer" in rewritten
    assert "T.reducer_init" in rewritten
    assert "T.reducer_update" in rewritten
    assert "T.finalize_reducer" in rewritten
    assert "T.reduce_max" in rewritten  # a real op is untouched
    assert {fix["from"] for fix in fixes} == {
        "T.alloc_reduce",
        "T.reduce_init",
        "T.reduce_update",
        "T.reducer_finalize",
    }


def test_sem001_message_names_the_canonical_api(tmp_path):
    kernel = tmp_path / "bad.py"
    kernel.write_text(
        "import tilelang.ascend.language as T\n"
        "def build():\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((8, 8), 'float32')):\n"
        "        with T.Kernel(1) as bx:\n"
        "            acc = T.alloc_reducer((8,), 'float32', op='sum')\n"
        "    return main\n",
        encoding="utf-8",
    )
    spec = TaskSpec(
        "row_reduce",
        "row_reduce",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128,), "float32")],
    )
    plan = build_plan(spec)
    report = check_semantics(spec, plan, kernel)
    message = " | ".join(f.message for f in report.findings if f.rule == "SEM001")
    assert "T.reducer_init" in message
    assert "T.finalize_reducer" in message
    assert "T.reducer_update" in message


def test_reducer_as_reduce_out_is_rejected(tmp_path):
    kernel = tmp_path / "bad2.py"
    kernel.write_text(
        "import tilelang.ascend.language as T\n"
        "def build():\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((8, 8), 'float32')):\n"
        "        with T.Kernel(1) as bx:\n"
        "            acc = T.alloc_reducer((8,), 'float32', op='max')\n"
        "            T.reducer_init(acc)\n"
        "            T.reduce_max(X, acc, dim=1)\n"
        "    return main\n",
        encoding="utf-8",
    )
    spec = TaskSpec(
        "row_reduce",
        "row_reduce",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128,), "float32")],
    )
    report = check_semantics(spec, build_plan(spec), kernel)
    assert any(f.rule == "SEM018" for f in report.findings)


def test_direct_reducer_read_is_rejected(tmp_path):
    kernel = tmp_path / "bad3.py"
    kernel.write_text(
        "import tilelang.ascend.language as T\n"
        "def build():\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((8, 8), 'float32')):\n"
        "        with T.Kernel(1) as bx:\n"
        "            acc = T.alloc_reducer((8,), 'float32', op='max')\n"
        "            T.reducer_init(acc)\n"
        "            T.reducer_update(acc[i], X[0, i])\n"
        "            Y = acc[i]\n"
        "    return main\n",
        encoding="utf-8",
    )
    spec = TaskSpec(
        "row_reduce",
        "row_reduce",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128,), "float32")],
    )
    report = check_semantics(spec, build_plan(spec), kernel)
    assert any(f.rule == "SEM018" for f in report.findings)


def test_generate_normalizes_llm_output(tmp_path, monkeypatch):
    bad = """\
import tilelang
import tilelang.ascend.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((8,), "float32"), Y: T.Tensor((8,), "float32")):
        with T.Kernel(1) as bx:
            acc = T.alloc_reduce((8,), "float32", op="sum")
            T.reduce_init(acc)
            T.reduce_update(acc[i], X[i])
            out = T.alloc_fragment((8,), "float32")
            T.reduce_finalize(acc, out)
            for i in T.Parallel(8):
                Y[i] = out[i]
    return main
"""
    cfg = load_config()
    cfg.agent.llm.enabled = True
    cfg.agent.llm.base_url = "http://fake-llm"
    cfg.agent.llm.model = "fake-model"
    monkeypatch.setattr(LLMClient, "available", property(lambda self: True))
    monkeypatch.setattr(
        LLMClient,
        "generate_kernel",
        lambda self, spec_dump, design, knowledge, open_mode=False: bad,
    )
    spec = TaskSpec(
        "row_reduce",
        "row_reduce",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128,), "float32")],
    )
    path, name = generate(spec, build_plan(spec), tmp_path, cfg, strategy="llm")
    source = path.read_text(encoding="utf-8")
    assert name == "llm"
    assert "T.alloc_reduce(" not in source
    assert "T.reduce_init(" not in source
    assert "T.reduce_update(" not in source
    assert "T.reduce_finalize(" not in source
    ast.parse(source)
    fixes = (tmp_path / "llm_alias_fixes.json").read_text(encoding="utf-8")
    assert "T.reducer_init" in fixes


@pytest.mark.slow
def test_llm_static_exhaustion_falls_back_to_template(tmp_path, monkeypatch):
    bad = """\
import tilelang
import tilelang.ascend.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((128, 1000), "float32"), Y: T.Tensor((128, 1000), "float32")):
        with T.Kernel(1) as bx:
            T.printf("bad")
            T.tile.vadd(X, X, X)
    return main
"""
    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.simulation.backend = "static"
    cfg.agent.max_attempts = 2
    cfg.agent.llm.enabled = True
    cfg.agent.llm.base_url = "http://fake-llm"
    cfg.agent.llm.model = "fake-model"
    monkeypatch.setattr(LLMClient, "available", property(lambda self: True))
    monkeypatch.setattr(
        LLMClient,
        "generate_kernel",
        lambda self, spec_dump, design, knowledge, open_mode=False: bad,
    )
    monkeypatch.setattr(
        LLMClient,
        "repair_kernel",
        lambda self, source, error, design, knowledge="": bad,
    )
    spec = TaskSpec(
        "softmax_fallback",
        "softmax",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )
    result = OperatorAgent(cfg).run(spec, simulate=False, strategy="llm")
    assert result["status"] == "success_static_only", result["status"]
    assert result["generator"] == "template"
    assert result["compile"]["ok"] is True
