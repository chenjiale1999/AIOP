"""Open-ended (LLM) mode: generating operators that have no A5 template."""

import json

import pytest

from aiop.agent.orchestrator import OperatorAgent
from aiop.config import load_config
from aiop.generation.llm import LLMClient
from aiop.runtime.hardware import build_open_plan, build_plan
from aiop.specs import TaskSpec, TensorSpec, UnsupportedOperator

GOOD_KERNEL = '''"""LLM-drafted kernel for an operator with no aiop template."""

import tilelang
import tilelang.ascend.language as T


def build():
    N = 131072
    TILE = 8192
    BLOCKS = 16

    @T.prim_func
    def main(x: T.Tensor((131072,), "float32"), y: T.Tensor((131072,), "float32")):
        with T.Kernel(BLOCKS) as bx:
            with T.SimtVF(threads=256):
                for i in T.Parallel(TILE):
                    y[bx * TILE + i] = T.max(x[bx * TILE + i], T.float32(0))

    return main
'''

BROKEN_KERNEL = GOOD_KERNEL.replace(
    "y[bx * TILE + i] = T.max(x[bx * TILE + i], T.float32(0))",
    "y[bx * TILE + i] = T.max(missing_buffer[bx * TILE + i], T.float32(0))",
)

GOOD_GOLDEN = "    out = torch.relu(x)"


def _spec(*, metadata: bool = False) -> TaskSpec:
    inputs = [TensorSpec("x", (256, 512), "float32")]
    if metadata:
        inputs.append(
            TensorSpec("cache_seqlens", (8,), "int32", role="metadata", values="as_tensor(lens)")
        )
    return TaskSpec(
        name="fusedprobe",
        family="attention",
        inputs=inputs,
        outputs=[TensorSpec("y", (256, 512), "float32")],
        supported=False,
        unsupported_reasons=["attention has no A5 template family"],
        manifest={"op": "FakeAttentionFwdOp"},
    )


def _cfg(tmp_path, *, llm: bool):
    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.simulation.backend = "static"
    cfg.agent.max_attempts = 2
    cfg.agent.llm.enabled = llm
    cfg.agent.llm.base_url = "http://fake-llm" if llm else ""
    cfg.agent.llm.model = "fake-model" if llm else ""
    return cfg


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(LLMClient, "available", property(lambda self: True))
    calls = {"kernel": 0, "golden": 0, "repair": 0}

    def generate_kernel(self, spec_dump, design, knowledge, open_mode=False):
        calls["kernel"] += 1
        return GOOD_KERNEL

    def generate_golden(self, spec_dump, design):
        calls["golden"] += 1
        return GOOD_GOLDEN

    def repair_kernel(self, source, error, design):
        calls["repair"] += 1
        return GOOD_KERNEL

    monkeypatch.setattr(LLMClient, "generate_kernel", generate_kernel)
    monkeypatch.setattr(LLMClient, "generate_golden", generate_golden)
    monkeypatch.setattr(LLMClient, "repair_kernel", repair_kernel)
    return calls


def test_open_plan_only_fixes_the_grid_limit():
    spec = _spec()
    with pytest.raises(UnsupportedOperator):
        build_plan(spec)
    plan = build_open_plan(spec)
    assert plan.variant == "llm_open"
    assert plan.blocks == 64
    assert plan.buffer_bytes == {}


def test_without_llm_the_operator_stays_blocked(tmp_path):
    cfg = _cfg(tmp_path, llm=False)
    result = OperatorAgent(cfg).run(_spec(), simulate=False)
    assert result["status"] == "blocked_unsupported"
    assert result["mode"] == "blocked"
    assert any("agent.llm" in note for note in result["notes"])


@pytest.mark.slow
def test_open_mode_drafts_and_compiles_with_llm(tmp_path, fake_llm):
    cfg = _cfg(tmp_path, llm=True)
    result = OperatorAgent(cfg).run(_spec(), simulate=True)
    assert result["status"] == "success_static_only", result["status"]
    assert result["mode"] == "open"
    assert result["generator"] == "llm"
    assert result["compile"]["ok"] is True
    assert result["golden_source"] == "llm-self-referenced"
    # A self-referenced golden never counts as independent evidence.
    assert result["verified"] is False
    assert result["self_referenced"] is True
    task_dir = tmp_path / "workspace" / result["task_id"]
    assert (task_dir / "golden_ref.py").is_file()
    test_src = (task_dir / "test_fusedprobe.py").read_text(encoding="utf-8")
    assert "golden_source: llm-self-referenced" in test_src
    assert "torch.relu(x)" in test_src
    design = (task_dir / "DESIGN.md").read_text(encoding="utf-8")
    assert "LLM 开放模式" in design
    data = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
    assert data["plan"]["variant"] == "llm_open"


@pytest.mark.slow
def test_compile_failure_is_repaired_by_the_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMClient, "available", property(lambda self: True))
    monkeypatch.setattr(LLMClient, "generate_golden", lambda self, s, d: GOOD_GOLDEN)
    monkeypatch.setattr(
        LLMClient,
        "generate_kernel",
        lambda self, s, d, k, open_mode=False: BROKEN_KERNEL,
    )
    repaired = {"n": 0}

    def repair_kernel(self, source, error, design):
        repaired["n"] += 1
        return GOOD_KERNEL

    monkeypatch.setattr(LLMClient, "repair_kernel", repair_kernel)

    cfg = _cfg(tmp_path, llm=True)
    result = OperatorAgent(cfg).run(_spec(), simulate=False)
    assert repaired["n"] == 1
    assert result["generator"] == "llm_repair"
    assert result["compile"]["ok"] is True
    assert result["status"] == "success_static_only"
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["ok"] is False
    assert result["attempts"][1]["ok"] is True


@pytest.mark.slow
def test_metadata_inputs_skip_numeric_verification(tmp_path, fake_llm):
    cfg = _cfg(tmp_path, llm=True)
    result = OperatorAgent(cfg).run(_spec(metadata=True), simulate=True)
    assert result["mode"] == "open"
    assert result["golden_source"] == "unavailable"
    assert fake_llm["golden"] == 0
    assert result["simulation"] is None
    assert result["compile"]["ok"] is True
    assert any("L3 skipped" in note for note in result["notes"])
    assert "test" not in result["artifacts"]
