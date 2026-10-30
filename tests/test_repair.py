"""Repair-loop behavior: a bad draft must be rejected and regenerated."""

from __future__ import annotations

import json

import pytest

from aiop.agent.orchestrator import OperatorAgent
from aiop.config import load_config
from aiop.generation import generator as real_generator
from aiop.specs import TaskSpec, TensorSpec

BAD_KERNEL = '''"""bad draft"""
import tilelang.ascend.language as T


def build():
    @T.prim_func
    def main(X: T.Tensor((128000,), "float32"), Y: T.Tensor((128000,), "float32")):
        with T.Kernel(1, threads=128) as bx:
            with T.SimtVF(threads=128):
                for i in T.Parallel(1024):
                    Y[i] = T.max(X[i], T.float32(0))

    return main
'''


@pytest.mark.slow
def test_static_failure_is_regenerated(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.simulation.backend = "static"
    state = {"first": True}

    def fake_generate(spec, plan, directory, cfg, **kwargs):
        if state["first"]:
            state["first"] = False
            path = directory / f"{spec.name}.py"
            path.write_text(BAD_KERNEL, encoding="utf-8")
            return path, "bad_draft"
        return real_generator.generate(spec, plan, directory, cfg, **kwargs)

    monkeypatch.setattr("aiop.agent.orchestrator.generate", fake_generate)
    agent = OperatorAgent(cfg)
    spec = TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )
    result = agent.run(spec, simulate=False)
    assert result["status"] in ("success", "success_static_only")
    assert result["compile"]["ok"] is True
    assert result["attempts"][0]["stage"] == "static"
    assert result["attempts"][0]["ok"] is False
    # The rejected draft must not survive in the delivered kernel.
    kernel = (tmp_path / "workspace" / result["task_id"] / "relu.py").read_text(encoding="utf-8")
    assert "threads=128)" not in kernel


@pytest.mark.slow
def test_accuracy_failure_triggers_regeneration(tmp_path, monkeypatch):
    from aiop.simulation.runner import SimResult

    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.simulation.backend = "static"
    calls = {"n": 0}

    def fake_sim(task_dir, test_path, spec, cfg, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimResult(status="failed", backend="fake", error="max_abs_error too large")
        return SimResult(status="passed", backend="fake", result={"matched_ratio": 1.0})

    monkeypatch.setattr("aiop.agent.orchestrator.run_simulation", fake_sim)
    agent = OperatorAgent(cfg)
    spec = TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )
    result = agent.run(spec, simulate=True)
    assert calls["n"] == 2
    assert result["status"] == "success"
    assert result["verified"] is True
    assert result["simulation"]["status"] == "passed"
