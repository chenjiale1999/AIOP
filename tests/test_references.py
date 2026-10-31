"""Dual-baseline discovery, integration gate, and 1e-5 parity encoding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiop.agent.orchestrator import OperatorAgent
from aiop.checks.integration import check_integration
from aiop.config import load_config
from aiop.references import (
    discover_tileops_repo,
    find_reference_pair,
    op_stem,
    render_reference_context,
)
from aiop.runtime.hardware import build_plan
from aiop.simulation.testgen import reference_precision, write_test
from aiop.specs import TaskSpec, TensorSpec


def _spec(name="ReluFwdOp", family="elementwise_unary", dtype="float32"):
    return TaskSpec(
        name,
        family,
        [TensorSpec("X", (128, 1000), dtype)],
        [TensorSpec("Y", (128, 1000), dtype)],
    )


def _fake_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "tileops" / "kernels" / "ascend" / "elementwise").mkdir(parents=True)
    (repo / "src" / "tileops" / "kernels" / "elementwise").mkdir(parents=True)
    (repo / "tests" / "ops").mkdir(parents=True)
    (repo / "tests" / "ops" / "test_other.py").write_text("OtherFwdOp\n", encoding="utf-8")
    (repo / "src" / "tileops" / "kernels" / "elementwise" / "other.py").write_text(
        "class OtherKernel:\n    pass\n", encoding="utf-8"
    )
    (repo / "src" / "tileops" / "kernels" / "ascend" / "elementwise" / "other.py").write_text(
        "class AscendOtherKernel:\n    pass\n", encoding="utf-8"
    )
    return repo


def test_op_stem_normalization():
    assert op_stem("ReluFwdOp") == "Relu"
    assert op_stem("GQADenseFwdOp") == "GQADense"
    assert op_stem("relu") == "relu"


def test_reference_repository_is_discovered():
    cfg = load_config()
    repo = discover_tileops_repo(cfg)
    if not repo:
        pytest.skip("TileOPs reference checkout is not present")
    assert (Path(repo) / "src" / "tileops" / "kernels" / "ascend").is_dir()


def test_relu_pair_has_both_baselines_and_target_path():
    cfg = load_config()
    if not discover_tileops_repo(cfg):
        pytest.skip("TileOPs reference checkout is not present")
    pair = find_reference_pair(_spec(), cfg)
    assert pair.cuda is not None and pair.cuda.class_name.endswith("Kernel")
    assert pair.ascend is not None and pair.ascend.class_name.startswith("Ascend")
    assert pair.ascend_exact is True
    assert pair.target_path().endswith("kernels/ascend/elementwise/relu.py")
    context = render_reference_context(pair)
    assert "CUDA peer" in context
    assert "Verified Ascend A5 peer" in context
    assert "1e-5" in context


def test_attention_pair_uses_cuda_peer_and_style_baseline():
    cfg = load_config()
    if not discover_tileops_repo(cfg):
        pytest.skip("TileOPs reference checkout is not present")
    pair = find_reference_pair(_spec("GQADenseFwdOp", "attention", "bfloat16"), cfg)
    assert pair.cuda is not None
    assert "gqa" in pair.cuda.rel_path
    assert pair.ascend is not None
    assert pair.ascend_exact is False
    assert pair.target_path().endswith("kernels/ascend/attention/gqa/gqadense.py")


def test_missing_cuda_reference_blocks_manifest_operator(tmp_path):
    repo = _fake_repo(tmp_path)
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    (manifest / "ops.yaml").write_text("NoSuchFwdOp:\n  family: elementwise\n", encoding="utf-8")
    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.paths.tileops_manifest = str(manifest)
    cfg.references.tileops_repo = str(repo)
    spec = _spec("NoSuchFwdOp")
    spec.manifest = {"op": "NoSuchFwdOp"}
    result = OperatorAgent(cfg).run(spec, simulate=False)
    assert result["status"] == "blocked_reference"
    task_dir = Path(cfg.paths.workspace) / result["task_id"]
    assert (task_dir / "references.json").is_file()
    assert "no CUDA reference kernel" in " ".join(result["reference_reasons"])


READY_SOURCE = '''\
import tilelang.ascend.language as T
from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel


def build():
    @T.prim_func
    def main(X: T.Tensor((16,), "float32"), Y: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((16,), "float32")
            T.copy(X, x_ub)
            T.copy(x_ub, Y)

    return main


class AscendReluKernel(Kernel):
    pass


register_kernel_type("ReluFwdOp", "ascend", AscendReluKernel)
'''


def test_integration_gate_accepts_repo_ready_kernel(tmp_path):
    kernel = tmp_path / "ReluFwdOp.py"
    kernel.write_text(READY_SOURCE, encoding="utf-8")
    report = check_integration(_spec(), kernel, None, load_config())
    assert report.ready is True, report.fixes
    assert report.class_name == "AscendReluKernel"


def test_integration_gate_reports_missing_registration(tmp_path):
    kernel = tmp_path / "ReluFwdOp.py"
    kernel.write_text(
        READY_SOURCE.replace(
            'register_kernel_type("ReluFwdOp", "ascend", AscendReluKernel)', ""
        ),
        encoding="utf-8",
    )
    report = check_integration(_spec(), kernel, None, load_config())
    assert report.ready is False
    assert any(check.rule == "INT003" for check in report.errors)
    assert any("register_kernel_type" in fix for fix in report.fixes)


def test_parity_policy_is_strict_for_fp32_and_floored_for_low_precision():
    strict = reference_precision("float32", 1e-5)
    assert strict["policy"] == "strict"
    assert strict["atol"] == 1e-5
    low = reference_precision("bfloat16", 1e-5)
    assert low["policy"] == "dtype_floor"
    assert low["atol"] > 1e-5
    assert low["declared_atol"] == 1e-5


def test_generated_test_embeds_reference_and_perf_gates(tmp_path):
    spec = _spec("relu", "elementwise_unary", "float32")
    spec.perf.target = 0.5
    cfg = load_config()
    path = write_test(
        spec,
        build_plan(spec),
        tmp_path,
        cann_home=cfg.cann.home,
        device_flags=cfg.compile.device_flags,
        warmup=1,
        repeats=2,
        tilelang_repo=cfg.paths.tilelang_repo,
        reference_atol=1e-5,
        enforce_reference=True,
    )
    text = path.read_text(encoding="utf-8")
    assert "REFERENCE_AIOP" in text
    assert "'declared_atol': 1e-05" in text
    assert "PERF_TARGET_AIOP" in text
    assert "'target': 0.5" in text
    assert "reference_policy" in text
    assert "failure_kind" in text
    compile(text, str(path), "exec")


def test_integrate_command_reports_target(tmp_path, capsys):
    from aiop.cli import main

    repo = _fake_repo(tmp_path)
    spec = _spec("ReluFwdOp")
    task = tmp_path / "task"
    task.mkdir()
    spec.save(task / "spec.yaml")
    (task / "ReluFwdOp.py").write_text(READY_SOURCE, encoding="utf-8")
    cfg = tmp_path / "aiop.yaml"
    cfg.write_text(f"references:\n  tileops_repo: {repo}\n", encoding="utf-8")
    code = main(["--config", str(cfg), "integrate", str(task), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ready"] is True
    assert payload["target_path"].endswith("kernels/ascend/elementwise/relu.py")
