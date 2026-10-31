"""Campaign architecture: contract, skills, search, evidence and audit."""

from __future__ import annotations

import ast
import json

import pytest

from aiop.agent.audit import build_lowering_audit
from aiop.agent.campaign import render_campaign_report, run_negative_probe
from aiop.agent.contract import build_contract, write_contract
from aiop.agent.evidence import EvidenceStore, diagnose
from aiop.agent.search import propose_candidates
from aiop.agent.skills import SkillRegistry, write_context_manifest, write_skill_lock
from aiop.checks.compiler import CompileResult
from aiop.config import load_config
from aiop.generation.templates import write_kernel
from aiop.runtime.hardware import build_plan, check_plan
from aiop.simulation.testgen import render_test
from aiop.specs import TaskSpec, TensorSpec


def _softmax() -> TaskSpec:
    return TaskSpec(
        "softmax",
        "softmax",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )


def _gemm() -> TaskSpec:
    return TaskSpec(
        "gemm",
        "gemm",
        [TensorSpec("A", (256, 256), "bfloat16"), TensorSpec("B", (448, 256), "bfloat16")],
        [TensorSpec("C", (256, 448), "float32")],
    )


def test_contract_freezes_abi_precision_and_resources(tmp_path):
    spec = _softmax()
    plan = build_plan(spec)
    bundle = SkillRegistry().select(spec)
    contract = build_contract(spec, plan, obligations=bundle.obligations)
    data = contract.to_dict()

    assert data["contract_hash"]
    assert data["family"] == "softmax"
    assert data["determinism"] is True
    assert data["inputs"][0]["direction"] == "input"
    assert data["outputs"][0]["direction"] == "output"
    assert data["precision"]["matched_ratio"] == spec.precision.matched_ratio
    assert data["limits"]["ub_bytes"] > 0
    assert data["plan_facts"]["variant"] == plan.variant
    assert data["plan_facts"]["ub_used"] == plan.ub_used
    assert data["headroom"]["ub_bytes"] == data["limits"]["ub_bytes"] - plan.ub_used
    assert "L0-static" in data["obligations"]
    assert "API-whitelist" in data["obligations"]

    path = write_contract(tmp_path, contract)
    assert json.loads(path.read_text(encoding="utf-8"))["contract_hash"] == data["contract_hash"]


def test_skill_registry_selects_layout_for_gemm_and_locks_content(tmp_path):
    spec = _gemm()
    bundle = SkillRegistry().select(spec, board=True)
    names = {meta.name for meta in bundle.skills}
    assert {"aiop-op-pipeline", "tilelang-ascend", "asc-npu-arch", "tilelang-semantic"} <= names
    assert "tilelang-layout" in names
    assert "ascendc-api-kernel-debug" in names
    assert "layout-inference" in bundle.obligations
    assert all(len(meta.sha256) == 64 for meta in bundle.skills)

    lock = json.loads(write_skill_lock(tmp_path, bundle).read_text(encoding="utf-8"))
    assert set(lock["skills"]) == names
    manifest = json.loads(
        write_context_manifest(
            tmp_path,
            task_id="gemm-test",
            spec=spec,
            bundle=bundle,
            knowledge_hits=[{"doc": "knowledge/06", "heading": "roofline", "score": 1.0}],
            open_mode=False,
        ).read_text(encoding="utf-8")
    )
    assert manifest["task_id"] == "gemm-test"
    assert manifest["knowledge"][0]["doc"] == "knowledge/06"
    assert manifest["skills"][0]["sha256"]


@pytest.mark.parametrize("spec", [_softmax(), _gemm(), TaskSpec(
    "layernorm",
    "layernorm",
    [
        TensorSpec("X", (128, 1000), "float32"),
        TensorSpec("W", (1000,), "float32"),
        TensorSpec("B", (1000,), "float32"),
    ],
    [TensorSpec("Y", (128, 1000), "float32")],
)], ids=["softmax", "gemm", "layernorm"])
def test_plan_candidates_are_legal(spec):
    base = build_plan(spec)
    candidates = propose_candidates(spec, base)
    assert candidates
    assert candidates[0].score >= candidates[-1].score
    assert any(cand.plan.to_dict() == base.to_dict() for cand in candidates)
    for cand in candidates:
        assert check_plan(spec, cand.plan)["errors"] == [], cand.label


def test_lowering_audit_detects_printf_and_missing_sync():
    spec = _softmax()
    plan = build_plan(spec)
    bad = CompileResult(
        ok=True,
        source_path="",
        audit={
            "balanced_sync": False,
            "has_printf": True,
            "vf_calls": 1,
            "dma_copies": 2,
            "scalar_global_refs": 0,
            "notify": 2,
            "wait": 1,
        },
    )
    audit = build_lowering_audit(spec, plan, bad)
    assert audit["status"] == "failed"
    failed = {item["id"] for item in audit["checks"] if item["status"] == "fail"}
    assert {"sync-balanced", "no-debug-printf"} <= failed

    good = CompileResult(
        ok=True,
        source_path="",
        audit={
            "balanced_sync": True,
            "has_printf": False,
            "vf_calls": 1,
            "dma_copies": 2,
            "scalar_global_refs": 0,
            "notify": 2,
            "wait": 2,
        },
    )
    assert build_lowering_audit(spec, plan, good)["status"] == "passed"
    assert build_lowering_audit(spec, plan, None)["status"] == "unavailable"


def test_evidence_and_diagnosis_route_capacity_to_schedule(tmp_path):
    store = EvidenceStore(tmp_path, "task-x")
    store.record("design", "contract", "passed", facts={"contract_hash": "abc"})
    store.record(
        "compile",
        "compile_gate",
        "failed",
        category="capacity",
        facts={"error": "UB budget exceeded"},
    )
    store.record(
        "static",
        "static_gate",
        "failed",
        category="static",
        facts={"rules": ["API001"]},
    )
    diagnosis = diagnose(store.records)
    loops = {cluster["loop"] for cluster in diagnosis["clusters"]}
    assert "schedule" in loops
    assert "kernel" in loops
    assert diagnosis["failed_records"] == 2
    assert (tmp_path / "evidence.jsonl").is_file()


def test_generated_runner_has_a_case_matrix_and_keeps_legacy_api():
    spec = _softmax()
    plan = build_plan(spec)
    source = render_test(
        spec,
        plan,
        {"cann_home": "/x", "device_flags": [], "warmup": 0, "repeats": 1},
        "/repo",
    )
    ast.parse(source)
    assert "CASE_SPECS" in source
    for name in ("normal", "zeros", "ones", "large", "tiny"):
        assert f'"{name}"' in source
    assert "def make_inputs(device, case=None):" in source
    assert "[PRECISION_PASS]" in source
    assert "golden(" in source

    # exp() must not run the "large" case: exp(100) overflows both sides.
    exp = TaskSpec(
        "exp",
        "elementwise_unary",
        [TensorSpec("X", (16,), "float32")],
        [TensorSpec("Y", (16,), "float32")],
    )
    exp_src = render_test(
        exp,
        build_plan(exp),
        {"cann_home": "/x", "device_flags": [], "warmup": 0, "repeats": 1},
        "/repo",
    )
    assert '"large"' not in exp_src


def test_negative_probe_rejects_forbidden_debug_api(tmp_path):
    spec = _softmax()
    plan = build_plan(spec)
    write_kernel(spec, plan, tmp_path)
    probe = run_negative_probe(tmp_path, spec, plan, load_config())
    assert probe["status"] == "passed"
    assert probe["expected_rule"] == "SEM004"
    assert "SEM004" in probe["observed_rules"]
    assert (tmp_path / "negative_probe.json").is_file()
    # the delivered kernel must stay untouched
    assert "T.device_assert" not in (tmp_path / "softmax.py").read_text(encoding="utf-8")


def test_campaign_report_contains_all_six_steps():
    spec = _softmax()
    plan = build_plan(spec)
    contract = build_contract(spec, plan).to_dict()
    result = {
        "task_id": "softmax-test",
        "status": "success_static_only",
        "verified": False,
        "mode": "template",
        "generator": "template",
        "golden_source": "aiop-builtin",
        "elapsed_s": 1.0,
        "spec": spec.to_dict(),
        "plan": plan.to_dict(),
        "contract": contract,
        "skills": {"skills": [{"name": "aiop-op-pipeline", "reason": "x", "sha256": "a" * 64}]},
        "lowering_audit": {"status": "passed", "checks": []},
        "diagnosis": {"verdict": "no_failure", "clusters": []},
        "simulation": None,
        "board": None,
        "attempts": [],
        "knowledge": [],
        "notes": ["L3 unavailable"],
        "artifacts": {},
    }
    text = render_campaign_report(result)
    for marker in (
        "1. 算子契约",
        "2. 可复用技能",
        "3. 初步实现",
        "4. 仿真验证",
        "5. 硬件上板",
        "6. 对齐 some_ref.md",
        "TaskRegistry",
        "LoweringAudit",
    ):
        assert marker in text


@pytest.mark.slow
def test_campaign_search_archives_contract_frontier_and_audit(tmp_path):
    from aiop.agent.orchestrator import OperatorAgent

    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.simulation.backend = "static"
    spec = TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )
    result = OperatorAgent(cfg).run(spec, simulate=False, search=True, search_budget=2)
    assert result["status"] == "success_static_only"
    task_dir = tmp_path / "workspace" / result["task_id"]
    from aiop.agent.campaign import write_campaign_report

    write_campaign_report(task_dir, result)
    for name in (
        "contract.json",
        "skill.lock",
        "context_manifest.json",
        "frontier.json",
        "lowering_audit.json",
        "evidence.jsonl",
        "diagnosis.json",
        "campaign_report.md",
    ):
        assert (task_dir / name).is_file(), name
    frontier = json.loads((task_dir / "frontier.json").read_text(encoding="utf-8"))
    assert frontier["incumbent"] != "base" or frontier["candidates"][0]["status"] in (
        "passed",
        "failed",
    )
    assert result["contract"]["plan_facts"]["blocks"] == result["plan"]["blocks"]
    assert result["lowering_audit"]["status"] in ("passed", "warning")
    assert result["diagnosis"]["verdict"] == "no_failure"
    assert "6. 对齐 some_ref.md" in (task_dir / "campaign_report.md").read_text(encoding="utf-8")
