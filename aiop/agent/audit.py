"""LoweringAudit: compare the declared plan with the lowered CCE artifact.

The plan states intent (grid, threads, stages, epilogue, buffers); the lowered
source is the compiler's interpretation. This module records the facts that can
be observed from the generated CCE (VF calls, DMA copies, sync balance, debug
printf) and flags mismatches instead of assuming the plan was honored.

Reference: ``LoweringAudit`` in ``some_ref.md``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..checks.compiler import CompileResult
from ..checks.static import source_audit
from ..runtime.hardware import Plan
from ..specs import TaskSpec

SIMT_FAMILIES = {
    "elementwise_unary",
    "elementwise_binary",
    "row_reduce",
    "softmax",
    "rmsnorm",
    "layernorm",
}
DMA_VARIANTS = {
    "simt_ub_staged",
    "persistent_gemm",
    "persistent_gemm_epilogue",
}


def _check(cid: str, status: str, detail: str) -> dict[str, str]:
    return {"id": cid, "status": status, "detail": detail}


def build_lowering_audit(
    spec: TaskSpec,
    plan: Plan,
    compile_result: CompileResult | None,
) -> dict[str, Any]:
    declared = {
        "variant": plan.variant,
        "blocks": plan.blocks,
        "threads": plan.threads,
        "tile": plan.tile,
        "rows": plan.rows,
        "tile_extent": plan.tile_extent,
        "num_stages": plan.num_stages,
        "tail_mode": plan.tail_mode,
        "block_m": plan.block_m,
        "block_n": plan.block_n,
        "block_k": plan.block_k,
        "epilogue": plan.epilogue,
        "out_cast": plan.out_cast,
    }
    if compile_result is None or not compile_result.ok:
        return {
            "status": "unavailable",
            "declared": declared,
            "observed": {},
            "checks": [
                _check(
                    "artifact",
                    "unavailable",
                    "no successful device compilation; lowering facts cannot be audited",
                )
            ],
            "reason": (compile_result.error[-800:] if compile_result else "compile not run"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    observed = dict(compile_result.audit or {})
    if not observed and compile_result.source_path:
        source = Path(compile_result.source_path).read_text(encoding="utf-8")
        observed = source_audit(source)

    checks: list[dict[str, str]] = []
    checks.append(
        _check(
            "sync-balanced",
            "pass" if observed.get("balanced_sync") else "fail",
            f"notify={observed.get('notify', '?')} wait={observed.get('wait', '?')}",
        )
    )
    checks.append(
        _check(
            "no-debug-printf",
            "fail" if observed.get("has_printf") else "pass",
            "delivered kernel must not contain printf/asc_printf",
        )
    )
    if spec.family in SIMT_FAMILIES:
        vf = int(observed.get("vf_calls", 0))
        checks.append(
            _check(
                "vf-present",
                "pass" if vf > 0 else "warn",
                f"asc_vf_call={vf} for SIMT family {spec.family}",
            )
        )
    if plan.variant in DMA_VARIANTS:
        dma = int(observed.get("dma_copies", 0))
        checks.append(
            _check(
                "dma-present",
                "pass" if dma > 0 else "warn",
                f"asc_copy_gm2ub/ub2gm={dma} for variant {plan.variant}",
            )
        )
    scalar = int(observed.get("scalar_global_refs", 0))
    checks.append(
        _check(
            "scalar-global-refs",
            "warn" if scalar > 8 else "pass",
            f"scalar GM references={scalar}",
        )
    )

    if any(c["status"] == "fail" for c in checks):
        status = "failed"
    elif any(c["status"] == "warn" for c in checks):
        status = "warning"
    else:
        status = "passed"
    return {
        "status": status,
        "declared": declared,
        "observed": observed,
        "checks": checks,
        "source_path": compile_result.source_path,
        "elapsed_s": round(compile_result.elapsed_s, 2),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_lowering_audit(task_dir: str | Path, audit: dict[str, Any]) -> Path:
    path = Path(task_dir) / "lowering_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
