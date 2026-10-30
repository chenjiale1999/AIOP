"""Ascend 950 (dav-3510) hardware facts and tiling planner.

Every constant carries its source. Values are scheduling budgets, not guesses:
over-budget plans are rejected before code generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..specs import DTYPE_BYTES, TaskSpec

# Source: src/ascend/transform/auto_schedule.cc
UB_PHYSICAL_BYTES = 248 * 1024
SIMT_CONTEXT_BYTES = 32 * 1024
UB_BUDGET_BYTES = 208 * 1024  # official rmsnorm example keeps 8 KiB headroom
L1_BUDGET_BYTES = 512 * 1024
L0C_BUDGET_BYTES = 256 * 1024
L0A_BUDGET_BYTES = 64 * 1024
L0B_BUDGET_BYTES = 64 * 1024
SS_BUDGET_BYTES = 3 * 1024

# Source: examples/ascend (vecadd/rmsnorm use 64 vector blocks, gemm uses 32 cube blocks)
AIV_CORES = 64
AIC_CORES = 32

VECTOR_BITS = 2048
ALIGN_BYTES = 32

PLAN_SOURCE_NOTE = (
    "src/ascend/transform/auto_schedule.cc (UB/L1/L0C/L0A/L0B) + "
    "examples/ascend/example_{vecadd,rmsnorm,gemm}.py (grid sizes)"
)

FACTS = {
    "soc": "Ascend 950PR/DT (Atlas A5)",
    "npu_arch": "3510",
    "compiler_arch": "dav-3510",
    "aic_cores": AIC_CORES,
    "aiv_cores": AIV_CORES,
    "ub_physical_bytes": UB_PHYSICAL_BYTES,
    "ub_budget_bytes": UB_BUDGET_BYTES,
    "simt_context_bytes": SIMT_CONTEXT_BYTES,
    "l1_bytes": L1_BUDGET_BYTES,
    "l0c_bytes": L0C_BUDGET_BYTES,
    "l0a_bytes": L0A_BUDGET_BYTES,
    "l0b_bytes": L0B_BUDGET_BYTES,
    "ss_bytes": SS_BUDGET_BYTES,
    "vector_bits": VECTOR_BITS,
    "align_bytes": ALIGN_BYTES,
}


def align_up(value: int, align: int = ALIGN_BYTES) -> int:
    return (value + align - 1) // align * align


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def largest_divisor(value: int, limit: int) -> int:
    """Largest divisor of ``value`` that is <= ``limit``."""
    limit = max(1, min(limit, value))
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


@dataclass
class Plan:
    family: str
    variant: str
    blocks: int
    tile: int = 0
    rows: int = 0
    threads: int = 0
    tile_extent: int = 0
    num_stages: int = 0
    iterations: int = 0
    block_m: int = 0
    block_n: int = 0
    block_k: int = 0
    epilogue: str = "none"
    tail_mode: str = "exact"
    buffer_bytes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "variant": self.variant,
            "blocks": self.blocks,
            "tile": self.tile,
            "rows": self.rows,
            "threads": self.threads,
            "tile_extent": self.tile_extent,
            "num_stages": self.num_stages,
            "iterations": self.iterations,
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "epilogue": self.epilogue,
            "tail_mode": self.tail_mode,
            "buffer_bytes": dict(self.buffer_bytes),
        }

    @property
    def ub_used(self) -> int:
        return sum(self.buffer_bytes.values())


def _elementwise_plan(spec: TaskSpec) -> Plan:
    dtype = spec.inputs[0].dtype
    elem = DTYPE_BYTES[dtype]
    target = 8192 if elem <= 2 else 4096
    total = spec.output.numel
    # Prefer an exact tile decomposition so the UB-staged path is legal.
    tile = largest_divisor(total, target)
    blocks = min(AIV_CORES, max(1, total // tile))
    blocks = largest_divisor(max(1, total // tile), blocks)
    exact = total % (blocks * tile) == 0
    if not exact or tile < 256 or blocks < 1:
        blocks = AIV_CORES
        tile = 8192 if elem <= 2 else 4096
        exact = False
    iterations = math.ceil(total / (blocks * tile))
    nbuf = len(spec.inputs) + len(spec.outputs)
    buffer_bytes = {f"tile{i}": tile * elem * 2 for i in range(nbuf)}
    return Plan(
        family=spec.family,
        variant="simt_ub_staged" if exact else "simt_direct_global",
        blocks=blocks,
        tile=tile,
        threads=min(256, tile),
        num_stages=2 if exact else 1,
        iterations=iterations,
        tail_mode="exact" if exact else "guarded",
        buffer_bytes=buffer_bytes if exact else {},
    )


def _row_plan(spec: TaskSpec) -> Plan:
    shape = spec.input.shape
    rows_total = spec.input.numel // shape[-1]
    d = shape[-1]
    extent = next_power_of_two(d)
    threads = 256 if extent > 4096 else 128
    threads = min(threads, max(32, extent))
    rows = 4
    # Keep whole row groups: avoid launching more row slots than there is work.
    while rows > 1 and AIV_CORES * rows > rows_total:
        rows //= 2
    # Keep two fp32 tiles + one weight tile within budget while allowing a
    # second buffer version for pipelining.
    while rows > 1 and (3 * rows + 2) * extent * 4 > UB_BUDGET_BYTES:
        rows //= 2
    per_iter = AIV_CORES * rows
    iterations = math.ceil(rows_total / per_iter)
    exact = rows_total % per_iter == 0
    buffer_bytes = {
        "x": rows * extent * 4 * 2,
        "acc": rows * 4 * 2,
    }
    if spec.family in ("rmsnorm", "layernorm"):
        buffer_bytes["w"] = align_up(d * max(4, DTYPE_BYTES[spec.inputs[1].dtype]))
    return Plan(
        family=spec.family,
        variant="simt_row_reducer",
        blocks=AIV_CORES,
        rows=rows,
        tile_extent=extent,
        threads=threads,
        num_stages=2,
        iterations=iterations,
        tail_mode="exact" if exact else "guarded",
        buffer_bytes=buffer_bytes,
    )


def _gemm_plan(spec: TaskSpec) -> Plan:
    a, b = spec.inputs
    m, k = a.shape
    n = b.shape[0]
    elem = DTYPE_BYTES[a.dtype]
    block_m = largest_divisor(m, 256)
    block_n = largest_divisor(n, 224)
    block_k = largest_divisor(k, 128)
    # Keep L0C and L1 within hardware budgets; shrink only what is needed.
    while block_m * block_n * 4 > L0C_BUDGET_BYTES and block_n > 1:
        block_n = largest_divisor(n, block_n // 2)
    while (block_m + block_n) * block_k * elem * 2 > L1_BUDGET_BYTES and block_k > 1:
        block_k = largest_divisor(k, block_k // 2)
    tiles = (m // block_m) * (n // block_n)
    blocks = min(AIC_CORES, max(1, tiles))
    epilogue = "relu" if spec.name == "gemm_relu" or spec.attrs.get("epilogue") == "relu" else "none"
    if epilogue != "none" and block_m % 2 != 0:
        block_m = largest_divisor(m, block_m - 1 if block_m > 1 else 1)
        if block_m % 2 != 0:
            epilogue = "none"
    buffer_bytes = {}
    if epilogue != "none":
        buffer_bytes["c_ub"] = (block_m // 2) * block_n * 4
    return Plan(
        family=spec.family,
        variant="persistent_gemm" if epilogue == "none" else "persistent_gemm_epilogue",
        blocks=blocks,
        num_stages=2,
        iterations=max(1, tiles // blocks),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        epilogue=epilogue,
        tail_mode="exact",
        buffer_bytes=buffer_bytes,
    )


def build_plan(spec: TaskSpec) -> Plan:
    issues = spec.validate()
    if issues:
        raise ValueError("; ".join(issues))
    if spec.family in ("elementwise_unary", "elementwise_binary"):
        plan = _elementwise_plan(spec)
    elif spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm"):
        plan = _row_plan(spec)
    elif spec.family == "gemm":
        plan = _gemm_plan(spec)
    else:  # pragma: no cover - guarded by TaskSpec
        raise ValueError(f"unsupported family {spec.family}")
    audit = check_plan(spec, plan)
    if audit["errors"]:
        raise ValueError("plan violates A5 constraints: " + "; ".join(audit["errors"]))
    return plan


def check_plan(spec: TaskSpec, plan: Plan) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if plan.blocks < 1 or plan.blocks > AIV_CORES:
        errors.append(f"blocks={plan.blocks} outside 1..{AIV_CORES}")
    if spec.family == "gemm":
        if plan.blocks > AIC_CORES:
            errors.append(f"cube grid {plan.blocks} exceeds {AIC_CORES}")
        l0c = plan.block_m * plan.block_n * 4
        if l0c > L0C_BUDGET_BYTES:
            errors.append(f"L0C {l0c}B exceeds {L0C_BUDGET_BYTES}B")
        l1 = (plan.block_m + plan.block_n) * plan.block_k * DTYPE_BYTES[spec.inputs[0].dtype]
        if l1 * plan.num_stages > L1_BUDGET_BYTES:
            errors.append(f"L1 {l1 * plan.num_stages}B exceeds {L1_BUDGET_BYTES}B")
        if plan.epilogue != "none" and plan.block_m % 2:
            errors.append("epilogue requires even block_m for dual_copy")
    else:
        if plan.threads <= 0 or plan.threads > 2048:
            errors.append(f"threads={plan.threads} outside 1..2048")
        if plan.tile_extent and plan.tile_extent > 65536:
            errors.append(f"row tile extent {plan.tile_extent} is not cacheable")
        if plan.threads and plan.tile_extent and plan.threads > plan.tile_extent:
            errors.append("threads exceed the parallel row extent")
    if plan.ub_used > UB_BUDGET_BYTES:
        errors.append(f"UB plan {plan.ub_used}B exceeds {UB_BUDGET_BYTES}B")
    if plan.tail_mode == "guarded" and spec.family in ("softmax", "rmsnorm", "layernorm"):
        warnings.append("guarded row tail: last iteration runs partially masked lanes")
    if plan.variant == "simt_direct_global":
        warnings.append(
            "shape is not tile-divisible: guarded scalar GM path (correctness first, lower bandwidth)"
        )
    return {"errors": errors, "warnings": warnings, "ub_used": plan.ub_used}


def roofline(spec: TaskSpec, latency_ms: float | None = None) -> dict[str, Any]:
    nbytes = spec.bytes()
    flops = spec.flops()
    result: dict[str, Any] = {
        "bytes": nbytes,
        "flops": flops,
        "arithmetic_intensity": round(spec.arithmetic_intensity(), 4),
        "bottleneck": "compute" if spec.arithmetic_intensity() > 64 else "memory",
    }
    if latency_ms and latency_ms > 0:
        result["achieved_bandwidth_gbps"] = round(nbytes / (latency_ms * 1e6), 3)
        result["achieved_tflops"] = round(flops / (latency_ms * 1e9), 6)
    return result
