"""Bounded plan search: propose, rank and gate candidate tiling plans.

The planner (``runtime.hardware``) produces one conservative plan. This module
implements the ``Planner`` + ``SearchController`` idea from ``some_ref.md``:
enumerate a small, family-aware neighborhood of legal plans (tile geometry,
grid, threads, stages, GEMM blocks), rank them with an explicit cost model, and
promote the best candidate that passes the static + device-compile gates.

Only *plans* are searched. Kernel semantics, precision and checks stay fixed.
Without a runnable device the ranking is a model estimate, not a measurement;
this is recorded in ``frontier.json``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..checks.compiler import compile_kernel
from ..checks.static import run_static
from ..config import Config
from ..generation.generator import generate
from ..runtime.hardware import (
    AIC_CORES,
    AIV_CORES,
    L0C_BUDGET_BYTES,
    UB_BUDGET_BYTES,
    Plan,
    align_up,
    check_plan,
    largest_divisor,
    next_power_of_two,
)
from ..specs import DTYPE_BYTES, TaskSpec


@dataclass
class Candidate:
    label: str
    plan: Plan
    score: float


def _valid_plan(spec: TaskSpec, plan: Plan) -> bool:
    try:
        return not check_plan(spec, plan)["errors"]
    except Exception:  # noqa: BLE001 - invalid candidates are simply skipped
        return False


def score_plan(spec: TaskSpec, plan: Plan) -> float:
    """Deterministic cost-model score (higher is better). Not a measurement."""
    if spec.family in ("elementwise_unary", "elementwise_binary"):
        score = 0.5
        if plan.variant == "simt_ub_staged":
            score += 0.4
        score += 0.3 * plan.blocks / AIV_CORES
        score += 0.1 * min(plan.num_stages, 2)
        if plan.tail_mode == "guarded":
            score -= 0.3
        if plan.iterations > 2:
            score -= 0.1 * (plan.iterations - 2)
        return round(score, 4)
    if spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm"):
        score = 0.5 + 0.3 * plan.rows / 4 + 0.2 * plan.threads / 256
        if plan.tail_mode == "guarded":
            score -= 0.2
        if plan.ub_used > 0.9 * UB_BUDGET_BYTES:
            score -= 0.2
        return round(score, 4)
    if spec.family == "gemm":
        m = spec.inputs[0].shape[0]
        n = spec.inputs[1].shape[0]
        tiles = (m // plan.block_m) * (n // plan.block_n)
        area = plan.block_m * plan.block_n / (256 * 224)
        score = 0.4 + 0.4 * min(area, 1.0) + 0.2 * plan.blocks / AIC_CORES
        if tiles % max(1, plan.blocks) == 0:
            score += 0.1
        if plan.block_k < 64:
            score -= 0.1
        return round(score, 4)
    return 0.0


def _candidate(label: str, spec: TaskSpec, plan: Plan) -> Candidate | None:
    if not _valid_plan(spec, plan):
        return None
    return Candidate(label=label, plan=plan, score=score_plan(spec, plan))


def _elementwise(spec: TaskSpec, base: Plan) -> list[Candidate]:
    total = spec.output.numel
    elem = DTYPE_BYTES[spec.inputs[0].dtype]
    nbuf = len(spec.inputs) + len(spec.outputs)
    tiles = sorted({base.tile, max(256, base.tile // 2), min(16384, base.tile * 2)})
    blocks_opts = sorted({base.blocks, AIV_CORES, 32}, reverse=True)
    out: list[Candidate] = []
    for tile in tiles:
        if tile < 1:
            continue
        for blocks in blocks_opts:
            if blocks < 1 or blocks > AIV_CORES:
                continue
            exact = total % (blocks * tile) == 0
            variant = "simt_ub_staged" if exact and tile >= 256 else "simt_direct_global"
            buffer_bytes = (
                {f"tile{i}": tile * elem * 2 for i in range(nbuf)}
                if variant == "simt_ub_staged"
                else {}
            )
            plan = Plan(
                family=spec.family,
                variant=variant,
                blocks=blocks,
                tile=tile,
                threads=min(256, tile),
                num_stages=2 if variant == "simt_ub_staged" else 1,
                iterations=math.ceil(total / (blocks * tile)),
                tail_mode="exact" if exact else "guarded",
                buffer_bytes=buffer_bytes,
            )
            cand = _candidate(f"e-t{tile}-b{blocks}", spec, plan)
            if cand:
                out.append(cand)
    return out


def _row(spec: TaskSpec, base: Plan) -> list[Candidate]:
    rows_total = spec.input.numel // spec.input.shape[-1]
    extent = next_power_of_two(spec.input.shape[-1])
    d = spec.input.shape[-1]
    rows_opts = sorted({base.rows, 2, 1}, reverse=True)
    threads_opts = sorted({base.threads, 128, 256}, reverse=True)
    out: list[Candidate] = []
    for rows in rows_opts:
        if rows < 1 or rows > max(1, rows_total):
            continue
        for threads in threads_opts:
            if threads < 32 or threads > extent:
                continue
            buffer_bytes = {"x": rows * extent * 4 * 2, "acc": rows * 4 * 2}
            if spec.family in ("rmsnorm", "layernorm"):
                buffer_bytes["w"] = align_up(
                    d * max(4, DTYPE_BYTES[spec.inputs[1].dtype])
                )
            per_iter = AIV_CORES * rows
            plan = Plan(
                family=spec.family,
                variant="simt_row_reducer",
                blocks=AIV_CORES,
                rows=rows,
                tile_extent=extent,
                threads=threads,
                num_stages=2,
                iterations=math.ceil(rows_total / per_iter),
                tail_mode="exact" if rows_total % per_iter == 0 else "guarded",
                buffer_bytes=buffer_bytes,
            )
            cand = _candidate(f"r-r{rows}-t{threads}", spec, plan)
            if cand:
                out.append(cand)
    return out


def _gemm(spec: TaskSpec, base: Plan) -> list[Candidate]:
    a, b = spec.inputs
    m, k = a.shape
    n = b.shape[0]
    elem = DTYPE_BYTES[a.dtype]
    bm_opts = {base.block_m, largest_divisor(m, max(16, base.block_m // 2))}
    if base.block_m * 2 <= 256:
        bm_opts.add(largest_divisor(m, base.block_m * 2))
    bn_opts = {base.block_n, largest_divisor(n, max(16, base.block_n // 2))}
    if base.block_n * 2 <= 224:
        bn_opts.add(largest_divisor(n, base.block_n * 2))
    bk_opts = {base.block_k, largest_divisor(k, max(16, base.block_k // 2))}
    out: list[Candidate] = []
    for bm in sorted(bm_opts, reverse=True):
        for bn in sorted(bn_opts, reverse=True):
            for bk in sorted(bk_opts, reverse=True):
                if bm * bn * 4 > L0C_BUDGET_BYTES:
                    continue
                tiles = (m // bm) * (n // bn)
                if tiles <= 0:
                    continue
                blocks = min(AIC_CORES, max(1, tiles))
                buffer_bytes: dict[str, int] = {}
                if base.epilogue != "none" or base.out_cast:
                    buffer_bytes["c_ub"] = (bm // 2) * bn * 4
                if base.out_cast:
                    buffer_bytes["c_out"] = (bm // 2) * bn * DTYPE_BYTES[spec.output.dtype]
                plan = Plan(
                    family="gemm",
                    variant=(
                        "persistent_gemm"
                        if base.epilogue == "none" and not base.out_cast
                        else "persistent_gemm_epilogue"
                    ),
                    blocks=blocks,
                    num_stages=base.num_stages,
                    iterations=max(1, tiles // blocks),
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    epilogue=base.epilogue,
                    out_cast=base.out_cast,
                    tail_mode="exact",
                    buffer_bytes=buffer_bytes,
                )
                cand = _candidate(f"g-m{bm}-n{bn}-k{bk}", spec, plan)
                if cand:
                    out.append(cand)
    return out


def propose_candidates(spec: TaskSpec, base: Plan) -> list[Candidate]:
    """Family-aware neighborhood around the base plan, best first."""
    if base.variant == "llm_open":
        return [Candidate("base-open", base, 0.0)]
    if spec.family in ("elementwise_unary", "elementwise_binary"):
        found = _elementwise(spec, base)
    elif spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm"):
        found = _row(spec, base)
    elif spec.family == "gemm":
        found = _gemm(spec, base)
    else:
        found = []
    found.append(Candidate("base", base, score_plan(spec, base)))
    deduped: dict[str, Candidate] = {}
    for cand in found:
        key = json.dumps(cand.plan.to_dict(), sort_keys=True)
        current = deduped.get(key)
        if current is None or cand.score > current.score:
            deduped[key] = cand
    return sorted(deduped.values(), key=lambda c: (-c.score, c.label))


def _notify(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:  # noqa: BLE001 - progress must not break a search
        pass


class SearchController:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def search(
        self,
        spec: TaskSpec,
        base: Plan,
        task_dir: str | Path,
        *,
        design: str = "",
        knowledge: str = "",
        budget: int = 4,
        on_event: Callable[[str], None] | None = None,
    ) -> tuple[Plan, Path]:
        task_dir = Path(task_dir)
        candidates = propose_candidates(spec, base)
        budget = max(1, int(budget))
        tried = candidates[:budget]
        _notify(on_event, f"proposed={len(candidates)} budget={budget}")
        records: list[dict[str, Any]] = []
        incumbent: Candidate | None = None
        for cand in tried:
            cdir = task_dir / "candidates" / cand.label
            cdir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            static_ok = False
            compile_ok = False
            category = "none"
            error = ""
            _notify(on_event, f"candidate={cand.label} score={cand.score}")
            try:
                kernel, generator = generate(
                    spec,
                    cand.plan,
                    cdir,
                    self.cfg,
                    design=design,
                    knowledge=knowledge,
                    strategy="template",
                )
                report = run_static(spec, cand.plan, kernel, self.cfg)
                (cdir / "static_report.json").write_text(
                    json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                static_ok = report.ok
                if static_ok:
                    result = compile_kernel(kernel, cdir, self.cfg)
                    compile_ok = result.ok
                    if not compile_ok:
                        category = result.category
                        error = result.error[-1200:]
                else:
                    category = "static"
                    error = "; ".join(
                        f.message for f in report.findings if f.severity == "error"
                    )[-1200:]
            except Exception as exc:  # noqa: BLE001 - a bad candidate is an outcome
                category = "generator"
                error = f"{type(exc).__name__}: {exc}"
            status = "passed" if (static_ok and compile_ok) else "failed"
            _notify(
                on_event,
                f"candidate={cand.label} status={status} static={static_ok} "
                f"compile={compile_ok} category={category}",
            )
            records.append(
                {
                    "label": cand.label,
                    "score": cand.score,
                    "status": status,
                    "static_ok": static_ok,
                    "compile_ok": compile_ok,
                    "category": category,
                    "error": error,
                    "elapsed_s": round(time.time() - started, 2),
                    "plan": cand.plan.to_dict(),
                }
            )
            if status == "passed" and incumbent is None:
                incumbent = cand
        for cand in candidates[budget:]:
            records.append(
                {
                    "label": cand.label,
                    "score": cand.score,
                    "status": "not_tried",
                    "static_ok": None,
                    "compile_ok": None,
                    "category": "none",
                    "error": "",
                    "elapsed_s": 0.0,
                    "plan": cand.plan.to_dict(),
                }
            )
        chosen = incumbent.plan if incumbent else base
        frontier = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "budget": budget,
            "objective": "model score (no runtime measurement on this host)",
            "base": base.to_dict(),
            "incumbent": incumbent.label if incumbent else "base",
            "chosen_plan": chosen.to_dict(),
            "candidates": records,
        }
        path = task_dir / "frontier.json"
        path.write_text(json.dumps(frontier, indent=2, ensure_ascii=False), encoding="utf-8")
        return chosen, path
