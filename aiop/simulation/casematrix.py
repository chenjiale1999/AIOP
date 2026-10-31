"""Legal shape/dtype case enumeration and per-case kernel variants.

A TileLang kernel is specialised to the concrete shapes and dtypes of its
``TaskSpec``.  Covering every legal ``(shape, dtype)`` combination therefore
requires one kernel variant per combination, not one kernel reused with
different tensors.  For manifest-bound template operators this module enumerates
all workload rows and dtype cases from the TileOPs manifest, binds each of them
into a ``TaskSpec``, and renders a dedicated kernel through the deterministic
template layer.

Operators without a manifest (or non-template kernels such as LLM/draft
implementations) only expose their own bound contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..generation import templates
from ..manifest import bind_operator, load_operator
from ..runtime.hardware import Plan, build_plan
from ..specs import TaskSpec


def input_key(spec: TaskSpec) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Identity of the input shape/dtype combination (coverage key)."""
    return tuple((tuple(t.shape), t.dtype) for t in spec.inputs)


def contract_key(spec: TaskSpec) -> tuple:
    """Identity of the full input+output contract (dedup key)."""
    inputs = tuple((tuple(t.shape), t.dtype) for t in spec.inputs)
    outputs = tuple((tuple(t.shape), t.dtype) for t in spec.outputs)
    return inputs, outputs


def describe_key(key: tuple[tuple[tuple[int, ...], str], ...]) -> list[dict]:
    return [{"shape": list(shape), "dtype": dtype} for shape, dtype in key]


@dataclass
class KernelVariant:
    """One concrete ``(shape, dtype)`` contract plus its kernel source."""

    spec: TaskSpec
    plan: Plan | None
    source: str
    golden_override: str | None = None
    label: str = ""
    workload: int = -1
    dtype_case: int = -1
    error: str = ""

    @property
    def available(self) -> bool:
        return bool(self.source) and self.plan is not None and not self.error


def bound_variant(
    spec: TaskSpec,
    plan: Plan,
    source: str,
    golden_override: str | None = None,
) -> KernelVariant:
    meta = spec.manifest or {}
    return KernelVariant(
        spec=spec,
        plan=plan,
        source=source,
        golden_override=golden_override,
        label="task-bound",
        workload=int(meta.get("workload_index", -1)),
        dtype_case=int(meta.get("dtype_case", -1)),
    )


def plan_variants(
    spec: TaskSpec,
    plan: Plan,
    kernel_source: str,
    *,
    golden_override: str | None = None,
    enumerate_variants: bool = True,
) -> list[KernelVariant]:
    """Return the bound variant plus every manifest variant when allowed.

    Enumeration is only attempted for specs carrying TileOPs manifest metadata.
    Variants that cannot be planned/rendered are returned with ``error`` set, so
    the generated test can report them as unavailable instead of silently
    dropping a legal combination.
    """
    bound = bound_variant(spec, plan, kernel_source, golden_override)
    if not enumerate_variants:
        return [bound]

    meta = spec.manifest or {}
    op_name = str(meta.get("op") or "")
    source_path = str(meta.get("source") or "")
    if not op_name or not source_path:
        return [bound]

    try:
        entry = load_operator(source_path, op_name)
    except Exception:
        return [bound]

    rows = list(entry.get("workloads") or [])
    if not rows:
        return [bound]

    variants: list[KernelVariant] = []
    seen: set[tuple] = set()
    bound_key = contract_key(spec)
    bound_added = False

    for workload_index, row in enumerate(rows):
        label = str(row.get("label") or f"workload-{workload_index}")
        dtype_cases = row.get("dtype_cases") or [{}]
        for dtype_index in range(len(dtype_cases)):
            try:
                case_spec = bind_operator(source_path, op_name, workload_index, dtype_index)
            except Exception:
                continue
            key = contract_key(case_spec)
            if key in seen:
                continue
            seen.add(key)
            if key == bound_key:
                variants.append(bound)
                bound_added = True
                continue
            try:
                case_plan = build_plan(case_spec)
                case_source = templates.render(case_spec, case_plan, None)
            except Exception as exc:  # noqa: BLE001 - reported to the test
                variants.append(
                    KernelVariant(
                        spec=case_spec,
                        plan=None,
                        source="",
                        label=label,
                        workload=workload_index,
                        dtype_case=dtype_index,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            variants.append(
                KernelVariant(
                    spec=case_spec,
                    plan=case_plan,
                    source=case_source,
                    label=label,
                    workload=workload_index,
                    dtype_case=dtype_index,
                )
            )

    if not bound_added and bound_key not in seen:
        variants.insert(0, bound)
    return variants
