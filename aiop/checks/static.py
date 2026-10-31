"""Static checks: syntax, API legality, hardware constraints, semantics.

These run before any compiler invocation so the agent gets precise, cheap
feedback. Every finding has a stable rule id used by the repair loop.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import ascend_ops
from ..config import Config
from ..runtime.hardware import ALIGN_BYTES, UB_BUDGET_BYTES, check_plan
from ..specs import DTYPE_BYTES, TaskSpec

# A3-era (tilelang_ascend) names that the merged A5 main repo no longer exports.
# Value is the A5 replacement; names missing from A3_RENAMED_OPS and listed in
# A3_UNAVAILABLE_OPS have no A5 equivalent ("A5架构暂无", see knowledge/12).
A3_RENAMED_OPS = {
    "printf": "T.print",
    "mma": "T.gemm",
    "gemm_v0": "T.gemm",
    "alloc_L1": "T.alloc_l1",
    "alloc_L0A": "T.alloc_l0a",
    "alloc_L0B": "T.alloc_l0b",
    "alloc_L0C": "T.alloc_l0c",
    "alloc_ub": "T.alloc_shared",
    "set_cross_flag": "T.ascend_cross_core_set_flag",
    "wait_cross_flag": "T.ascend_cross_core_wait_flag",
    "set_flag": "T.ascend_set_flag",
    "wait_flag": "T.ascend_wait_flag",
    "barrier_all": 'T.ascend_pipe_barrier("PIPE_ALL")',
    "pipe_barrier": "T.ascend_pipe_barrier",
}
A3_UNAVAILABLE_OPS = {
    "sync_all",
    "dump_tensor",
    "reinterpretcast",
    "set_deq_scale",
    "shmem_put_nbi",
    "shmem_ub_put_nbi",
    "shmem_get_nbi",
    "shmem_ub_get_nbi",
}
A3_NAMESPACES = {
    "T.tile.": "A3 的 T.tile.* 命名空间已移除；SIMD 改用 T.simd.*，高层运算改用 T.Parallel/表达式",
    "T.ascend.": "A3 的 T.ascend.* 命名空间已移除；改用 A5 的 T.ascend_* 平铺接口",
}


def _a3_alias_message(dotted: str) -> str | None:
    """Return a standardization message when ``dotted`` is an A3 legacy name."""
    for prefix, message in A3_NAMESPACES.items():
        if dotted.startswith(prefix):
            return message
    if dotted.count(".") == 1:
        attr = dotted.split(".", 1)[1]
        if attr in A3_RENAMED_OPS:
            return f"A3 残留别名，A5 请改用 {A3_RENAMED_OPS[attr]}"
        if attr in A3_UNAVAILABLE_OPS:
            return "A3 专属接口，A5架构暂无等价实现（见 knowledge/12-op-compatibility.md）"
    return None


@dataclass
class Finding:
    rule: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "message": self.message, "severity": self.severity}


@dataclass
class Report:
    stage: str
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not [f for f in self.findings if f.severity == "error"]

    def error(self, rule: str, message: str) -> None:
        self.findings.append(Finding(rule, message, "error"))

    def warn(self, rule: str, message: str) -> None:
        self.findings.append(Finding(rule, message, "warning"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
            "metadata": self.metadata,
        }


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def check_syntax(kernel_path: Path) -> Report:
    report = Report("syntax")
    if not kernel_path.is_file():
        report.error("SYN001", f"kernel file not found: {kernel_path}")
        return report
    source = kernel_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report.error("SYN002", f"Python syntax error at line {exc.lineno}: {exc.msg}")
        return report
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "build" not in funcs:
        report.error("SYN003", "missing `build()` entry point")
    else:
        build = funcs["build"]
        has_prim = any(
            isinstance(n, ast.Attribute) and _dotted(n) in ("T.prim_func", "tilelang.prim_func")
            for n in ast.walk(build)
        )
        if not has_prim:
            report.error("SYN004", "`build()` does not define a @T.prim_func")
        if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(build)):
            report.error("SYN005", "`build()` does not return the prim_func")
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = imported.intersection({"torch", "torch_npu", "numpy", "requests", "socket", "subprocess"})
    if forbidden:
        report.error("SYN006", f"kernel must stay pure; forbidden imports: {sorted(forbidden)}")
    for pat, rule, msg in (
        (r"\bTODO\b|\bFIXME\b|placeholder", "SYN007", "generated code contains a placeholder"),
        (r"\bopen\s*\(", "SYN008", "kernel performs file IO"),
        (r"\beval\s*\(|\bexec\s*\(", "SYN009", "kernel uses eval/exec"),
    ):
        if re.search(pat, source):
            report.error(rule, msg)
    report.metadata["lines"] = len(source.splitlines())
    return report


def check_api(kernel_path: Path, cfg: Config) -> Report:
    report = Report("api")
    source = kernel_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    try:
        scope = ascend_ops.load_scope(cfg)
    except Exception as exc:  # noqa: BLE001 - API lint degrades to a warning
        report.warn("API000", f"cannot build the Ascend op scope: {exc}")
        return report
    report.metadata["scope"] = scope.to_dict()
    for violation in ascend_ops.scan_tree(tree, scope, ascend_ops.ascend_aliases(tree)):
        # Normalize any import alias back to T for the A3 compatibility map.
        normalized = "T." + violation.op.split(".", 1)[1]
        alias = _a3_alias_message(normalized)
        if alias is not None:
            report.error("API003", f"A3 legacy API `{violation.op}`: {alias}")
        elif violation.kind == "namespace":
            report.error("API004", f"`{violation.op}` is outside the Ascend dialect scope: {violation.detail}")
        elif violation.kind == "simd_unknown":
            suggestion = scope.suggest(violation.op.rsplit(".", 1)[-1], scope.simd)
            hint = f"; did you mean T.simd.{suggestion[0]}?" if suggestion else ""
            report.error(
                "API004",
                f"`{violation.op}` is not defined in {scope.language_dir}/simd.py{hint}",
            )
        else:
            suggestion = scope.suggest(violation.op.split(".", 1)[1], scope.declared, n=1)
            hint = f"; did you mean T.{suggestion[0]}?" if suggestion else ""
            report.error(
                "API001",
                f"unknown Ascend dialect API `{violation.op}`{hint}; "
                "legal ops are the definitions under tilelang/ascend/language",
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _dotted(node.func) == "T.Kernel":
            if any(kw.arg == "threads" for kw in node.keywords):
                report.error("API002", "T.Kernel does not accept threads=; move it into T.SimtVF")
    return report


def check_constraints(spec: TaskSpec, plan, kernel_path: Path) -> Report:
    report = Report("constraints")
    issues = spec.validate()
    for issue in issues:
        report.error("HW001", issue)
    audit = check_plan(spec, plan)
    for err in audit["errors"]:
        report.error("HW002", err)
    for warn in audit["warnings"]:
        report.warn("HW003", warn)
    for tensor in spec.inputs + spec.outputs:
        if tensor.dtype not in DTYPE_BYTES:
            report.error("HW004", f"{tensor.name}: dtype {tensor.dtype} is not supported on A5")
    if spec.family == "gemm":
        if spec.output.dtype not in ("float32", "float16", "bfloat16"):
            report.error("HW005", f"gemm output dtype {spec.output.dtype} has no L0C->UB cast path")
        if spec.output.dtype != "float32" and not getattr(plan, "out_cast", False):
            report.error("HW005", "gemm non-fp32 output requires the UB cast epilogue in the plan")
        for tensor in spec.inputs:
            if tensor.dtype not in ("float16", "bfloat16", "float32"):
                report.error("HW006", f"gemm input dtype {tensor.dtype} is not supported by the L1 path")
        m, k = spec.inputs[0].shape
        n = spec.inputs[1].shape[0]
        for label, value, block in (("M", m, plan.block_m), ("N", n, plan.block_n), ("K", k, plan.block_k)):
            if value % block:
                report.error("HW007", f"{label}={value} is not divisible by block {block}")
        if (plan.block_m % 16 or plan.block_n % 16 or plan.block_k % 16) and min(m, n, k) >= 16:
            report.warn("HW008", "GEMM block is not 16-aligned; hardware may pad and waste L0")
    if plan.variant == "simt_ub_staged":
        row_bytes = plan.tile * DTYPE_BYTES[spec.inputs[0].dtype]
        if row_bytes % ALIGN_BYTES:
            report.warn("HW009", f"tile row is {row_bytes}B, not a multiple of {ALIGN_BYTES}B")
    if plan.ub_used > UB_BUDGET_BYTES:
        report.error("HW010", f"UB plan {plan.ub_used}B exceeds budget {UB_BUDGET_BYTES}B")
    if plan.tail_mode == "guarded" and spec.family == "gemm":
        report.error("HW011", "gemm template does not support guarded tails")
    return report


def check_semantics(spec: TaskSpec, plan, kernel_path: Path) -> Report:
    report = Report("semantics")
    source = kernel_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    dotted = [_dotted(c.func) for c in calls]
    alloc = dotted.count("T.alloc_reducer")
    init = dotted.count("T.reducer_init")
    final = dotted.count("T.finalize_reducer")
    if not (alloc == init == final):
        report.error(
            "SEM001",
            f"reducer lifecycle mismatch: alloc={alloc} init={init} finalize={final}. "
            "Required lifecycle: acc = T.alloc_reducer(shape, dtype, op='sum'|'max'|'min'); "
            "T.reducer_init(acc); T.reducer_update(acc[i], value); "
            "out = T.alloc_fragment(shape, dtype); T.finalize_reducer(acc, out); read out[i]. "
            "There is no T.reduce_init / T.reduce_finalize / T.reducer_finalize / T.alloc_reduce.",
        )
    if spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm") and alloc == 0:
        report.error("SEM002", "row-wise family must use T.alloc_reducer")
    reducer_vars = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _dotted(node.value.func) == "T.alloc_reducer"
        ):
            reducer_vars.update(t.id for t in node.targets if isinstance(t, ast.Name))
    if reducer_vars:
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        reduce_ops = {
            "T.reduce",
            "T.reduce_sum",
            "T.reduce_max",
            "T.reduce_min",
            "T.reduce_abssum",
            "T.reduce_absmax",
            "T.reduce_bitand",
            "T.reduce_bitor",
            "T.reduce_bitxor",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted(node.func) in reduce_ops:
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Name) and node.args[1].id in reducer_vars:
                    report.error(
                        "SEM018",
                        f"`{_dotted(node.func)}` out must be a fragment/buffer, not reducer "
                        f"`{node.args[1].id}`; contribute with T.reducer_update instead",
                    )
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in reducer_vars:
                parent = parents.get(node)
                allowed = (
                    isinstance(parent, ast.Call)
                    and _dotted(parent.func) == "T.reducer_update"
                    and bool(parent.args)
                    and parent.args[0] is node
                )
                if not allowed:
                    report.error(
                        "SEM018",
                        f"reducer `{node.value.id}` cannot be read or assigned directly; "
                        "T.finalize_reducer(acc, out) and read `out` instead",
                    )
    for forbidden, rule in (
        ("T.print", "SEM003"),
        ("T.device_assert", "SEM004"),
        ("T.annotate_unlimit_memory", "SEM005"),
    ):
        if forbidden in source:
            report.error(rule, f"{forbidden} must not appear in a delivered kernel")
    if spec.family == "gemm":
        for needle, rule, message in (
            ("T.gemm(", "SEM006", "GEMM template must call T.gemm"),
            ("transpose_B=True", "SEM007", "L1 GEMM path requires transpose_B=True"),
            ("clear_accum=(kk == 0)", "SEM008", "K loop must clear the accumulator on the first step"),
            ("T.Persistent", "SEM009", "GEMM template must use the persistent schedule"),
        ):
            if needle not in source:
                report.error(rule, message)
        if getattr(plan, "out_cast", False) and "T.dual_copy" not in source:
            report.error(
                "SEM017",
                "non-fp32 GEMM output must cast L0C through UB (T.dual_copy) before the GM store",
            )
    if spec.family == "softmax" and "T.exp(" not in source:
        report.error("SEM010", "softmax kernel does not compute exp")
    if plan.tail_mode == "guarded":
        if spec.family in ("elementwise_unary", "elementwise_binary") and "if idx < N" not in source:
            report.error("SEM011", "guarded elementwise path is missing the idx < N guard")
        if spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm") and "base_row + r < BATCH" not in source:
            report.error("SEM012", "guarded row path is missing the row guard")
    if spec.family in ("rmsnorm", "layernorm") and "T.rsqrt" not in source:
        report.error("SEM013", "normalization kernel does not compute rsqrt")
    if spec.family == "layernorm" and "T.max(var[r], T.float32(0))" not in source:
        report.warn("SEM014", "layernorm variance is not clamped; tiny negative values can produce NaN")
    if plan.blocks > 64:
        report.error("SEM015", f"grid {plan.blocks} exceeds the 64 AIV limit")
    if "for i in T.Parallel" in source and plan.threads and plan.threads > 2048:
        report.error("SEM016", "thread domain exceeds hardware limit")
    return report


def run_static(spec: TaskSpec, plan, kernel_path: Path, cfg: Config) -> Report:
    combined = Report("static")
    for report in (
        check_syntax(kernel_path),
        check_api(kernel_path, cfg),
        check_constraints(spec, plan, kernel_path),
        check_semantics(spec, plan, kernel_path),
    ):
        combined.findings.extend(report.findings)
        combined.metadata[report.stage] = report.to_dict()
    return combined


def source_audit(source: str) -> dict[str, Any]:
    """Audit the generated CCE source for structural red flags."""
    scalar_gm = len(re.findall(r"(?<!align\()\b[A-Z]\w*\[", source))
    return {
        "chars": len(source),
        "notify": source.count("asc_sync_notify"),
        "wait": source.count("asc_sync_wait"),
        "vf_calls": source.count("asc_vf_call"),
        "dma_copies": source.count("asc_copy_gm2ub") + source.count("asc_copy_ub2gm"),
        "scalar_global_refs": scalar_gm,
        "has_printf": "printf" in source or "asc_printf" in source,
        "balanced_sync": source.count("asc_sync_notify") == source.count("asc_sync_wait"),
    }


def dump_report(report: Report, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
