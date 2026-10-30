"""Static checks: syntax, API legality, hardware constraints, semantics.

These run before any compiler invocation so the agent gets precise, cheap
feedback. Every finding has a stable rule id used by the repair loop.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..runtime.hardware import ALIGN_BYTES, UB_BUDGET_BYTES, check_plan
from ..runtime import tilelang_a5
from ..specs import DTYPE_BYTES, TaskSpec


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
        snapshot = tilelang_a5.api_snapshot(cfg)
    except Exception as exc:  # noqa: BLE001 - API lint degrades to a warning
        report.warn("API000", f"cannot introspect tilelang API: {exc}")
        return report
    language = set(snapshot["language"])
    simd = set(snapshot["simd"])
    unknown: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name.startswith("T.simd."):
                attr = name.split(".", 2)[2].split(".")[0]
                if attr not in simd:
                    unknown.append(name)
            elif name.startswith("T.") and name.count(".") == 1:
                attr = name.split(".", 1)[1]
                if attr not in language and attr not in {"prim_func"}:
                    unknown.append(name)
    if unknown:
        for name in sorted(set(unknown)):
            suggestion = difflib.get_close_matches(name.split(".")[-1], language, n=1)
            hint = f"; did you mean T.{suggestion[0]}?" if suggestion else ""
            report.error("API001", f"unknown Ascend dialect API `{name}`{hint}")
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
        if spec.output.dtype != "float32":
            report.error("HW005", "gemm template requires a float32 accumulator/output")
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
            f"reducer lifecycle mismatch: alloc={alloc} init={init} finalize={final}",
        )
    if spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm") and alloc == 0:
        report.error("SEM002", "row-wise family must use T.alloc_reducer")
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
