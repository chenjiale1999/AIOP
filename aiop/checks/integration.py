"""TileOPs integration gate for generated Ascend A5 kernels.

A generated kernel is only useful if it can live in the upstream repository:
it must follow the ``Ascend<Stem>Kernel`` naming, register itself for the
``ascend`` target, depend on the repository's kernel base classes, and land in
the mirrored ``kernels/ascend/<family>/`` directory. The source of truth for
those conventions is the verified Ascend reference discovered by
:mod:`aiop.references`.
"""

from __future__ import annotations

import ast
import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..references import ReferencePair, op_stem
from ..specs import TaskSpec

_REGISTER_RE = re.compile(
    r"register_kernel_type\(\s*[\"']([^\"']+)[\"']\s*,\s*[\"']([^\"']+)[\"']\s*,\s*([A-Za-z_]\w*)\s*\)"
)


@dataclass
class IntegrationCheck:
    rule: str
    status: str  # passed | failed | warning | skipped
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "status": self.status, "message": self.message}


@dataclass
class IntegrationReport:
    op: str
    ready: bool
    class_name: str
    target_path: str
    checks: list[IntegrationCheck] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    needs_test: bool = False

    @property
    def errors(self) -> list[IntegrationCheck]:
        return [check for check in self.checks if check.status == "failed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "ready": self.ready,
            "class_name": self.class_name,
            "target_path": self.target_path,
            "needs_test": self.needs_test,
            "checks": [check.to_dict() for check in self.checks],
            "fixes": list(self.fixes),
        }


def expected_class_name(op: str) -> str:
    stem = op_stem(op)
    return f"Ascend{stem[:1].upper()}{stem[1:]}Kernel"


@functools.lru_cache(maxsize=64)
def _has_test(repo_text: str, op_name: str) -> bool:
    repo = Path(repo_text)
    tests = repo / "tests"
    if not tests.is_dir():
        return False
    needles = (op_name, op_stem(op_name))
    for path in tests.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(needle in text for needle in needles):
            return True
    return False


def check_integration(
    spec: TaskSpec,
    kernel_path: Path,
    pair: ReferencePair | None,
    cfg: Config,
) -> IntegrationReport:
    """Static integration gate; no repository files are modified."""
    class_name = expected_class_name(spec.name)
    target = pair.target_path() if pair and pair.repo else ""
    report = IntegrationReport(
        op=spec.name,
        ready=False,
        class_name=class_name,
        target_path=target,
    )

    source = ""
    if not kernel_path.is_file():
        report.checks.append(
            IntegrationCheck("INT001", "failed", f"kernel file not found: {kernel_path}")
        )
        report.fixes.append("generate the kernel before integration")
        return report
    source = kernel_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report.checks.append(
            IntegrationCheck("INT001", "failed", f"syntax error at line {exc.lineno}: {exc.msg}")
        )
        report.fixes.append("fix the Python syntax before integration")
        return report
    report.checks.append(IntegrationCheck("INT001", "passed", "kernel parses as Python"))

    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "build" in functions:
        report.checks.append(
            IntegrationCheck(
                "INT000",
                "passed",
                "keeps a top-level build() entry for the aiop compile/simulation gates",
            )
        )
    else:
        report.checks.append(
            IntegrationCheck(
                "INT000",
                "failed",
                "missing top-level def build(); the TileOPs class alone cannot be gated",
            )
        )
        report.fixes.append(
            "keep `def build()` returning the @T.prim_func and reuse it from the "
            "Ascend<Kernel> class"
        )

    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if class_name in classes:
        report.checks.append(
            IntegrationCheck("INT002", "passed", f"defines class {class_name}")
        )
    else:
        report.checks.append(
            IntegrationCheck(
                "INT002",
                "failed",
                f"missing class {class_name}; found classes: {sorted(classes) or 'none'}",
            )
        )
        report.fixes.append(
            f"wrap the kernel as class {class_name}(Kernel, <Family>Interface) like the "
            f"Ascend reference {pair.ascend.class_name if pair and pair.ascend else '<ascend peer>'}"
        )

    registrations = [
        (match.group(1), match.group(2), match.group(3))
        for match in _REGISTER_RE.finditer(source)
    ]
    matching = [
        reg
        for reg in registrations
        if reg[0] == spec.name and reg[1] == "ascend" and reg[2] == class_name
    ]
    if matching:
        report.checks.append(
            IntegrationCheck(
                "INT003",
                "passed",
                f'register_kernel_type("{spec.name}", "ascend", {class_name})',
            )
        )
    else:
        report.checks.append(
            IntegrationCheck(
                "INT003",
                "failed",
                "missing register_kernel_type registration for the ascend target",
            )
        )
        report.fixes.append(
            f'add `register_kernel_type("{spec.name}", "ascend", {class_name})` '
            "after the class definition"
        )

    if "tileops.kernels.kernel_base" in source or "from tileops.kernels" in source:
        report.checks.append(
            IntegrationCheck("INT004", "passed", "imports the TileOPs kernel layer")
        )
    else:
        report.checks.append(
            IntegrationCheck(
                "INT004",
                "warning",
                "does not import tileops.kernels.kernel_base; integration is not verified",
            )
        )

    if target:
        parent = Path(target).parent
        if parent.is_dir():
            report.checks.append(
                IntegrationCheck("INT005", "passed", f"target directory exists: {parent}")
            )
        else:
            report.checks.append(
                IntegrationCheck(
                    "INT005",
                    "warning",
                    f"target directory will be created: {parent}",
                )
            )
        report.needs_test = not _has_test(str(pair.repo), spec.name)
        if report.needs_test:
            report.checks.append(
                IntegrationCheck(
                    "INT006",
                    "warning",
                    f"no upstream test references {spec.name}; add a tests/ops case",
                )
            )
            report.fixes.append(
                f"add a tests/ops case for {spec.name} covering semantics and 1e-5 parity"
            )
        else:
            report.checks.append(
                IntegrationCheck("INT006", "passed", "an upstream test references this op")
            )
    else:
        report.checks.append(
            IntegrationCheck(
                "INT006", "skipped", "reference repository not configured; target path unknown"
            )
        )

    report.ready = not report.errors
    return report


def write_integration(task_dir: str | Path, report: IntegrationReport) -> Path:
    path = Path(task_dir) / "integration.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
