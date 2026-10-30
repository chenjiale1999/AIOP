"""Compile gate: lower + device-compile a kernel in an isolated worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Config
from .static import source_audit


@dataclass
class CompileResult:
    ok: bool
    source_path: str = ""
    error: str = ""
    category: str = "none"
    hint: str = ""
    audit: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source_path": self.source_path,
            "error": self.error,
            "category": self.category,
            "hint": self.hint,
            "audit": self.audit,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def classify(error: str) -> tuple[str, str]:
    lower = error.lower()
    if "floatn.h" in lower or "__float128" in lower:
        return "toolchain_host_headers", "check CANN version and device compile flags"
    if "unknown type name 'asc_" in lower or "use of undeclared identifier 'asc_" in lower:
        return "cann_too_old", "A5 codegen needs CANN 9.2+ headers/instructions"
    if "capacit" in lower or "exceed" in lower or "out of memory" in lower or "buffer size" in lower:
        return "capacity", "reduce tile/rows/num_stages or reuse buffers"
    if "unsupported" in lower or "not implemented" in lower or "cannot be lowered" in lower:
        return "lowering_unsupported", "rewrite the construct or fall back to a supported template"
    if "target context required" in lower:
        return "target_context", "run lowering inside Target('ascend')"
    if "syntax" in lower or "indentation" in lower:
        return "python_syntax", "fix generated Python syntax"
    if "device compilation failed" in lower:
        return "device_compile", "inspect the CCE source saved next to the report"
    return "compile_error", "inspect the compiler traceback"


def compile_kernel(module_path: Path, workdir: Path, cfg: Config, timeout: int | None = None) -> CompileResult:
    import time

    started = time.time()
    out_json = workdir / "compile_result.json"
    env = os.environ.copy()
    repo = str(Path(cfg.paths.tilelang_repo))
    pythonpath = [str(PROJECT_ROOT), repo, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([p for p in pythonpath if p])
    env["ASCEND_HOME_PATH"] = cfg.cann.home
    env["ASCEND_TOOLKIT_HOME"] = cfg.cann.home
    for rel in ("bin", "tools/ccec_compiler/bin"):
        path = str(Path(cfg.cann.home) / rel)
        env["PATH"] = path + os.pathsep + env.get("PATH", "")
    for rel in ("lib64", "x86_64-linux/lib64"):
        path = str(Path(cfg.cann.home) / rel)
        env["LD_LIBRARY_PATH"] = path + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        sys.executable,
        "-m",
        "aiop.runtime.compile_worker",
        "--module",
        str(module_path),
        "--out",
        str(out_json),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout or cfg.compile.timeout_s,
        )
        elapsed = time.time() - started
        if out_json.is_file():
            data = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            data = {"ok": False, "error": proc.stdout[-8000:] or "compile worker produced no result"}
        if data.get("ok"):
            source_path = data.get("source_path", "")
            source = Path(source_path).read_text(encoding="utf-8") if source_path else ""
            return CompileResult(
                ok=True,
                source_path=source_path,
                audit=source_audit(source),
                elapsed_s=elapsed,
            )
        error = data.get("error", "")
        category, hint = classify(error)
        return CompileResult(
            ok=False,
            error=error,
            category=category,
            hint=hint,
            elapsed_s=elapsed,
        )
    except subprocess.TimeoutExpired:
        category, hint = "timeout", "reduce shape or inspect compiler hang"
        return CompileResult(
            ok=False,
            error=f"compile timed out after {timeout or cfg.compile.timeout_s}s",
            category=category,
            hint=hint,
            elapsed_s=time.time() - started,
        )
