"""Simulation orchestration: npusim / local NPU / static-only fallback."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Config
from ..runtime import tilelang_a5
from ..runtime.command_log import CommandLogger
from ..runtime.hardware import roofline
from ..specs import TaskSpec


@dataclass
class SimResult:
    status: str  # passed | failed | unavailable | error
    backend: str = "static"
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "result": self.result,
            "error": self.error,
            "notes": self.notes,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _runtime_site() -> str:
    site = PROJECT_ROOT / ".runtime" / "site"
    return str(site) if site.is_dir() else ""


def select_backend(
    cfg: Config, command_log: CommandLogger | None = None
) -> tuple[str, list[str]]:
    """Pick a usable backend; unavailable ones are skipped, never forced."""
    notes: list[str] = []
    requested = cfg.simulation.backend
    if requested not in ("auto", "npusim", "local_npu", "static"):
        notes.append(f"unknown simulation.backend={requested!r}; using auto")
        requested = "auto"
    if requested == "static":
        return "static", notes
    if requested in ("auto", "local_npu"):
        try:
            import torch  # noqa: F401
            import torch_npu  # noqa: F401

            if torch.npu.is_available():
                return "local_npu", notes
            notes.append("torch_npu imported but no NPU device is visible")
        except Exception as exc:  # noqa: BLE001 - probe
            notes.append(f"torch_npu unavailable: {type(exc).__name__}")
        if requested == "local_npu":
            notes.append("simulation.backend=local_npu is not usable here; falling back")
    if Path(cfg.npusim_bin).is_file():
        site = _runtime_site()
        site_ok = False
        probe_cmd = [
            sys.executable,
            "-c",
            "import torch, torch_npu; print(torch.npu.is_available())",
        ]
        if site:
            env = os.environ.copy()
            env["PYTHONPATH"] = site + os.pathsep + env.get("PYTHONPATH", "")
            if command_log is not None:
                command_log.log(
                    "simulate.probe",
                    probe_cmd,
                    timeout_s=120,
                    context={"backend": "npusim", "runtime_site": site},
                )
            probe = subprocess.run(
                probe_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
            site_ok = probe.returncode == 0
            if not site_ok:
                tail = (probe.stdout or "").strip().splitlines()[-1:] or ["probe failed"]
                notes.append(f"torch_npu runtime probe failed: {tail[0]}")
        if site_ok:
            return "npusim", notes
        notes.append(f"npusim found at {cfg.npusim_bin}, but no usable torch_npu runtime")
    else:
        notes.append(f"npusim not found at {cfg.npusim_bin}")
    if requested == "npusim":
        notes.append("simulation.backend=npusim is not usable here; skipping simulation")
    return "static", notes


def run_simulation(
    task_dir: Path,
    test_path: Path,
    spec: TaskSpec,
    cfg: Config,
    *,
    compile_audit: dict[str, Any] | None = None,
    backend: str | None = None,
    backend_notes: list[str] | None = None,
    command_log: CommandLogger | None = None,
) -> SimResult:
    started = time.time()
    if backend is None:
        backend, notes = select_backend(cfg, command_log)
    else:
        notes = list(backend_notes or [])
    if backend == "static":
        result = roofline(spec)
        result.update(
            validated="lowering+device-compile only",
            reason="no functional simulator is usable in this environment",
        )
        if compile_audit:
            result["cce_audit"] = compile_audit
        return SimResult(
            status="unavailable",
            backend="static",
            result=result,
            notes=notes,
            elapsed_s=time.time() - started,
        )

    json_path = task_dir / "sim_result.json"
    if json_path.exists():
        json_path.unlink()

    if backend == "local_npu":
        cmd = [
            sys.executable,
            str(test_path),
            "--json",
            str(json_path),
            "--warmup",
            str(cfg.simulation.warmup),
            "--repeats",
            str(cfg.simulation.repeats),
        ]
        env = os.environ.copy()
        if cfg.cann.home:
            env["ASCEND_HOME_PATH"] = cfg.cann.home
        if command_log is not None:
            command_log.log(
                "simulate.local_npu",
                cmd,
                timeout_s=cfg.simulation.timeout_s,
                context={
                    "task_dir": str(task_dir),
                    "test": str(test_path),
                    "backend": "local_npu",
                    "result": str(json_path),
                    "warmup": cfg.simulation.warmup,
                    "repeats": cfg.simulation.repeats,
                    "ascend_home_path": env.get("ASCEND_HOME_PATH", ""),
                },
            )
        proc = subprocess.run(
            cmd,
            cwd=str(task_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=cfg.simulation.timeout_s,
        )
        output = proc.stdout
    else:
        sim_out = task_dir / "npusim"
        if sim_out.exists():
            shutil.rmtree(sim_out)
        sim_out.mkdir(parents=True, exist_ok=True)
        site = _runtime_site()
        app = (
            f"{sys.executable} {test_path} --json {json_path} "
            f"--warmup {cfg.simulation.warmup} --repeats {cfg.simulation.repeats}"
        )
        cmd = [cfg.npusim_bin, "record", "-s", cfg.cann.soc_version, "-o", str(sim_out), app]
        env = os.environ.copy()
        if site:
            env["PYTHONPATH"] = site + os.pathsep + env.get("PYTHONPATH", "")
        if cfg.cann.home:
            env["ASCEND_HOME_PATH"] = cfg.cann.home
        if command_log is not None:
            command_log.log(
                "simulate.npusim",
                cmd,
                timeout_s=cfg.simulation.timeout_s,
                context={
                    "task_dir": str(task_dir),
                    "test": str(test_path),
                    "backend": "npusim",
                    "soc_version": cfg.cann.soc_version,
                    "output": str(sim_out),
                    "result": str(json_path),
                    "app": app,
                    "warmup": cfg.simulation.warmup,
                    "repeats": cfg.simulation.repeats,
                },
            )
        proc = subprocess.run(
            cmd,
            cwd=str(task_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=cfg.simulation.timeout_s,
        )
        output = proc.stdout
        (task_dir / "npusim.log").write_text(output, encoding="utf-8")

    elapsed = time.time() - started
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        data = {
            "status": "error",
            "error": f"runner produced no JSON (exit={proc.returncode})",
            "output_tail": output[-3000:],
        }
    status = str(data.get("status", "error"))
    sim = SimResult(
        status=status if status in ("passed", "failed", "unavailable", "error") else "error",
        backend=backend,
        result=data,
        error=str(data.get("error", "")),
        notes=notes,
        elapsed_s=elapsed,
    )
    if sim.passed:
        latency = float(data.get("latency_ms") or 0.0)
        sim.result["roofline"] = roofline(spec, latency if latency > 0 else None)
        sim.result["perf_note"] = (
            "simulator timing is relative; use board measurement for real latency"
            if backend == "npusim"
            else "device timing"
        )
    return sim
