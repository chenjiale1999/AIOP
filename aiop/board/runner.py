"""Cross-server A5 board deployment and measurement."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..specs import TaskSpec


@dataclass
class BoardResult:
    status: str  # passed | failed | blocked | skipped
    host: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "host": self.host,
            "result": self.result,
            "error": self.error,
            "notes": self.notes,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _ssh_base(cfg: Config, host: str) -> list[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", str(cfg.board.port)]
    if cfg.board.key:
        cmd += ["-i", str(Path(cfg.board.key).expanduser())]
    cmd.append(host)
    return cmd


def _scp_base(cfg: Config) -> list[str]:
    cmd = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", str(cfg.board.port)]
    if cfg.board.key:
        cmd += ["-i", str(Path(cfg.board.key).expanduser())]
    return cmd


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired:
        return 124, f"command timed out after {timeout}s: {' '.join(cmd[:4])} ..."
    except FileNotFoundError as exc:
        return 127, f"command not found: {exc}"


def preflight(cfg: Config) -> tuple[bool, list[str]]:
    host = cfg.board.host
    notes: list[str] = []
    if not host:
        return False, ["board.host is empty; board measurement is skipped"]
    code, out = _run(_ssh_base(cfg, host) + ["true"], 30)
    if code != 0:
        return False, [f"ssh not usable: {out.strip()[-500:]}"]
    checks = {
        "npu-smi": "npu-smi info | head -5",
        "torch_npu": f"{cfg.board.python} -c 'import torch_npu; print(torch_npu.__version__)'",
        "tilelang": (
            f"{cfg.board.python} -c 'import sys; sys.path.insert(0, \"{cfg.board.tilelang_repo}\"); "
            "import tilelang; print(tilelang.__file__)'"
        )
        if cfg.board.tilelang_repo
        else "true",
        "disk": f"df -h {shlex.quote(cfg.board.remote_dir)} | tail -1",
    }
    for name, command in checks.items():
        env_prefix = f"{cfg.board.env_script} && " if cfg.board.env_script else ""
        code, out = _run(_ssh_base(cfg, host) + [f"bash -lc {shlex.quote(env_prefix + command)}"], 120)
        ok = code == 0
        notes.append(f"{name}: {'ok' if ok else 'FAIL'} {out.strip()[-300:]}")
        if not ok and name in ("npu-smi", "torch_npu"):
            return False, notes
    return True, notes


def _package(task_dir: Path, files: list[str], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name in files:
            path = task_dir / name
            if path.is_file():
                tar.add(path, arcname=name)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_board(
    task_dir: Path,
    spec: TaskSpec,
    cfg: Config,
    *,
    test_name: str | None = None,
    kernel_name: str | None = None,
    force: bool = False,
) -> BoardResult:
    started = time.time()
    host = cfg.board.host
    if not host and not force:
        return BoardResult(
            status="skipped",
            notes=["board.host is empty; set AIOP_BOARD_HOST or config/aiop.yaml"],
            elapsed_s=time.time() - started,
        )
    notes: list[str] = []
    ok, pre_notes = preflight(cfg)
    notes.extend(pre_notes)
    if not ok:
        return BoardResult(status="blocked", host=host, notes=notes, error="remote preflight failed")

    kernel = task_dir / (kernel_name or f"{spec.name}.py")
    test = task_dir / (test_name or f"test_{spec.name}.py")
    manifest = task_dir / "manifest.json"
    if not test.is_file():
        return BoardResult(status="failed", host=host, error=f"missing test script {test.name}")

    remote_base = cfg.board.remote_dir.rstrip("/")
    remote_dir = f"{remote_base}/{task_dir.name}"
    archive = task_dir / "board_package.tar.gz"
    _package(task_dir, [kernel.name, test.name, manifest.name], archive)

    code, out = _run(_ssh_base(cfg, host) + [f"mkdir -p {shlex.quote(remote_dir)}"], 60)
    if code != 0:
        return BoardResult(status="blocked", host=host, notes=notes, error=f"mkdir failed: {out[-500:]}")
    code, out = _run(_scp_base(cfg) + [str(archive), f"{host}:{remote_dir}/board_package.tar.gz"], 300)
    if code != 0:
        return BoardResult(status="blocked", host=host, notes=notes, error=f"upload failed: {out[-500:]}")

    result_path = f"{remote_dir}/board_result.json"
    remote_cmd = (
        f"cd {shlex.quote(remote_dir)} && tar xzf board_package.tar.gz && "
        f"{cfg.board.python} {test.name} --json {result_path} "
        f"--warmup {cfg.board.warmup} --repeats {cfg.board.repeats}"
    )
    if cfg.board.env_script:
        remote_cmd = f"{cfg.board.env_script} && {remote_cmd}"
    code, out = _run(
        _ssh_base(cfg, host) + [f"bash -lc {shlex.quote(remote_cmd)}"],
        cfg.board.timeout_s,
    )
    (task_dir / "board_run.log").write_text(out, encoding="utf-8")

    code2, fetched = _run(
        _scp_base(cfg)
        + [f"{host}:{result_path}", str(task_dir / "board_result.json")],
        120,
    )
    if code2 != 0:
        return BoardResult(
            status="failed",
            host=host,
            notes=notes,
            error=f"run exit={code}; could not fetch result: {fetched[-500:]}",
            elapsed_s=time.time() - started,
        )
    data = json.loads((task_dir / "board_result.json").read_text(encoding="utf-8"))
    if cfg.board.profile_cmd:
        code3, profile = _run(
            _ssh_base(cfg, host)
            + [
                f"bash -lc {shlex.quote((cfg.board.env_script + ' && ' if cfg.board.env_script else '') + cfg.board.profile_cmd)}"
            ],
            cfg.board.timeout_s,
        )
        (task_dir / "board_profile.txt").write_text(profile, encoding="utf-8")
        data["profile_status"] = "ok" if code3 == 0 else f"exit={code3}"
        data["profile_output"] = str(task_dir / "board_profile.txt")
    status = str(data.get("status", "error"))
    board = BoardResult(
        status="passed" if status == "passed" else ("blocked" if status == "unavailable" else "failed"),
        host=host,
        result=data,
        error=str(data.get("error", "")),
        notes=notes,
        elapsed_s=time.time() - started,
    )
    data["kernel_sha256"] = _sha256(kernel) if kernel.is_file() else ""
    if board.status == "passed":
        data["utilization"] = data.get("utilization") or {"aic": None, "aiv": None, "hbm": None}
        data["utilization_note"] = "null = not measurable without CANN profiler on the remote run"
    (task_dir / "board_report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return board
