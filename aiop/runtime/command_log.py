"""Full command logging for compile / simulate / board execution.

Every external command is written *before* it is launched. The record keeps the
complete argv list, the shell-quoted command string, the working directory,
timeout and caller-supplied context. Records are appended as JSON Lines to
``commands.log`` so nothing is truncated, and an optional compact echo can be
streamed to stderr.

Logging is fail-safe and off the hot path: one append plus one optional print
per command, guarded by a lock. A logging failure never blocks the command.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

LEVELS = {"off": 0, "error": 1, "warn": 2, "warning": 2, "info": 3, "debug": 4}

# Debug records add the environment that shapes execution (never secrets).
_DEBUG_ENV_KEYS = (
    "PATH",
    "PYTHONPATH",
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "LD_LIBRARY_PATH",
)
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _debug_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _DEBUG_ENV_KEYS if key in os.environ}
    for key, value in os.environ.items():
        if key.startswith("AIOP_") and not any(m in key.upper() for m in _SECRET_MARKERS):
            env[key] = value
    return env


def normalize_level(value: Any) -> str:
    key = str(value or "info").strip().lower()
    if key == "warning":
        key = "warn"
    if key not in LEVELS:
        raise ValueError(
            f"invalid command log level {value!r}; "
            "expected off, error, warn, info or debug"
        )
    return key


class CommandLogger:
    """Append-only, thread-safe, non-truncating command log."""

    def __init__(
        self,
        path: str | Path,
        *,
        level: str = "info",
        echo: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.level = normalize_level(level)
        self.path = Path(path)
        self.echo = bool(echo)
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()
        if LEVELS[self.level] >= LEVELS["info"]:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    @property
    def enabled(self) -> bool:
        return LEVELS[self.level] >= LEVELS["info"]

    def log(
        self,
        stage: str,
        argv: Sequence[str],
        *,
        context: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        shell: bool = False,
    ) -> dict[str, Any] | None:
        """Record one command immediately before execution."""
        if not self.enabled:
            return None
        args = [str(item) for item in argv]
        try:
            command = shlex.join(args)
        except Exception:  # noqa: BLE001 - never fail a command for logging
            command = " ".join(args)
        record: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "level": "info",
            "stage": stage,
            "command": command,
            "argv": args,
            "argc": len(args),
            "cwd": os.getcwd(),
            "shell": bool(shell),
            "timeout_s": timeout_s,
            "context": context or {},
        }
        if LEVELS[self.level] >= LEVELS["debug"]:
            record["env"] = _debug_env()
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
            except OSError:
                pass
        if self.echo:
            with self._lock:
                try:
                    print(f"[CMD][{stage}] {command}", file=self.stream, flush=True)
                except Exception:  # noqa: BLE001 - closed stream
                    pass
        return record


def null_logger() -> CommandLogger:
    return CommandLogger(os.devnull, level="off")


def logger_for(
    task_dir: str | Path | None,
    *,
    level: str = "info",
    echo: bool = False,
) -> CommandLogger:
    """Create a per-task logger; fall back to a workspace-level file."""
    if task_dir is None:
        path = Path("commands.log")
    else:
        path = Path(task_dir) / "commands.log"
    return CommandLogger(path, level=level, echo=echo)
