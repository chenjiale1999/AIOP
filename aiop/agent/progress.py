"""Unified task progress: one ``run.log`` per task plus line-by-line output.

Every stage emits one timestamped line that is appended to
``workspace/<task>/run.log`` and (optionally) streamed to a terminal stream.
The reporter is intentionally fail-safe: a logging error must never abort an
operator run. Console writes are serialized process-wide so concurrent HTTP
tasks do not interleave partial lines.

The log is the machine-readable event stream; ``debug_log.md`` keeps the
detailed per-attempt diagnostics and ``result.json`` stays the final verdict.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

_PRINT_LOCK = threading.Lock()


def _fmt(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


class ProgressReporter:
    def __init__(
        self,
        task_dir: str | Path,
        task_id: str,
        *,
        enabled: bool = True,
        echo: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.task_id = task_id
        self.path = Path(task_dir) / "run.log"
        self.enabled = enabled
        self.echo = echo
        self.stream = stream or sys.stderr
        self.started = time.time()
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"\n===== run {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                    f"task={self.task_id} =====\n"
                )

    def emit(
        self,
        stage: str,
        message: str,
        *,
        status: str = "info",
        **facts: Any,
    ) -> None:
        if not self.enabled:
            return
        elapsed = time.time() - self.started
        line = (
            f"[{time.strftime('%H:%M:%S')}] [+{elapsed:5.1f}s] "
            f"[{self.task_id}] [{stage}] {message}"
        )
        if status != "info":
            line += f" status={status}"
        if facts:
            line += " " + " ".join(f"{key}={_fmt(value)}" for key, value in facts.items())
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
            except OSError:
                pass
        if self.echo:
            with _PRINT_LOCK:
                try:
                    print(line, file=self.stream, flush=True)
                except Exception:  # noqa: BLE001 - a closed stream must not fail the run
                    pass


def tail(path: str | Path, lines: int = 50) -> list[str]:
    """Return the last ``lines`` lines of a log without loading the whole file."""
    file_path = Path(path)
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.readlines()[-max(1, lines) :]
    except OSError:
        return []
