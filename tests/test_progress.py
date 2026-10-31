"""Unified progress log and terminal streaming."""

from __future__ import annotations

import io

from aiop.agent.orchestrator import OperatorAgent
from aiop.agent.progress import ProgressReporter, tail
from aiop.config import load_config
from aiop.specs import TaskSpec, TensorSpec


def test_reporter_writes_unified_log_and_streams_lines(tmp_path):
    stream = io.StringIO()
    reporter = ProgressReporter(tmp_path, "task-1", stream=stream)
    reporter.emit("design", "start")
    reporter.emit("compile", "passed", category="none", elapsed_s=1.2)

    log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "task=task-1" in log
    assert "[task-1] [design] start" in log
    assert "[task-1] [compile] passed" in log
    assert "elapsed_s=1.2" in log
    printed = stream.getvalue().splitlines()
    assert len(printed) == 2
    assert "compile" in printed[1]
    assert tail(tmp_path / "run.log", 1)[0].strip().endswith("elapsed_s=1.2")


def test_disabled_reporter_is_a_noop(tmp_path):
    stream = io.StringIO()
    reporter = ProgressReporter(tmp_path, "task-2", enabled=False, stream=stream)
    reporter.emit("design", "start")
    assert not (tmp_path / "run.log").exists()
    assert stream.getvalue() == ""


def test_unsupported_task_progress_is_logged_and_streamed(tmp_path, capfd):
    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    spec = TaskSpec(
        "fusedprobe",
        "attention",
        [TensorSpec("x", (8, 8), "float32")],
        [TensorSpec("y", (8, 8), "float32")],
        supported=False,
        unsupported_reasons=["attention has no A5 template family"],
    )
    result = OperatorAgent(cfg).run(spec, simulate=False, progress=True)
    assert result["status"] == "blocked_unsupported"
    task_dir = tmp_path / "workspace" / result["task_id"]
    log = (task_dir / "run.log").read_text(encoding="utf-8")
    assert "[design] blocked" in log
    assert "[done] blocked" in log
    assert result["artifacts"]["run_log"] == str(task_dir / "run.log")
    assert "[design] blocked" in capfd.readouterr().err
