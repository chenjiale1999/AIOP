"""Execution switches, full command logging, and unsupported-stage skipping."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aiop.config import ConfigError, apply_steps, load_config
from aiop.runtime.command_log import CommandLogger, logger_for, normalize_level


def _cfg_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "aiop.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_steps_from_config_file(tmp_path):
    path = _cfg_file(
        tmp_path,
        """
steps:
  compile: false
  simulate: "off"
  npu: 0
  profile: yes
logging:
  command_level: debug
  echo_commands: true
""",
    )
    cfg = load_config(path)
    assert cfg.steps.compile is False
    assert cfg.steps.simulate is False
    assert cfg.steps.npu is False
    assert cfg.steps.profile is True
    assert cfg.logging.command_level == "debug"
    assert cfg.logging.echo_commands is True


def test_steps_env_overrides(tmp_path, monkeypatch):
    path = _cfg_file(tmp_path, "steps:\n  compile: false\n")
    monkeypatch.setenv("AIOP_STEP_COMPILE", "true")
    monkeypatch.setenv("AIOP_STEP_PROFILE", "no")
    monkeypatch.setenv("AIOP_COMMAND_LOG_LEVEL", "warning")
    cfg = load_config(path)
    assert cfg.steps.compile is True
    assert cfg.steps.profile is False
    assert cfg.logging.command_level == "warn"


def test_invalid_steps_value_is_rejected(tmp_path):
    path = _cfg_file(tmp_path, "steps:\n  compile: maybe\n")
    with pytest.raises(ConfigError, match="steps.compile"):
        load_config(path)


def test_unknown_step_option_is_rejected(tmp_path):
    path = _cfg_file(tmp_path, "steps:\n  comiple: false\n")
    with pytest.raises(ConfigError, match="unknown steps option"):
        load_config(path)
    path = _cfg_file(tmp_path, "logging:\n  echo: true\n")
    with pytest.raises(ConfigError, match="unknown logging option"):
        load_config(path)
    path = _cfg_file(tmp_path, "references:\n  require_cdua: true\n")
    with pytest.raises(ConfigError, match="unknown references option"):
        load_config(path)


def test_invalid_env_switch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AIOP_STEP_COMPILE", "sometimes")
    with pytest.raises(ConfigError, match="steps.compile"):
        load_config(_cfg_file(tmp_path, "steps:\n  compile: true\n"))


def test_invalid_command_level_is_rejected(tmp_path):
    path = _cfg_file(tmp_path, "logging:\n  command_level: verbose\n")
    with pytest.raises(ConfigError, match="logging.command_level"):
        load_config(path)


def test_invalid_simulation_backend_is_rejected(tmp_path):
    path = _cfg_file(tmp_path, "simulation:\n  backend: gpu\n")
    with pytest.raises(ConfigError, match="simulation.backend"):
        load_config(path)


def test_apply_steps_only_touches_given_switches():
    cfg = load_config()
    apply_steps(cfg, compile=False, profile=False)
    assert cfg.steps.compile is False
    assert cfg.steps.profile is False
    assert cfg.steps.simulate is True
    assert cfg.steps.npu is True


def test_command_log_keeps_full_argv_and_context(tmp_path):
    path = tmp_path / "logs" / "commands.log"
    stream = io.StringIO()
    log = CommandLogger(path, echo=True, stream=stream)
    argv = [
        "python3",
        "-c",
        "print('参数 带空格')",
        "--flag=值 with spaces",
    ]
    context = {"task_dir": str(tmp_path), "remote_command": "cd /tmp && echo '全部内容'"}
    record = log.log("board.run", argv, timeout_s=42, context=context)
    assert record is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["stage"] == "board.run"
    assert saved["argv"] == argv
    assert saved["argc"] == len(argv)
    assert saved["timeout_s"] == 42
    assert saved["context"] == context
    assert "参数 带空格" in saved["command"]
    assert saved["command"] in stream.getvalue()


def test_command_log_off_writes_nothing(tmp_path):
    path = tmp_path / "commands.log"
    log = CommandLogger(path, level="off")
    assert log.enabled is False
    assert log.log("compile", ["true"]) is None
    assert not path.exists()
    assert normalize_level("warning") == "warn"


def test_logger_for_uses_task_directory(tmp_path):
    log = logger_for(tmp_path)
    assert log.path == tmp_path / "commands.log"


@pytest.mark.slow
def test_disabled_steps_skip_compile_and_simulation(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.steps.compile = False
    cfg.steps.simulate = False
    cfg.steps.npu = False
    cfg.steps.profile = False
    cfg.board.host = ""

    def boom(*args, **kwargs):
        raise AssertionError("compile must not run when steps.compile is false")

    monkeypatch.setattr("aiop.agent.orchestrator.compile_kernel", boom)
    from aiop.agent.orchestrator import OperatorAgent

    result = OperatorAgent(cfg).run(_relu_spec(), board=True, progress=True)

    assert result["compile"]["category"] == "skipped"
    assert result["simulation"] is None
    assert result["board"] is None
    assert result["steps"] == {
        "compile": False,
        "simulate": False,
        "npu": False,
        "profile": False,
    }
    assert result["status"] == "success_static_only"
    task_dir = Path(cfg.paths.workspace) / result["task_id"]
    skipped = json.loads((task_dir / "compile_result.json").read_text(encoding="utf-8"))
    assert skipped["category"] == "skipped"
    run_log = (task_dir / "run.log").read_text(encoding="utf-8")
    assert "simulate_step_disabled" in run_log
    assert "npu_step_disabled" in run_log


def test_cli_step_flags_map_to_overrides(tmp_path):
    from aiop.cli import _apply_cli_steps, build_parser

    args = build_parser().parse_args(
        [
            "generate",
            "--no-compile",
            "--no-simulate",
            "--no-npu",
            "--no-profile",
            "--command-log-level",
            "debug",
            "--command-log-echo",
        ]
    )
    assert args.step_compile is False
    assert args.simulate is False
    assert args.npu is False
    assert args.profile is False
    cfg = load_config()
    _apply_cli_steps(cfg, args)
    assert cfg.steps.compile is False
    assert cfg.steps.simulate is False
    assert cfg.steps.npu is False
    assert cfg.steps.profile is False
    assert cfg.logging.command_level == "debug"
    assert cfg.logging.echo_commands is True


def test_compile_subcommand_honors_disabled_step(tmp_path, capsys):
    from aiop.cli import main

    spec = _relu_spec()
    task = tmp_path / "task"
    task.mkdir()
    spec.save(task / "spec.yaml")
    cfg = _cfg_file(tmp_path, "steps:\n  compile: false\n")
    assert main(["--config", str(cfg), "compile", str(task)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["category"] == "skipped"
    assert "steps.compile=false" in payload["error"]


def test_cli_reports_config_error(tmp_path, capsys):
    from aiop.cli import main

    bad = _cfg_file(tmp_path, "logging:\n  command_level: loud\n")
    assert main(["--config", str(bad), "env"]) == 2
    captured = capsys.readouterr()
    assert "config error:" in captured.err
    assert "logging.command_level" in captured.err


# ------------------------------------------------------------------ board profile


def _relu_spec():
    from aiop.specs import TaskSpec, TensorSpec

    return TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )


def _board_task(tmp_path: Path):
    spec = _relu_spec()
    task = tmp_path / "task"
    task.mkdir()
    (task / f"{spec.name}.py").write_text("kernel = 1\n", encoding="utf-8")
    (task / f"test_{spec.name}.py").write_text("test = 1\n", encoding="utf-8")
    (task / "manifest.json").write_text("{}", encoding="utf-8")
    return spec, task


def _patch_board(monkeypatch, task: Path, *, profiler: bool = True):
    stages: list[str] = []

    def fake_preflight(cfg, command_log=None):
        return True, ["npu-smi: ok"]

    def fake_run(cmd, timeout, *, command_log=None, stage="board", context=None):
        stages.append(stage)
        if command_log is not None:
            command_log.log(stage, cmd, timeout_s=timeout, context=context)
        if stage == "board.fetch":
            (task / "board_result.json").write_text(
                json.dumps({"status": "passed", "latency_ms": 1.0}), encoding="utf-8"
            )
        if stage == "board.profile_probe":
            return (0, "/usr/local/bin/msprof\n") if profiler else (1, "")
        return 0, ""

    monkeypatch.setattr("aiop.board.runner.preflight", fake_preflight)
    monkeypatch.setattr("aiop.board.runner._run", fake_run)
    return stages


def test_board_profile_runs_only_after_precision_passes(tmp_path, monkeypatch):
    from aiop.board.runner import run_board

    spec, task = _board_task(tmp_path)
    cfg = load_config()
    cfg.board.host = "fake-host"
    cfg.steps.profile = True
    stages = _patch_board(monkeypatch, task)
    log = CommandLogger(task / "commands.log")

    result = run_board(task, spec, cfg, command_log=log)
    assert result.status == "passed"
    assert result.result["profile"]["status"] == "ok"
    assert "board.profile" in stages
    logged = [
        json.loads(line)["stage"]
        for line in (task / "commands.log").read_text(encoding="utf-8").splitlines()
    ]
    for stage in ("board.mkdir", "board.upload", "board.run", "board.fetch", "board.profile"):
        assert stage in logged


def test_board_skips_profile_when_step_disabled(tmp_path, monkeypatch):
    from aiop.board.runner import run_board

    spec, task = _board_task(tmp_path)
    cfg = load_config()
    cfg.board.host = "fake-host"
    cfg.steps.profile = False
    stages = _patch_board(monkeypatch, task)
    log = CommandLogger(task / "commands.log")

    result = run_board(task, spec, cfg, command_log=log)
    assert result.result["profile"]["reason"] == "profile_step_disabled"
    assert "board.profile" not in stages
    assert any("profile skipped" in note for note in result.notes)


def test_board_skips_profile_without_profiler(tmp_path, monkeypatch):
    from aiop.board.runner import run_board

    spec, task = _board_task(tmp_path)
    cfg = load_config()
    cfg.board.host = "fake-host"
    cfg.steps.profile = True
    stages = _patch_board(monkeypatch, task, profiler=False)
    log = CommandLogger(task / "commands.log")

    result = run_board(task, spec, cfg, command_log=log)
    assert result.result["profile"]["reason"] == "profiler_not_available"
    assert "board.profile" not in stages
