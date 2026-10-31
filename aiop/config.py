"""Configuration loading for the aiop operator-generation agent.

Values are resolved in this order: environment variable > config YAML > default.
Only the handful of knobs the pipeline actually needs are exposed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """Invalid user configuration (wrong type, unknown step, bad log level)."""


def _expand(value: str | os.PathLike[str] | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return os.path.expandvars(os.path.expanduser(text)) if text else ""


def _project_path(value: str | os.PathLike[str] | None) -> str:
    text = _expand(value)
    path = Path(text) if text else PROJECT_ROOT
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _is_tilelang_root(path: str | os.PathLike[str] | None) -> bool:
    if not path:
        return False
    root = Path(path)
    return (root / "tilelang" / "__init__.py").is_file() or (
        root / "tilelang" / "ascend" / "language"
    ).is_dir()


def discover_tilelang_repo() -> str:
    """Find a native A5 tilelang checkout without assuming a user directory."""
    for key in ("AIOP_TILELANG_REPO", "TILELANG_REPO"):
        value = os.environ.get(key)
        if not value:
            continue
        root = Path(_expand(value))
        if _is_tilelang_root(root):
            return str(root.resolve())
        # The variable may point directly at the ``tilelang`` package.
        if (root / "__init__.py").is_file() and (root / "ascend" / "language").is_dir():
            return str(root.resolve().parent)
    candidates = [
        PROJECT_ROOT.parent / "tilelang",
        PROJECT_ROOT / "tilelang",
        Path.cwd() / "tilelang",
        Path.cwd().parent / "tilelang",
        Path.home() / "tilelang",
        Path("/opt/tilelang"),
    ]
    for candidate in candidates:
        if _is_tilelang_root(candidate):
            return str(candidate.resolve())
    return ""


def discover_cann_home() -> str:
    """Find an installed CANN toolkit (env first, then common install roots)."""
    for key in ("AIOP_CANN_HOME", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "CANN_HOME"):
        value = os.environ.get(key)
        if value and Path(_expand(value)).is_dir():
            return str(Path(_expand(value)).resolve())
    candidates: list[Path] = [
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
        Path("/usr/local/Ascend/latest"),
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
    ]
    for base in (Path("/usr/local/Ascend"), Path.home() / "Ascend"):
        if base.is_dir():
            candidates.extend(sorted(base.glob("cann-*"), reverse=True))
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    return ""


def discover_manifest_dir() -> str:
    """Find the TileOPs manifest spec directory, falling back to bundled specs."""
    value = os.environ.get("AIOP_TILEOPS_MANIFEST")
    if value and Path(_expand(value)).is_dir():
        return str(Path(_expand(value)).resolve())
    candidates = [
        PROJECT_ROOT.parent / "TileOPs" / "src" / "tileops" / "manifest" / "spec",
        PROJECT_ROOT / "TileOPs" / "src" / "tileops" / "manifest" / "spec",
        PROJECT_ROOT / "examples" / "spec",
        Path.cwd() / "examples" / "spec",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    return ""


@dataclass
class Paths:
    tilelang_repo: str = ""
    workspace: str = str(PROJECT_ROOT / "workspace")
    knowledge: str = str(PROJECT_ROOT / "knowledge")
    samples: str = str(PROJECT_ROOT / "samples")
    reports: str = str(PROJECT_ROOT / "reports")
    tileops_manifest: str = ""


@dataclass
class CannConfig:
    home: str = ""
    npusim: str = ""
    soc_version: str = "Ascend950"


@dataclass
class CompileConfig:
    target: str = "ascend"
    npu_arch: str = "dav-3510"
    device_flags: list[str] = field(
        default_factory=lambda: [
            "-D__INTEL_LLVM_COMPILER=1",
            "-DSYCL_LANGUAGE_VERSION=1",
        ]
    )
    timeout_s: int = 600


@dataclass
class SimulationConfig:
    backend: str = "auto"  # auto | npusim | local_npu | static
    timeout_s: int = 1800
    repeats: int = 20
    warmup: int = 5


@dataclass
class StepsConfig:
    """Independent on/off switches for optional execution stages."""

    compile: bool = True
    simulate: bool = True
    npu: bool = True
    profile: bool = True


@dataclass
class LoggingConfig:
    command_level: str = "info"  # off | error | warn | info | debug
    echo_commands: bool = False


@dataclass
class ReferenceConfig:
    """Dual-baseline alignment with the TileOPs CUDA/Ascend kernels."""

    tileops_repo: str = ""
    enabled: bool = True
    require_cuda: bool = True
    require_ascend: bool = True
    max_chars: int = 40000
    parity_atol: float = 1e-5
    enforce_parity: bool = True
    integration_required: bool = True


@dataclass
class BoardConfig:
    host: str = ""
    port: int = 22
    key: str = ""
    remote_dir: str = "/tmp/aiop_runs"
    python: str = "python3"
    env_script: str = ""
    tilelang_repo: str = ""
    profile_cmd: str = ""
    timeout_s: int = 1800
    repeats: int = 50
    warmup: int = 10


@dataclass
class LLMConfig:
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key_env: str = "AIOP_LLM_API_KEY"
    timeout_s: int = 120


@dataclass
class AgentConfig:
    max_attempts: int = 3
    max_sim_attempts: int = 2
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    cann: CannConfig = field(default_factory=CannConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    steps: StepsConfig = field(default_factory=StepsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    references: ReferenceConfig = field(default_factory=ReferenceConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def workspace(self) -> Path:
        return Path(self.paths.workspace)

    @property
    def knowledge_dir(self) -> Path:
        return Path(self.paths.knowledge)

    @property
    def npusim_bin(self) -> str:
        if self.cann.npusim:
            return self.cann.npusim
        if not self.cann.home:
            return ""
        return str(Path(self.cann.home) / "x86_64-linux" / "bin" / "npusim")


def _apply_dataclass(obj: Any, data: dict[str, Any] | None) -> None:
    if not data:
        return
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _default_config_path() -> Path:
    return PROJECT_ROOT / "config" / "aiop.yaml"


def _validate_section(data: dict[str, Any], name: str, allowed: set[str]) -> None:
    """Reject typos instead of silently ignoring an unknown switch."""
    section = data.get(name)
    if section is None:
        return
    if not isinstance(section, dict):
        raise ConfigError(
            f"invalid {name} section: expected a mapping with keys "
            f"{', '.join(sorted(allowed))}"
        )
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown {name} option(s): {', '.join(unknown)}; "
            f"expected {', '.join(sorted(allowed))}"
        )


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("1", "true", "yes", "on"):
            return True
        if key in ("0", "false", "no", "off"):
            return False
    raise ConfigError(
        f"invalid {name}={value!r}; expected a boolean "
        "(true/false, yes/no, on/off, 1/0)"
    )


def _normalize_log_level(value: Any) -> str:
    key = str(value or "info").strip().lower()
    if key == "warning":
        key = "warn"
    if key not in ("off", "error", "warn", "info", "debug"):
        raise ConfigError(
            f"invalid logging.command_level={value!r}; "
            "expected one of off, error, warn, info, debug"
        )
    return key


def apply_steps(
    cfg: Config,
    *,
    compile: bool | None = None,
    simulate: bool | None = None,
    npu: bool | None = None,
    profile: bool | None = None,
) -> None:
    """Apply CLI overrides without touching unrelated switches."""
    for name, value in (
        ("compile", compile),
        ("simulate", simulate),
        ("npu", npu),
        ("profile", profile),
    ):
        if value is not None:
            setattr(cfg.steps, name, bool(value))


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg = Config()
    candidate = Path(path) if path else Path(os.environ.get("AIOP_CONFIG", _default_config_path()))
    if candidate.is_file():
        with open(candidate, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("config root must be a YAML mapping")
    cfg.raw = data
    _validate_section(data, "steps", {"compile", "simulate", "npu", "profile"})
    _validate_section(data, "logging", {"command_level", "echo_commands"})
    _validate_section(
        data,
        "references",
        {
            "tileops_repo",
            "enabled",
            "require_cuda",
            "require_ascend",
            "max_chars",
            "parity_atol",
            "enforce_parity",
            "integration_required",
        },
    )
    _apply_dataclass(cfg.paths, data.get("paths"))
    _apply_dataclass(cfg.cann, data.get("cann"))
    _apply_dataclass(cfg.compile, data.get("compile"))
    _apply_dataclass(cfg.simulation, data.get("simulation"))
    _apply_dataclass(cfg.steps, data.get("steps"))
    _apply_dataclass(cfg.logging, data.get("logging"))
    _apply_dataclass(cfg.references, data.get("references"))
    _apply_dataclass(cfg.board, data.get("board"))
    _apply_dataclass(cfg.agent, data.get("agent"))
    _apply_dataclass(cfg.agent.llm, (data.get("agent") or {}).get("llm"))

    # Environment overrides for deployment-sensitive values.
    env_map = {
        "AIOP_TILELANG_REPO": (cfg.paths, "tilelang_repo"),
        "AIOP_WORKSPACE": (cfg.paths, "workspace"),
        "AIOP_CANN_HOME": (cfg.cann, "home"),
        "AIOP_NPUSIM": (cfg.cann, "npusim"),
        "AIOP_TILEOPS_MANIFEST": (cfg.paths, "tileops_manifest"),
        "AIOP_TILEOPS_REPO": (cfg.references, "tileops_repo"),
        "AIOP_SOC_VERSION": (cfg.cann, "soc_version"),
        "AIOP_BOARD_HOST": (cfg.board, "host"),
        "AIOP_BOARD_KEY": (cfg.board, "key"),
        "AIOP_BOARD_DIR": (cfg.board, "remote_dir"),
        "AIOP_BOARD_PYTHON": (cfg.board, "python"),
        "AIOP_BOARD_ENV_SCRIPT": (cfg.board, "env_script"),
        "AIOP_LLM_BASE_URL": (cfg.agent.llm, "base_url"),
        "AIOP_LLM_MODEL": (cfg.agent.llm, "model"),
    }
    for env_name, (obj, attr) in env_map.items():
        value = os.environ.get(env_name)
        if value:
            setattr(obj, attr, value)

    # Normalize every deployment-sensitive path: expand env vars/~, resolve
    # relative paths against the project root, and fall back to autodetection
    # instead of assuming a specific user directory.
    cfg.paths.workspace = _project_path(cfg.paths.workspace or "workspace")
    cfg.paths.knowledge = _project_path(cfg.paths.knowledge or "knowledge")
    cfg.paths.samples = _project_path(cfg.paths.samples or "samples")
    cfg.paths.reports = _project_path(cfg.paths.reports or "reports")
    cfg.paths.tilelang_repo = _expand(cfg.paths.tilelang_repo)
    if not _is_tilelang_root(cfg.paths.tilelang_repo):
        cfg.paths.tilelang_repo = discover_tilelang_repo()
    cfg.paths.tileops_manifest = _expand(cfg.paths.tileops_manifest)
    if not cfg.paths.tileops_manifest or not Path(cfg.paths.tileops_manifest).is_dir():
        cfg.paths.tileops_manifest = discover_manifest_dir()
    cfg.cann.home = _expand(cfg.cann.home)
    if not cfg.cann.home or not Path(cfg.cann.home).is_dir():
        cfg.cann.home = discover_cann_home()
    cfg.cann.npusim = _expand(cfg.cann.npusim)
    cfg.simulation.backend = str(cfg.simulation.backend or "auto").strip().lower()
    if cfg.simulation.backend not in ("auto", "npusim", "local_npu", "static"):
        raise ConfigError(
            f"invalid simulation.backend={cfg.simulation.backend!r}; "
            "expected one of auto, npusim, local_npu, static"
        )

    for field_name in ("compile", "simulate", "npu", "profile"):
        env_name = f"AIOP_STEP_{field_name.upper()}"
        raw = os.environ.get(env_name, getattr(cfg.steps, field_name))
        setattr(cfg.steps, field_name, _coerce_bool(f"steps.{field_name}", raw))
    cfg.logging.command_level = _normalize_log_level(
        os.environ.get("AIOP_COMMAND_LOG_LEVEL", cfg.logging.command_level)
    )
    cfg.logging.echo_commands = _coerce_bool(
        "logging.echo_commands",
        os.environ.get("AIOP_COMMAND_LOG_ECHO", cfg.logging.echo_commands),
    )
    cfg.references.tileops_repo = _expand(cfg.references.tileops_repo)
    for key in (
        "enabled",
        "require_cuda",
        "require_ascend",
        "enforce_parity",
        "integration_required",
    ):
        raw = os.environ.get(f"AIOP_REFERENCE_{key.upper()}", getattr(cfg.references, key))
        setattr(cfg.references, key, _coerce_bool(f"references.{key}", raw))
    raw_atol = os.environ.get("AIOP_REFERENCE_ATOL", cfg.references.parity_atol)
    try:
        cfg.references.parity_atol = float(raw_atol)
        cfg.references.max_chars = int(cfg.references.max_chars)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"invalid references.parity_atol/max_chars: {exc}"
        ) from exc
    if cfg.references.parity_atol < 0:
        raise ConfigError("references.parity_atol must be >= 0")
    if cfg.references.max_chars < 0:
        raise ConfigError("references.max_chars must be >= 0")

    if cfg.agent.llm.base_url and cfg.agent.llm.model:
        cfg.agent.llm.enabled = True
    return cfg
