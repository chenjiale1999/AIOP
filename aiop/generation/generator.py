"""Kernel source selection: verified templates first, optional LLM draft."""

from __future__ import annotations

import ast
from pathlib import Path

from ..config import Config
from ..runtime.hardware import Plan
from ..specs import TaskSpec
from . import templates
from .llm import LLMClient


def generate(
    spec: TaskSpec,
    plan: Plan,
    directory: Path,
    cfg: Config,
    *,
    design: str = "",
    knowledge: str = "",
    strategy: str = "auto",
) -> tuple[Path, str]:
    """Return (kernel_path, generator_name).

    ``strategy``: auto | template | llm. In auto mode the LLM is used only when
    configured. LLM output must parse as Python and define build(); otherwise
    the verified template is used.
    """
    path = directory / f"{spec.name}.py"
    use_llm = strategy == "llm" or (strategy == "auto" and cfg.agent.llm.enabled)
    if use_llm:
        client = LLMClient(cfg)
        if client.available:
            try:
                source = client.generate_kernel(spec.to_dict(), design, knowledge)
                ast.parse(source)
                if "def build(" not in source:
                    raise ValueError("LLM output has no build()")
                path.write_text(source, encoding="utf-8")
                return path, "llm"
            except Exception as exc:  # noqa: BLE001 - fall through to templates
                (directory / "llm_error.log").write_text(str(exc), encoding="utf-8")
    return templates.write_kernel(spec, plan, directory, None), "template"


def repair(
    spec: TaskSpec,
    plan: Plan,
    kernel_path: Path,
    error: str,
    cfg: Config,
    *,
    design: str = "",
) -> tuple[Path, str]:
    if cfg.agent.llm.enabled:
        client = LLMClient(cfg)
        if client.available:
            try:
                source = client.repair_kernel(kernel_path.read_text(encoding="utf-8"), error, design)
                ast.parse(source)
                kernel_path.write_text(source, encoding="utf-8")
                return kernel_path, "llm_repair"
            except Exception:  # noqa: BLE001
                pass
    return templates.write_kernel(spec, plan, kernel_path.parent, None), "template_regenerate"
