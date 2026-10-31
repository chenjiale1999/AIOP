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
    open_mode: bool = False,
    draft_path: str | Path | None = None,
) -> tuple[Path, str]:
    """Return (kernel_path, generator_name).

    ``strategy``: auto | template | llm | draft.

    * ``template``: always render the verified A5 template.
    * ``llm``: always ask the LLM first, then fall back to the template.
    * ``draft``: take the kernel from an externally written file (e.g. the
      Trae/IDE agent); no network call, no template.
    * ``auto``: verified template when this family has one; LLM only for
      families without a template (open-ended mode). This keeps the
      deterministic path for everything the A5 templates already cover, and
      spends LLM calls where they add capability.

    LLM output must parse as Python and define build(); otherwise the template
    is used.
    """
    path = directory / f"{spec.name}.py"
    if strategy == "draft":
        source_path = Path(draft_path) if draft_path else Path(directory) / "draft.py"
        if not source_path.is_file():
            raise RuntimeError(
                f"draft not found: {source_path}. Run "
                "`python -m aiop prompt --op <Op> --workload <N>` and write draft.py first."
            )
        source = source_path.read_text(encoding="utf-8")
        ast.parse(source)
        if "def build(" not in source:
            raise ValueError("draft has no build()")
        path.write_text(source, encoding="utf-8")
        (Path(directory) / "draft_source.txt").write_text(str(source_path), encoding="utf-8")
        return path, "draft"
    has_template = spec.family in templates.TEMPLATE_NAME
    use_llm = strategy == "llm" or (
        strategy == "auto" and cfg.agent.llm.enabled and not has_template
    )
    if use_llm:
        client = LLMClient(cfg)
        if client.available:
            try:
                source = client.generate_kernel(
                    spec.to_dict(), design, knowledge, open_mode=open_mode
                )
                ast.parse(source)
                if "def build(" not in source:
                    raise ValueError("LLM output has no build()")
                path.write_text(source, encoding="utf-8")
                return path, "llm"
            except Exception as exc:  # noqa: BLE001 - fall through to templates
                (directory / "llm_error.log").write_text(str(exc), encoding="utf-8")
    if spec.family not in templates.TEMPLATE_NAME:
        raise RuntimeError(
            f"open-ended generation produced no usable kernel for family "
            f"{spec.family!r} and no template exists; see llm_error.log"
        )
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
    if spec.family not in templates.TEMPLATE_NAME:
        raise RuntimeError(
            f"LLM repair failed for family {spec.family!r} and no template exists"
        )
    return templates.write_kernel(spec, plan, kernel_path.parent, None), "template_regenerate"
