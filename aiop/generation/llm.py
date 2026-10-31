"""Optional OpenAI-compatible LLM adapter.

The deterministic template path is the default and always available. When an
OpenAI-compatible endpoint is configured (AIOP_LLM_BASE_URL / AIOP_LLM_MODEL),
the agent can additionally draft or repair kernels; every draft still has to
pass the same static/compile/simulation gates, so an LLM cannot bypass checks.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .. import ascend_ops
from ..config import Config

REDUCER_CONTRACT = """\
REDUCER CONTRACT — this is the only legal lifecycle:
    acc = T.alloc_reducer((rows,), "float32", op="sum")   # op: sum | max | min
    T.reducer_init(acc)
    for r, i in T.Parallel(rows, tile):
        T.reducer_update(acc[r], value)
    out = T.alloc_fragment((rows,), "float32")
    T.finalize_reducer(acc, out)
    # read out[r]; the reducer handle acc cannot be read or assigned.
There is NO T.reduce_init, T.reduce_update, T.reduce_finalize,
T.reducer_finalize, T.finalize_reduce or T.alloc_reduce.
"""

SYSTEM_TEMPLATE = """You are a TileLang Ascend 950 (dav-3510) kernel engineer.
Write a single self-contained Python module with:

    def build(): -> @T.prim_func

Rules:
- import tilelang and tilelang.ascend.language as T only.
- Never use A3 aliases (T.printf, T.tile.*, T.ascend.*, alloc_L1/L0A/L0B/L0C/ub);
  use T.print only for debugging, T.simd.*, T.ascend_*, and
  alloc_l1/l0a/l0b/l0c/alloc_shared instead.
- T.Kernel takes a ONE-dimensional grid and no threads argument.
- thread domains live in T.SimtVF(threads=...); SIMD in T.SimdVF().
- No torch, file IO, prints, asserts, or dynamic loop bounds.
- Follow the supplied design (tiling, memory tier, dtype) exactly.

{allowlist}

{reducer}

Any T.<op> or T.simd.<op> outside the allowlist is rejected by the static gate
(API001/API003/API004). Return only the module code, no markdown fences."""


def _allowlist_text(cfg: Config) -> str:
    """Render the directory-derived legal op surface for the prompt."""
    try:
        scope = ascend_ops.load_scope(cfg)
    except Exception:  # noqa: BLE001 - keep the LLM path usable without tilelang
        return (
            "AUTHORITATIVE API ALLOWLIST unavailable (tilelang import failed). "
            "Use only the ops named in the supplied design and knowledge."
        )
    if scope.source != "runtime":
        # A partial static surface would wrongly tell the model that common ops
        # do not exist; ask it to follow the design instead of guessing.
        return (
            "AUTHORITATIVE API ALLOWLIST unavailable (tilelang runtime import failed). "
            "Use only the ops named in the supplied design and knowledge."
        )
    top = ", ".join(sorted(scope.declared))
    simd = ", ".join(sorted(scope.simd))
    return (
        f"AUTHORITATIVE API ALLOWLIST — only these T.* names exist ({len(scope.declared)}):\n"
        f"{top}\n\n"
        f"T.simd.* — only these names exist ({len(scope.simd)}):\n{simd}"
    )


def build_system_prompt(cfg: Config) -> str:
    return SYSTEM_TEMPLATE.format(allowlist=_allowlist_text(cfg), reducer=REDUCER_CONTRACT)


class LLMClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg.agent.llm
        self.api_key = os.environ.get(self.cfg.api_key_env, "")
        self.system = build_system_prompt(cfg)

    @property
    def available(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.base_url and self.cfg.model)

    def complete(self, prompt: str, system: str | None = None, timeout: int | None = None) -> str:
        if not self.available:
            raise RuntimeError("LLM is not configured")
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system or self.system},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.cfg.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        return data["choices"][0]["message"]["content"]

    def generate_kernel(
        self,
        spec_dump: dict[str, Any],
        design: str,
        knowledge: str,
        *,
        open_mode: bool = False,
    ) -> str:
        extra = ""
        if open_mode:
            extra = (
                "\nOPEN-ENDED MODE: no aiop template exists for this family, so the tiling and\n"
                "memory plan are yours. Declare the primary output named in SPEC outputs[0] as\n"
                "the single result tensor returned by build(); metadata inputs are int32/int64\n"
                "tensors that already hold valid indices. Keep every on-chip buffer within\n"
                "UB 208 KiB / L1 512 KiB / L0C 256 KiB and the grid within 64 vector blocks.\n"
            )
        prompt = (
            "Implement this operator.\n\n"
            f"SPEC:\n{json.dumps(spec_dump, indent=2)}\n\n"
            f"DESIGN:\n{design}\n\n"
            f"KNOWLEDGE:\n{knowledge}\n"
            f"{extra}\n"
            f"{REDUCER_CONTRACT}\n"
            "Output only the Python module."
        )
        return _strip_fences(self.complete(prompt))

    def generate_golden(self, spec_dump: dict[str, Any], design: str) -> str:
        """Plain-PyTorch CPU reference, used only when no aiop golden exists.

        The result is a *self-referenced* oracle: it never counts as independent
        evidence, and the orchestrator records it as such in the report.
        """
        names = ", ".join(t["name"] for t in spec_dump.get("inputs", []))
        prompt = (
            "Write the body of a CPU PyTorch reference function for this operator.\n\n"
            f"SPEC:\n{json.dumps(spec_dump, indent=2)}\n\n"
            f"DESIGN (context):\n{design}\n\n"
            f"The function signature is golden({names}); every input is a CPU tensor named as\n"
            "in SPEC. Compute the PRIMARY output (SPEC outputs[0]) in float32 and assign it to\n"
            "a variable named `out` shaped like SPEC outputs[0]. Rules:\n"
            "- torch only; no tilelang, numpy, file IO or prints;\n"
            "- do not return anything, just assign `out`;\n"
            "- examples: `out = (x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps))`.\n"
            "Output only the body indented by four spaces, no `def`, no fences."
        )
        body = _strip_fences(self.complete(prompt))
        return _normalize_body(body)

    def repair_kernel(self, source: str, error: str, design: str, knowledge: str = "") -> str:
        prompt = (
            "The following generated kernel failed validation. Fix the root cause, keep the design.\n"
            "Do not invent API names: use only the allowlist in the system message.\n\n"
            f"DESIGN:\n{design}\n\n"
            f"KNOWLEDGE:\n{knowledge}\n\n"
            f"{REDUCER_CONTRACT}\n"
            f"ERROR:\n{error[-6000:]}\n\nSOURCE:\n{source[-12000:]}\n\n"
            "Output only the full corrected Python module."
        )
        return _strip_fences(self.complete(prompt))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _normalize_body(text: str) -> str:
    """Re-indent a reference body to one uniform 4-space base level."""
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("LLM returned an empty reference body")
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    base = min(indents) if indents else 0
    return "\n".join("    " + line[base:] if line.strip() else "" for line in lines)
