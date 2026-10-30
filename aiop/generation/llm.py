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

from ..config import Config

SYSTEM = """You are a TileLang Ascend 950 (dav-3510) kernel engineer.
Write a single self-contained Python module with:

    def build(): -> @T.prim_func

Rules:
- import tilelang and tilelang.ascend.language as T only.
- T.Kernel takes a ONE-dimensional grid and no threads argument.
- thread domains live in T.SimtVF(threads=...); SIMD in T.SimdVF().
- reducer ops are limited to sum/max/min with explicit init/finalize.
- No torch, file IO, prints, asserts, or dynamic loop bounds.
- Follow the supplied design (tiling, memory tier, dtype) exactly.
Return only the module code, no markdown fences."""


class LLMClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg.agent.llm
        self.api_key = os.environ.get(self.cfg.api_key_env, "")

    @property
    def available(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.base_url and self.cfg.model)

    def complete(self, prompt: str, system: str = SYSTEM, timeout: int | None = None) -> str:
        if not self.available:
            raise RuntimeError("LLM is not configured")
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
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

    def generate_kernel(self, spec_dump: dict[str, Any], design: str, knowledge: str) -> str:
        prompt = (
            "Implement this operator.\n\n"
            f"SPEC:\n{json.dumps(spec_dump, indent=2)}\n\n"
            f"DESIGN:\n{design}\n\n"
            f"KNOWLEDGE:\n{knowledge}\n\n"
            "Output only the Python module."
        )
        return _strip_fences(self.complete(prompt))

    def repair_kernel(self, source: str, error: str, design: str) -> str:
        prompt = (
            "The following generated kernel failed validation. Fix the root cause, keep the design.\n\n"
            f"DESIGN:\n{design}\n\nERROR:\n{error[-6000:]}\n\nSOURCE:\n{source[-12000:]}\n\n"
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
