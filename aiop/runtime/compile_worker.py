"""Isolated compile worker: lower + device-compile one generated kernel.

Run as a subprocess so compiler crashes and timeouts cannot take down the agent,
and so a stale editable tilelang install can be bypassed in a fresh interpreter.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from ..config import load_config
from . import tilelang_a5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--entry", default="build")
    parser.add_argument("--config", default="")
    args = parser.parse_args(argv)

    cfg = load_config(args.config or None)
    result: dict[str, object] = {"ok": False}
    try:
        tilelang_a5.bootstrap(cfg)
        module = tilelang_a5.load_module(args.module, "aiop_compile_target")
        func = tilelang_a5.build_entry(module, args.entry)
        source = tilelang_a5.lower_prim_func(func, cfg)
        source_path = Path(args.out).with_suffix(".cce.c")
        source_path.write_text(source, encoding="utf-8")
        kernel = tilelang_a5.compile_prim_func(func, cfg)
        result.update(
            ok=True,
            lowered_chars=len(source),
            source_path=str(source_path),
            kernel_type=type(kernel).__name__,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the agent
        result.update(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            stage="compile",
            traceback=traceback.format_exc()[-8000:],
        )
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
