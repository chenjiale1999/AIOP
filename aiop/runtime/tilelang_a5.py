"""Native Ascend A5 TileLang bootstrap, lowering and compilation helpers.

The default Python environment has an editable install that redirects
``import tilelang`` to a different checkout (tilelang-mlir-ascend). The A5
backend lives in the native repo, so the redirecting meta-path finder must be
removed *before* importing tilelang. This module is the only place allowed to
import tilelang.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..config import Config, discover_tilelang_repo

_BOOTSTRAPPED = False
_TILELANG = None


def _tilelang_repo(cfg: Config) -> str:
    value = str(cfg.paths.tilelang_repo or "").strip()
    if value:
        candidate = Path(value).expanduser()
        if (candidate / "tilelang").is_dir():
            return str(candidate.resolve())
    return discover_tilelang_repo()


def _has_ascend_language() -> bool:
    try:
        importlib.import_module("tilelang.ascend.language")
        return True
    except Exception:  # noqa: BLE001 - probing
        return False


def _prepend_path(key: str, value: str) -> None:
    current = os.environ.get(key, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if value in parts:
        return
    os.environ[key] = os.pathsep.join([value, *parts]) if parts else value


def setup_ascend_env(cfg: Config) -> None:
    home = Path(cfg.cann.home) if cfg.cann.home else None
    if home is not None and home.is_dir():
        os.environ["ASCEND_HOME_PATH"] = str(home)
        os.environ["ASCEND_TOOLKIT_HOME"] = str(home)
        for rel in ("bin", "tools/ccec_compiler/bin", "tools/profiler/bin", "tools/mspti/bin"):
            candidate = home / rel
            if candidate.is_dir():
                _prepend_path("PATH", str(candidate))
        lib_dirs = ["lib64", "x86_64-linux/lib64", "runtime/lib64", "compiler/lib64"]
        for rel in lib_dirs:
            candidate = home / rel
            if candidate.is_dir():
                _prepend_path("LD_LIBRARY_PATH", str(candidate))
    repo = _tilelang_repo(cfg)
    if repo and Path(repo).is_dir():
        _prepend_path("PATH", str(Path(repo) / "build" / "bin"))


def bootstrap(cfg: Config) -> Any:
    """Import a usable A5 tilelang checkout, bypassing editable redirects."""
    global _BOOTSTRAPPED, _TILELANG
    if _BOOTSTRAPPED and _TILELANG is not None:
        return _TILELANG
    import logging

    # Keep stdout clean for JSON CLI output; tilelang logs a dev-root warning.
    logging.getLogger("tilelang").setLevel(logging.ERROR)
    setup_ascend_env(cfg)

    repo = _tilelang_repo(cfg)
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if type(finder).__name__ != "ScikitBuildRedirectingFinder"
    ]

    existing = sys.modules.get("tilelang")
    if existing is not None and not _has_ascend_language():
        # A non-A5 checkout was imported first. Reloading native extension
        # modules would register the same FFI symbols twice, so refuse with an
        # actionable message instead of comparing paths.
        raise RuntimeError(
            "tilelang was already imported from "
            f"{getattr(existing, '__file__', '?')} and does not provide "
            "`tilelang.ascend.language`; start a fresh process with "
            "AIOP_TILELANG_REPO pointing at an A5 tilelang checkout."
        )

    import tilelang  # noqa: F401

    if not _has_ascend_language():
        raise RuntimeError(
            "no usable Ascend A5 tilelang found: expected a checkout or installed "
            "package providing `tilelang.ascend.language`. Set AIOP_TILELANG_REPO "
            f"to the checkout root (currently: {repo or '<unset>'})."
        )
    _BOOTSTRAPPED = True
    _TILELANG = tilelang
    return tilelang


def tilelang_module(cfg: Config) -> Any:
    return bootstrap(cfg)


def pass_configs(cfg: Config) -> dict[str, Any]:
    return {"tl.device_compile_flags": list(cfg.compile.device_flags)}


def load_module(path: str | Path, module_name: str = "aiop_generated_kernel") -> Any:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def lower_prim_func(func: Any, cfg: Config) -> str:
    tilelang = bootstrap(cfg)
    from tvm.target import Target

    with Target(cfg.compile.target):
        artifact = tilelang.lower(func, target=cfg.compile.target)
    return artifact.kernel_source


def compile_prim_func(func: Any, cfg: Config, out_idx: int = -1) -> Any:
    tilelang = bootstrap(cfg)
    return tilelang.compile(
        func,
        target=cfg.compile.target,
        out_idx=out_idx,
        pass_configs=pass_configs(cfg),
    )


def build_entry(module: Any, entry: str = "build") -> Any:
    builder = getattr(module, entry, None)
    if builder is None:
        raise AttributeError(f"generated module has no `{entry}()` entry")
    func = builder()
    if func is None:
        raise ValueError(f"`{entry}()` returned None")
    return func


def environment_probe(cfg: Config) -> dict[str, Any]:
    """Report what this machine can actually execute (never fake capability)."""
    cann_home = Path(cfg.cann.home) if cfg.cann.home else None
    bisheng = (cann_home / "bin" / "bisheng") if cann_home is not None else None
    result: dict[str, Any] = {
        "cann_home": cfg.cann.home,
        "cann_exists": bool(cann_home is not None and cann_home.is_dir()),
        "bisheng": str(bisheng) if bisheng is not None else "",
        "bisheng_exists": bool(bisheng is not None and bisheng.is_file()),
        "npusim": cfg.npusim_bin,
        "npusim_exists": bool(cfg.npusim_bin and Path(cfg.npusim_bin).is_file()),
        "torch_npu": False,
        "npu_device": False,
        "tilelang_repo": cfg.paths.tilelang_repo,
        "tilelang_native": False,
        "compile_ok": None,
        "notes": [],
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001 - probe only
        result["notes"].append(f"torch probe failed: {exc}")
    try:
        import torch_npu  # noqa: F401

        result["torch_npu"] = True
        result["npu_device"] = bool(torch.npu.is_available()) if "torch" in result else False
    except Exception as exc:  # noqa: BLE001
        result["notes"].append(f"torch_npu unavailable: {type(exc).__name__}")
    try:
        tilelang = bootstrap(cfg)
        result["tilelang_native"] = True
        result["tilelang_version"] = getattr(tilelang, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        result["notes"].append(f"tilelang bootstrap failed: {exc}")
    try:
        devs = list(Path("/dev").glob("davinci*"))
        result["dev_davinci"] = [str(p) for p in devs]
    except Exception:
        result["dev_davinci"] = []
    return result


def compile_probe(cfg: Config) -> dict[str, Any]:
    """Compile the smallest possible A5 kernel to prove the toolchain works."""
    tilelang = bootstrap(cfg)
    import tilelang.ascend.language as T

    n = 1024

    @T.prim_func
    def probe(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(1) as bx:
            with T.SimtVF(threads=128):
                for i in T.Parallel(n):
                    B[i] = A[i] + T.float32(1.0)

    started = time.time()
    try:
        compile_prim_func(probe, cfg)
        return {"ok": True, "elapsed_s": round(time.time() - started, 2)}
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"ok": False, "elapsed_s": round(time.time() - started, 2), "error": str(exc)[-4000:]}
