#!/usr/bin/env python3
"""Auto-generated A5 TileLang verification runner (aiop).

task: rmsnorm (rmsnorm)
spec_sha: {"attrs": {"eps": 1e-06}, "family": "rmsnorm", "formula": "y = x / sqrt(mean(x^2) + eps) * w", "inputs": [{"dtype": "float32", "name": "X", "shape": [128, 4096]}, {"dtype": "float32", "name": "W", "shape": [4096]}], "name": "rmsnorm", "notes": "", "outputs": [{"dtype": "float32", "name": "Y", "shape": [128, 4096]}], "perf": {"metric": "latency", "target": null, "unit": "ms"}, "precision": {"atol": 1e-05, "matched_ratio": 0.9999, "max_abs_error": 0.0001, "rtol": 1e-05}, "version": "1"}

Usage:
    python test_rmsnorm.py [--json result.json] [--device npu] [--warmup 5] [--repeats 20]
Exit codes: 0 pass, 1 accuracy/runtime failure, 3 environment unavailable.
"""
import argparse
import json
import os
import sys
import time

TILELANG_REPO = '/home/cjl/workspace/tilelang'
CANN_HOME = '/usr/local/Ascend/cann-9.2.0'
DEVICE_FLAGS = ['-D__INTEL_LLVM_COMPILER=1', '-DSYCL_LANGUAGE_VERSION=1']

if TILELANG_REPO not in sys.path:
    sys.path.insert(0, TILELANG_REPO)
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
os.environ.setdefault("ASCEND_HOME_PATH", CANN_HOME)
os.environ.setdefault("ASCEND_TOOLKIT_HOME", CANN_HOME)


def _prepend(key, value):
    parts = [p for p in os.environ.get(key, "").split(os.pathsep) if p]
    if value not in parts:
        os.environ[key] = os.pathsep.join([value, *parts])


_prepend("PATH", os.path.join(CANN_HOME, "bin"))
_prepend("PATH", os.path.join(CANN_HOME, "tools", "ccec_compiler", "bin"))
_prepend("LD_LIBRARY_PATH", os.path.join(CANN_HOME, "lib64"))
_prepend("LD_LIBRARY_PATH", os.path.join(CANN_HOME, "x86_64-linux", "lib64"))


import torch

try:
    import torch_npu  # noqa: F401
    HAS_NPU = bool(torch.npu.is_available())
except Exception as exc:  # noqa: BLE001
    HAS_NPU = False
    NPU_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    NPU_IMPORT_ERROR = ""

import tilelang
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(HERE, "rmsnorm.py")
spec = importlib.util.spec_from_file_location("kernel_rmsnorm", KERNEL_PATH)
kernel_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_mod)

ATOL = 1e-05
RTOL = 1e-05
MAX_ABS = 0.0001
REQUIRED_RATIO = 0.9999
OUT_DTYPE = torch.float32


def make_inputs(device):
    X = torch.randn((128, 4096), dtype=torch.float32)
    W = torch.randn((4096,), dtype=torch.float32)
    return [t.to(device) for t in [X, W]]


def golden(X, W):
    cpu = [t.detach().float().cpu() if t.is_floating_point() else t.detach().cpu() for t in [X, W]]
    x, w = cpu[0], cpu[1]
    xf = x.to(torch.float32)
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-06)
    wf = w.to(torch.float32).reshape(1, -1)
    out = xf * rstd * wf

    return out.to(OUT_DTYPE)


def compare(actual, expected):
    a = actual.detach().float().cpu().reshape(-1)
    e = expected.detach().float().cpu().reshape(-1)
    if a.numel() != e.numel():
        return {"passed": False, "error": f"size mismatch {a.numel()} vs {e.numel()}"}
    diff = (a - e).abs()
    denom = e.abs()
    tol = ATOL + RTOL * denom
    matched = (diff <= tol) | (torch.isnan(a) & torch.isnan(e))
    matched_ratio = float(matched.float().mean().item())
    max_abs = float(diff.nan_to_num(posinf=1e30).max().item()) if diff.numel() else 0.0
    return {
        "passed": bool(matched_ratio >= REQUIRED_RATIO and max_abs <= MAX_ABS),
        "max_abs_error": max_abs,
        "matched_ratio": matched_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="npu", choices=["npu", "cpu"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    result = {"task": "rmsnorm", "device": args.device, "status": "error"}

    if args.device == "npu" and not HAS_NPU:
        result.update(status="unavailable", error="torch_npu unavailable: " + (NPU_IMPORT_ERROR or "device not visible"))
    else:
        try:
            if args.device == "npu":
                torch.npu.set_device(0)
            inputs = make_inputs(args.device)
            raw_inputs = inputs
            if False:
                inputs = [t.reshape(-1) for t in inputs]
            expected = golden(*[t.cpu() for t in raw_inputs])
            kernel = tilelang.compile(
                kernel_mod.build(), target="ascend", out_idx=-1,
                pass_configs={"tl.device_compile_flags": DEVICE_FLAGS},
            )
            with torch.no_grad():
                outputs = kernel(*inputs)
                if args.device == "npu":
                    torch.npu.synchronize()
            outputs = outputs.reshape(expected.shape)
            acc = compare(outputs, expected)
            result.update(acc)
            if acc.get("passed"):
                for _ in range(max(0, args.warmup)):
                    kernel(*inputs)
                if args.device == "npu":
                    torch.npu.synchronize()
                times = []
                for _ in range(max(1, args.repeats)):
                    torch.npu.synchronize() if args.device == "npu" else None
                    t0 = time.perf_counter()
                    kernel(*inputs)
                    torch.npu.synchronize() if args.device == "npu" else None
                    times.append((time.perf_counter() - t0) * 1000.0)
                times.sort()
                result["latency_ms"] = times[len(times) // 2]
                result["p90_ms"] = times[min(len(times) - 1, int(len(times) * 0.9))]
                result["min_ms"] = times[0]
                result["samples"] = len(times)
                result["status"] = "passed"
            else:
                result["status"] = "failed"
        except Exception as exc:  # noqa: BLE001
            import traceback
            result.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])

    if result["status"] == "unavailable":
        print("[PRECISION_UNAVAILABLE]")
    else:
        print("[PRECISION_PASS]" if result.get("passed") else "[PRECISION_FAIL]")
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    if result["status"] == "unavailable":
        return 3
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
