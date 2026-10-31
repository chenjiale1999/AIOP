"""Generate a self-contained correctness/benchmark script for a kernel.

The script is used both by the local simulator (npusim) and by the remote board
runner, so it must not depend on the aiop package being installed on the target.

Every script runs a small, deterministic case matrix (normal / zeros / ones /
large / tiny, with domain guards for log/sqrt/rsqrt/div) against one compiled
kernel. This is the aiop counterpart of the ``IndependentEvaluator`` in
``some_ref.md``: fixed inputs, fixed oracle, all cases must pass before a
measurement is reported.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..runtime.hardware import Plan
from ..specs import DTYPE_BYTES, TaskSpec

TORCH_DTYPE = {
    "float32": "torch.float32",
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "int8": "torch.int8",
    "int16": "torch.int16",
    "int32": "torch.int32",
    "uint8": "torch.uint8",
    "uint16": "torch.uint16",
    "bool": "torch.bool",
}

FLOAT_DTYPES = {
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "float8_e4m3fn",
    "float8_e5m2",
    "hifloat8",
}

GOLDEN_FAMILIES = {
    "elementwise_unary",
    "elementwise_binary",
    "row_reduce",
    "softmax",
    "rmsnorm",
    "layernorm",
    "gemm",
}


def _golden_body(spec: TaskSpec, override: str | None = None) -> tuple[str, str]:
    """Return (python statements, output expression list).

    ``override`` is an LLM-written CPU reference body for families without a
    built-in golden; it must assign the primary result to ``out``.
    """
    if override is not None:
        return override, ["out"]
    if spec.family == "elementwise_unary":
        expr = {
            "relu": "torch.relu(x)",
            "abs": "torch.abs(x)",
            "exp": "torch.exp(x)",
            "neg": "-x",
            "sigmoid": "torch.sigmoid(x)",
            "tanh": "torch.tanh(x)",
            "rsqrt": "torch.rsqrt(x)",
            "sqrt": "torch.sqrt(x)",
            "log": "torch.log(x)",
            "cast": "x.to(torch.float32)",
        }[spec.name]
        return "", [expr]
    if spec.family == "elementwise_binary":
        expr = {
            "add": "a + b",
            "sub": "a - b",
            "mul": "a * b",
            "div": "a / b",
            "maximum": "torch.maximum(a, b)",
            "minimum": "torch.minimum(a, b)",
        }[spec.name]
        return "", [expr]
    if spec.family == "row_reduce":
        op = {
            "reduce_sum": "torch.sum(x, dim=-1)",
            "reduce_mean": "torch.mean(x, dim=-1)",
            "reduce_max": "torch.max(x, dim=-1).values",
            "reduce_min": "torch.min(x, dim=-1).values",
        }[spec.name]
        return "", [op]
    if spec.family == "softmax":
        return "", ["torch.softmax(x, dim=-1)"]
    if spec.family == "rmsnorm":
        eps = float(spec.attrs.get("eps", 1e-6))
        body = (
            f"    xf = x.to(torch.float32)\n"
            f"    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + {eps!r})\n"
            f"    wf = w.to(torch.float32).reshape(1, -1)\n"
            f"    out = xf * rstd * wf\n"
        )
        return body, ["out"]
    if spec.family == "layernorm":
        eps = float(spec.attrs.get("eps", 1e-6))
        has_bias = len(spec.inputs) == 3
        body = (
            f"    xf = x.to(torch.float32)\n"
            f"    mu = xf.mean(-1, keepdim=True)\n"
            f"    var = xf.var(-1, unbiased=False, keepdim=True)\n"
            f"    wf = w.to(torch.float32).reshape(1, -1)\n"
            f"    out = (xf - mu) * torch.rsqrt(var + {eps!r}) * wf\n"
        )
        if has_bias:
            body += "    out = out + bias.to(torch.float32).reshape(1, -1)\n"
        return body, ["out"]
    if spec.family == "gemm":
        body = (
            "    out = torch.matmul(a.to(torch.float32), b.to(torch.float32).transpose(0, 1))\n"
        )
        if spec.name == "gemm_relu" or spec.attrs.get("epilogue") == "relu":
            body += "    out = torch.relu(out)\n"
        return body, ["out"]
    raise ValueError(spec.family)


def _input_creation(spec: TaskSpec) -> str:
    lines = []
    for index, t in enumerate(spec.inputs):
        if t.dtype in ("int8", "int16", "int32", "uint8", "uint16"):
            lines.append(
                f"    {t.name} = torch.randint(-8, 8, {tuple(t.shape)}, dtype={TORCH_DTYPE[t.dtype]})"
            )
        elif t.dtype == "bool":
            lines.append(f"    {t.name} = torch.randint(0, 2, {tuple(t.shape)}, dtype=torch.int8).bool()")
        elif spec.name in ("log", "sqrt", "rsqrt"):
            # Positive domain so the reference and the kernel see the same
            # finite inputs instead of NaN on the random draws.
            lines.append(
                f"    {t.name} = torch.rand({tuple(t.shape)}, dtype={TORCH_DTYPE[t.dtype]}) * 2.0 + 0.1"
            )
        elif spec.name == "div" and index == 1:
            lines.append(
                f"    {t.name} = torch.rand({tuple(t.shape)}, dtype={TORCH_DTYPE[t.dtype]}) * 2.0 + 0.5"
            )
        else:
            lines.append(
                f"    {t.name} = torch.randn({tuple(t.shape)}, dtype={TORCH_DTYPE[t.dtype]})"
            )
    return "\n".join(lines)


def _case_specs(spec: TaskSpec) -> list[dict[str, object]]:
    """Deterministic case matrix; every case must pass for overall success."""
    cases: list[dict[str, object]] = [{"name": "normal", "kind": "as_is"}]
    has_float = any(t.dtype in FLOAT_DTYPES for t in spec.inputs)
    if not has_float:
        cases.append({"name": "zeros", "kind": "zeros"})
        cases.append({"name": "ones", "kind": "ones"})
        return cases
    cases.append({"name": "zeros", "kind": "zeros"})
    cases.append({"name": "ones", "kind": "ones"})
    if spec.name != "exp":  # exp(100) overflows fp32 in both kernel and golden
        cases.append({"name": "large", "kind": "scale", "scale": 100.0})
    cases.append({"name": "tiny", "kind": "scale", "scale": 0.001})
    if spec.name in ("log", "sqrt", "rsqrt"):
        for case in cases:
            case["floor"] = 1e-3
    if spec.name == "div":
        for case in cases:
            case["denom_floor"] = 1e-3
    return cases


def _golden_destructure(spec: TaskSpec) -> str:
    if spec.family in ("elementwise_binary", "gemm"):
        return "a, b = cpu[0], cpu[1]"
    if spec.family in ("rmsnorm", "layernorm"):
        if len(spec.inputs) == 3:
            return "x, w, bias = cpu[0], cpu[1], cpu[2]"
        return "x, w = cpu[0], cpu[1]"
    if spec.family in ("elementwise_unary", "row_reduce", "softmax"):
        return "x = cpu[0]"
    names = [tensor.name for tensor in spec.inputs]
    if len(names) == 1:
        return f"{names[0]} = cpu[0]"
    return ", ".join(names) + " = " + ", ".join(f"cpu[{i}]" for i in range(len(names)))


def _indent_body(body: str) -> str:
    """Normalize a golden body to a 4-space base indent for inlining."""
    if not body.strip():
        return ""
    lines = [line.rstrip() for line in body.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    base = min(indents) if indents else 0
    return "\n".join("    " + line[base:] if line.strip() else "" for line in lines)


def _input_reshape(spec: TaskSpec) -> str:
    """Kernel-side view of the tensors: flat for elementwise, [rows, D] for row families."""
    if spec.family in ("elementwise_unary", "elementwise_binary"):
        return "inputs = [t.reshape(-1) for t in inputs]"
    if spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm"):
        d = spec.input.shape[-1]
        return f"inputs = [inputs[0].reshape(-1, {d})] + list(inputs[1:])"
    return "pass"


def render_test(
    spec: TaskSpec,
    plan: Plan,
    cfg_dump: dict,
    tilelang_repo: str,
    golden_override: str | None = None,
) -> str:
    golden_body, outputs = _golden_body(spec, golden_override)
    golden_source = "llm-self-referenced" if golden_override else "aiop-builtin"
    body = _indent_body(golden_body)
    out_dtype = spec.output.dtype
    pr = spec.precision
    assert pr is not None
    input_names = ", ".join(t.name for t in spec.inputs)
    golden_args = ", ".join(t.name for t in spec.inputs)
    golden_hash = json.dumps(spec.to_dict(), sort_keys=True)
    cases_json = json.dumps(_case_specs(spec), ensure_ascii=False)
    reshape_code = _input_reshape(spec)

    return f'''#!/usr/bin/env python3
"""Auto-generated A5 TileLang verification runner (aiop).

task: {spec.name} ({spec.family})
golden_source: {golden_source}
spec_sha: {golden_hash}

Usage:
    python test_{spec.name}.py [--json result.json] [--device npu] [--warmup 5] [--repeats 20]
Exit codes: 0 pass, 1 accuracy/runtime failure, 3 environment unavailable.
"""
import argparse
import json
import os
import sys
import time

TILELANG_REPO = {tilelang_repo!r}
CANN_HOME = {cfg_dump['cann_home']!r}
DEVICE_FLAGS = {cfg_dump['device_flags']!r}

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
    NPU_IMPORT_ERROR = f"{{type(exc).__name__}}: {{exc}}"
else:
    NPU_IMPORT_ERROR = ""

import tilelang
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(HERE, "{spec.name}.py")
spec = importlib.util.spec_from_file_location("kernel_{spec.name}", KERNEL_PATH)
kernel_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_mod)

ATOL = {pr.atol!r}
RTOL = {pr.rtol!r}
MAX_ABS = {pr.max_abs_error!r}
REQUIRED_RATIO = {pr.matched_ratio!r}
OUT_DTYPE = {TORCH_DTYPE[out_dtype]}
CASE_SPECS = {cases_json}


def _apply_case(tensors, case):
    kind = case.get("kind", "as_is")
    scale = float(case.get("scale", 1.0))
    floor = case.get("floor")
    denom_floor = case.get("denom_floor")
    out = []
    for index, tensor in enumerate(tensors):
        if kind == "zeros":
            tensor = torch.zeros_like(tensor)
        elif kind == "ones":
            tensor = torch.ones_like(tensor)
        elif kind == "scale":
            tensor = tensor * scale
        if tensor.is_floating_point():
            if floor is not None and kind in ("zeros", "scale"):
                tensor = tensor.abs() + float(floor)
            if denom_floor is not None and index == 1:
                tensor = tensor.abs() + float(denom_floor)
        out.append(tensor)
    return out


def make_inputs(device, case=None):
    case = case or CASE_SPECS[0]
{_input_creation(spec)}
    tensors = _apply_case([{", ".join(t.name for t in spec.inputs)}], case)
    return [t.to(device) for t in tensors]


def golden({golden_args}):
    cpu = [t.detach().float().cpu() if t.is_floating_point() else t.detach().cpu() for t in [{golden_args}]]
    {_golden_destructure(spec)}
{body}
    return {outputs[0]}.to(OUT_DTYPE)


def compare(actual, expected):
    a = actual.detach().float().cpu().reshape(-1)
    e = expected.detach().float().cpu().reshape(-1)
    if a.numel() != e.numel():
        return {{"passed": False, "error": f"size mismatch {{a.numel()}} vs {{e.numel()}}"}}
    diff = (a - e).abs()
    denom = e.abs()
    tol = ATOL + RTOL * denom
    matched = (diff <= tol) | (torch.isnan(a) & torch.isnan(e))
    matched_ratio = float(matched.float().mean().item())
    max_abs = float(diff.nan_to_num(posinf=1e30).max().item()) if diff.numel() else 0.0
    return {{
        "passed": bool(matched_ratio >= REQUIRED_RATIO and max_abs <= MAX_ABS),
        "max_abs_error": max_abs,
        "matched_ratio": matched_ratio,
    }}


def _compile_kernel():
    return tilelang.compile(
        kernel_mod.build(), target="ascend", out_idx=-1,
        pass_configs={{"tl.device_compile_flags": DEVICE_FLAGS}},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="npu", choices=["npu", "cpu"])
    ap.add_argument("--warmup", type=int, default={int(cfg_dump['warmup'])})
    ap.add_argument("--repeats", type=int, default={int(cfg_dump['repeats'])})
    args = ap.parse_args()
    result = {{"task": "{spec.name}", "device": args.device, "status": "error"}}

    if args.device == "npu" and not HAS_NPU:
        result.update(status="unavailable", error="torch_npu unavailable: " + (NPU_IMPORT_ERROR or "device not visible"))
    else:
        try:
            if args.device == "npu":
                torch.npu.set_device(0)
            kernel = _compile_kernel()
            case_results = []
            for case in CASE_SPECS:
                inputs = make_inputs(args.device, case)
                raw_inputs = inputs
                {reshape_code}
                expected = golden(*[t.cpu() for t in raw_inputs])
                with torch.no_grad():
                    outputs = kernel(*inputs)
                    if args.device == "npu":
                        torch.npu.synchronize()
                outputs = outputs.reshape(expected.shape)
                acc = compare(outputs, expected)
                case_results.append({{
                    "name": case["name"],
                    "kind": case.get("kind", "as_is"),
                    "status": "passed" if acc.get("passed") else "failed",
                    "max_abs_error": acc.get("max_abs_error"),
                    "matched_ratio": acc.get("matched_ratio"),
                    "error": acc.get("error", ""),
                }})
            result["cases"] = case_results
            failed = [c for c in case_results if c["status"] != "passed"]
            if failed:
                result["status"] = "failed"
                result["error"] = f"{{len(failed)}}/{{len(case_results)}} cases failed: " + ", ".join(c["name"] for c in failed)
                result["max_abs_error"] = failed[0].get("max_abs_error")
                result["matched_ratio"] = failed[0].get("matched_ratio")
            else:
                normal = case_results[0]
                result["status"] = "passed"
                result["max_abs_error"] = normal.get("max_abs_error")
                result["matched_ratio"] = normal.get("matched_ratio")
                inputs = make_inputs(args.device, CASE_SPECS[0])
                raw_inputs = inputs
                {reshape_code}
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
        except Exception as exc:  # noqa: BLE001
            import traceback
            result.update(status="error", error=f"{{type(exc).__name__}}: {{exc}}", traceback=traceback.format_exc()[-4000:])

    if result["status"] == "unavailable":
        print("[PRECISION_UNAVAILABLE]")
    else:
        print("[PRECISION_PASS]" if result.get("status") == "passed" else "[PRECISION_FAIL]")
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    if result["status"] == "unavailable":
        return 3
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_test(
    spec: TaskSpec,
    plan: Plan,
    directory: str | Path,
    *,
    cann_home: str,
    device_flags: list[str],
    warmup: int,
    repeats: int,
    tilelang_repo: str,
    golden_override: str | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"test_{spec.name}.py"
    cfg_dump = {
        "cann_home": cann_home,
        "device_flags": device_flags,
        "warmup": warmup,
        "repeats": repeats,
    }
    path.write_text(
        render_test(spec, plan, cfg_dump, tilelang_repo, golden_override=golden_override),
        encoding="utf-8",
    )
    return path
