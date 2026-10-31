"""Generate a self-contained correctness/benchmark script for a kernel.

The script is used both by the local simulator (npusim) and by the remote board
runner, so it must not depend on the aiop package being installed on the target.

Every script carries a structured case matrix: each entry declares the explicit
input ``shape`` and ``datatype`` plus a data variant (normal / zeros / ones /
large / tiny, with domain guards for log/sqrt/rsqrt/div).  When the spec is
bound to a TileOPs manifest, every legal workload row and dtype case gets its
own specialised kernel variant, because a TileLang kernel is compiled for one
fixed shape/dtype.  Per case the script reports precision metrics (max/mean
absolute and relative error, matched ratio) and performance metrics (latency,
device compute time, bandwidth, throughput).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from pprint import pformat

from ..runtime.hardware import Plan
from ..specs import TaskSpec
from .casematrix import KernelVariant, input_key, plan_variants

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
    if spec.name.lower().startswith("hardsigmoid"):
        return "", ["torch.nn.functional.hardsigmoid(x)"]
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


def _case_specs(spec: TaskSpec) -> list[dict[str, object]]:
    """Deterministic data variants; every case must pass for overall success."""
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
    if spec.family == "unsupported" and len(spec.inputs) == 1:
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


def _split_future_imports(source: str) -> tuple[str, str]:
    """Hoist ``from __future__`` lines so the kernel can be inlined mid-file."""
    futures: list[str] = []
    kept: list[str] = []
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith("from __future__ import "):
            futures.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(futures), "".join(kept)


def _rename_build(source: str, index: int) -> str:
    renamed, count = re.subn(r"(?m)^def build\(", f"def build_{index}(", source)
    return renamed if count else ""


def _reshape_mode(spec: TaskSpec) -> tuple[str, int]:
    if spec.family in ("elementwise_unary", "elementwise_binary"):
        return "flat", 0
    if spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm"):
        return "rows", int(spec.input.shape[-1])
    return "none", 0


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_") or "case"


def _case_key(shape: list[list[int]], datatype: list[str]) -> str:
    return json.dumps({"shape": shape, "datatype": datatype}, sort_keys=True)


def _case_entries(
    variant: KernelVariant, index: int
) -> list[dict[str, object]]:
    spec = variant.spec
    shape = [list(t.shape) for t in spec.inputs]
    datatype = [t.dtype for t in spec.inputs]
    precision = spec.precision
    assert precision is not None
    output = spec.output
    reshape, reshape_d = _reshape_mode(spec)
    base = {
        "shape": shape,
        "datatype": datatype,
        "inputs": [
            {"name": t.name, "shape": list(t.shape), "dtype": t.dtype}
            for t in spec.inputs
        ],
        "output": {
            "name": output.name,
            "shape": list(output.shape),
            "dtype": output.dtype,
        },
        "denom_index": 1 if spec.name == "div" else None,
        "positive": spec.name in ("log", "sqrt", "rsqrt"),
        "reshape": reshape,
        "reshape_d": reshape_d,
        "kernel": f"build_{index}",
        "golden": f"golden_aiop_{index}",
        "precision": {
            "atol": float(precision.atol),
            "rtol": float(precision.rtol),
            "max_abs_error": float(precision.max_abs_error),
            "matched_ratio": float(precision.matched_ratio),
        },
        "bytes": int(spec.bytes()),
        "flops": int(spec.flops()),
        "elements": int(sum(t.numel for t in spec.inputs)),
        "workload": variant.workload,
        "dtype_case": variant.dtype_case,
        "workload_label": variant.label,
    }
    label = _slug(variant.label or "case")
    dtype_tag = "_".join(datatype)
    entries: list[dict[str, object]] = []
    for data_case in _case_specs(spec):
        entry = dict(base)
        entry["name"] = f"v{index}_{label}_{dtype_tag}_{data_case['name']}"
        entry.update(data_case)
        entries.append(entry)
    return entries


def _unavailable_entry(variant: KernelVariant, index: int, error: str) -> dict[str, object]:
    spec = variant.spec
    return {
        "name": f"v{index}_unavailable",
        "shape": [list(t.shape) for t in spec.inputs],
        "datatype": [t.dtype for t in spec.inputs],
        "inputs": [
            {"name": t.name, "shape": list(t.shape), "dtype": t.dtype}
            for t in spec.inputs
        ],
        "output": {
            "name": spec.output.name,
            "shape": list(spec.output.shape),
            "dtype": spec.output.dtype,
        },
        "available": False,
        "error": error,
        "kernel": None,
        "golden": None,
        "workload": variant.workload,
        "dtype_case": variant.dtype_case,
        "workload_label": variant.label,
    }


def render_test(
    spec: TaskSpec,
    plan: Plan,
    cfg_dump: dict,
    tilelang_repo: str,
    golden_override: str | None = None,
    kernel_source: str | None = None,
    variants: list[KernelVariant] | None = None,
) -> str:
    if variants is None:
        variants = plan_variants(
            spec,
            plan,
            kernel_source or "",
            golden_override=golden_override,
            enumerate_variants=False,
        )

    futures: set[str] = set()
    kernel_blocks: list[str] = []
    golden_blocks: list[str] = []
    out_dtype_consts: list[str] = []
    cases: list[dict[str, object]] = []
    unsupported: list[dict[str, object]] = []
    coverage_keys: list[str] = []
    executable_keys: list[str] = []

    for index, variant in enumerate(variants):
        case_spec = variant.spec
        key = _case_key(
            [list(t.shape) for t in case_spec.inputs],
            [t.dtype for t in case_spec.inputs],
        )
        if key not in coverage_keys:
            coverage_keys.append(key)

        future_lines, body_source = _split_future_imports(variant.source)
        if future_lines:
            futures.add(future_lines)
        error = variant.error
        renamed = ""
        if not error:
            renamed = _rename_build(body_source, index)
            if not renamed:
                error = "kernel source has no top-level `def build(`"

        out_dtype = case_spec.output.dtype
        if not error and out_dtype not in TORCH_DTYPE:
            error = f"unsupported output dtype {out_dtype!r}"

        golden_body = ""
        output_expr = ""
        if not error:
            try:
                golden_body, outputs = _golden_body(case_spec, variant.golden_override)
                output_expr = outputs[0]
                golden_body = _indent_body(golden_body)
            except Exception as exc:  # noqa: BLE001 - reported per case
                error = f"golden reference unavailable: {type(exc).__name__}: {exc}"

        if error:
            entry = _unavailable_entry(variant, index, error)
            cases.append(entry)
            unsupported.append(entry)
            continue

        label = variant.label or "task-bound"
        dtype_tag = "_".join(t.dtype for t in case_spec.inputs)
        if key not in executable_keys:
            executable_keys.append(key)
        kernel_blocks.append(
            f"# ---- kernel variant {index}: workload={variant.workload} "
            f"dtype_case={variant.dtype_case} ({label}, {dtype_tag}) ----\n"
            f"{renamed.rstrip()}\n"
        )
        out_dtype_consts.append(
            f"OUT_DTYPE_AIOP_{index} = {TORCH_DTYPE[out_dtype]}"
        )
        golden_args = ", ".join(t.name for t in case_spec.inputs)
        golden_blocks.append(
            f"def golden_aiop_{index}({golden_args}):\n"
            f"    cpu = [t.detach().float().cpu() if t.is_floating_point() else t.detach().cpu() for t in [{golden_args}]]\n"
            f"    {_golden_destructure(case_spec)}\n"
            f"{golden_body}\n"
            f"    return {output_expr}.to(OUT_DTYPE_AIOP_{index})\n"
        )
        cases.extend(_case_entries(variant, index))

    future_block = "".join(line + "\n" for line in sorted(futures))
    cases_json = pformat(cases, width=100, sort_dicts=False)
    unsupported_json = pformat(unsupported, width=100, sort_dicts=False)
    coverage_json = pformat(
        {
            "declared": len(coverage_keys),
            "executable": len(executable_keys),
            "keys": coverage_keys,
            "executable_keys": executable_keys,
        },
        width=100,
        sort_dicts=False,
    )
    dtype_map = ",\n    ".join(f'{key!r}: {value}' for key, value in TORCH_DTYPE.items())
    kernel_text = "\n".join(kernel_blocks).rstrip()
    golden_text = "\n\n".join(golden_blocks).rstrip()
    dtype_text = "\n".join(out_dtype_consts)

    return f'''#!/usr/bin/env python3
"""Auto-generated A5 TileLang verification runner (aiop).

task: {spec.name} ({spec.family})
golden_source: {"llm-self-referenced" if golden_override else "aiop-builtin"}
spec_sha: {json.dumps(spec.to_dict(), sort_keys=True)}

This file is fully self-contained. Every case declares an explicit `shape` and
`datatype`; the matching kernel variant, case data, golden reference and
expected-value computation are all inlined. No external kernel module,
test-data file or baseline file is imported or read.

Usage:
    python test_{spec.name}.py [--json result.json] [--device npu] [--warmup 5] [--repeats 20]
Exit codes: 0 pass, 1 accuracy/runtime failure, 3 environment unavailable.
"""
{future_block}import argparse
import glob
import json
import os
import statistics
import sys
import time

_DEFAULT_TILELANG_REPO_AIOP = {tilelang_repo!r}
_DEFAULT_CANN_HOME_AIOP = {cfg_dump['cann_home']!r}


def _discover_tilelang_repo_aiop():
    value = os.environ.get("AIOP_TILELANG_REPO")
    if value and os.path.isdir(value):
        return value
    if _DEFAULT_TILELANG_REPO_AIOP and os.path.isdir(_DEFAULT_TILELANG_REPO_AIOP):
        return _DEFAULT_TILELANG_REPO_AIOP
    return ""


def _discover_cann_home_aiop():
    for key in ("AIOP_CANN_HOME", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(key)
        if value and os.path.isdir(value):
            return value
    candidates = [
        _DEFAULT_CANN_HOME_AIOP,
        "/usr/local/Ascend/ascend-toolkit/latest",
    ]
    candidates.extend(sorted(glob.glob("/usr/local/Ascend/cann-*"), reverse=True))
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return ""


TILELANG_REPO_AIOP = _discover_tilelang_repo_aiop()
CANN_HOME_AIOP = _discover_cann_home_aiop()
DEVICE_FLAGS_AIOP = {cfg_dump['device_flags']!r}

if TILELANG_REPO_AIOP and TILELANG_REPO_AIOP not in sys.path:
    sys.path.insert(0, TILELANG_REPO_AIOP)
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
if CANN_HOME_AIOP:
    os.environ.setdefault("ASCEND_HOME_PATH", CANN_HOME_AIOP)
    os.environ.setdefault("ASCEND_TOOLKIT_HOME", CANN_HOME_AIOP)


def _prepend_aiop(key, value):
    parts = [p for p in os.environ.get(key, "").split(os.pathsep) if p]
    if value not in parts:
        os.environ[key] = os.pathsep.join([value, *parts])


if CANN_HOME_AIOP:
    _prepend_aiop("PATH", os.path.join(CANN_HOME_AIOP, "bin"))
    _prepend_aiop("PATH", os.path.join(CANN_HOME_AIOP, "tools", "ccec_compiler", "bin"))
    _prepend_aiop("LD_LIBRARY_PATH", os.path.join(CANN_HOME_AIOP, "lib64"))
    _prepend_aiop("LD_LIBRARY_PATH", os.path.join(CANN_HOME_AIOP, "x86_64-linux", "lib64"))


import torch

try:
    import torch_npu  # noqa: F401
    HAS_NPU_AIOP = bool(torch.npu.is_available())
except Exception as exc:  # noqa: BLE001
    HAS_NPU_AIOP = False
    NPU_IMPORT_ERROR_AIOP = f"{{type(exc).__name__}}: {{exc}}"
else:
    NPU_IMPORT_ERROR_AIOP = ""

import tilelang

# ===================== embedded kernel variants =====================
{kernel_text}
# =================== end kernel variants ===================

DTYPE_TO_TORCH_AIOP = {{
    {dtype_map},
}}
INT_DTYPES_AIOP = {{"int8", "int16", "int32", "uint8", "uint16"}}

{dtype_text}

CASE_SPECS_AIOP = {cases_json}

UNSUPPORTED_CONTRACTS_AIOP = {unsupported_json}

COVERAGE_AIOP = {coverage_json}

SYNC_AIOP = False


{golden_text}


def _sync_aiop():
    if SYNC_AIOP:
        torch.npu.synchronize()


def _apply_case_aiop(tensors, case):
    kind = case.get("kind", "as_is")
    scale = float(case.get("scale", 1.0))
    floor = case.get("floor")
    denom_floor = case.get("denom_floor")
    denom_index = case.get("denom_index")
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
            if denom_floor is not None and denom_index is not None and index == int(denom_index):
                tensor = tensor.abs() + float(denom_floor)
        out.append(tensor)
    return out


def make_inputs_aiop(case, device):
    tensors = []
    positive = bool(case.get("positive"))
    denom_index = case.get("denom_index")
    for index, item in enumerate(case["inputs"]):
        shape = tuple(int(v) for v in item["shape"])
        dtype = item["dtype"]
        torch_dtype = DTYPE_TO_TORCH_AIOP[dtype]
        if dtype in INT_DTYPES_AIOP:
            tensor = torch.randint(-8, 8, shape, dtype=torch_dtype)
        elif dtype == "bool":
            tensor = torch.randint(0, 2, shape, dtype=torch.int8).bool()
        elif positive:
            tensor = torch.rand(shape, dtype=torch_dtype) * 2.0 + 0.1
        elif denom_index is not None and index == int(denom_index):
            tensor = torch.rand(shape, dtype=torch_dtype) * 2.0 + 0.5
        else:
            tensor = torch.randn(shape, dtype=torch_dtype)
        tensors.append(tensor)
    tensors = _apply_case_aiop(tensors, case)
    return [t.to(device) for t in tensors]


def _reshape_inputs_aiop(inputs, case):
    mode = case.get("reshape")
    if mode == "flat":
        return [t.reshape(-1) for t in inputs]
    if mode == "rows":
        rows = int(case["reshape_d"])
        return [inputs[0].reshape(-1, rows)] + list(inputs[1:])
    return inputs


def compare_aiop(actual, expected, precision):
    a = actual.detach().float().cpu().reshape(-1)
    e = expected.detach().float().cpu().reshape(-1)
    if a.numel() != e.numel():
        return {{
            "passed": False,
            "error": f"size mismatch {{a.numel()}} vs {{e.numel()}}",
            "total_elements": int(a.numel()),
        }}
    diff = (a - e).abs()
    denom = e.abs()
    tol = float(precision["atol"]) + float(precision["rtol"]) * denom
    matched = (diff <= tol) | (torch.isnan(a) & torch.isnan(e))
    matched_ratio = float(matched.float().mean().item()) if diff.numel() else 1.0
    finite = diff.nan_to_num(posinf=1e30)
    rel = diff / denom.clamp_min(1e-12)
    rel = rel.nan_to_num(posinf=1e30)
    total = int(diff.numel())
    return {{
        "passed": bool(
            matched_ratio >= float(precision["matched_ratio"])
            and float(finite.max().item()) <= float(precision["max_abs_error"])
        ),
        "max_abs_error": float(finite.max().item()) if total else 0.0,
        "mean_abs_error": float(finite.mean().item()) if total else 0.0,
        "max_rel_error": float(rel.max().item()) if total else 0.0,
        "mean_rel_error": float(rel.mean().item()) if total else 0.0,
        "matched_ratio": matched_ratio,
        "mismatched_elements": int((~matched).sum().item()) if total else 0,
        "total_elements": total,
    }}


def measure_aiop(kernel, inputs, case, warmup, repeats):
    for _ in range(max(0, warmup)):
        kernel(*inputs)
    _sync_aiop()
    wall = []
    device = []
    for _ in range(max(1, repeats)):
        _sync_aiop()
        started = time.perf_counter()
        if SYNC_AIOP:
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            start_event.record()
            kernel(*inputs)
            end_event.record()
            _sync_aiop()
            device.append(float(start_event.elapsed_time(end_event)))
        else:
            kernel(*inputs)
            _sync_aiop()
        wall.append((time.perf_counter() - started) * 1000.0)
    wall.sort()
    latency = wall[len(wall) // 2]
    compute = sorted(device)[len(device) // 2] if device else latency
    seconds = max(latency, 1e-9) / 1000.0
    compute_seconds = max(compute, 1e-9) / 1000.0
    bytes_total = float(case.get("bytes", 0))
    flops = float(case.get("flops", 0))
    elements = float(case.get("elements", 0))
    return {{
        "latency_ms": latency,
        "compute_ms": compute,
        "bandwidth_gbps": bytes_total / seconds / 1e9,
        "throughput_gflops": flops / compute_seconds / 1e9,
        "throughput_melems_s": elements / seconds / 1e6,
        "bytes": int(bytes_total),
        "flops": int(flops),
        "samples": len(wall),
    }}


def _case_key_aiop(case):
    return json.dumps(
        {{"shape": case.get("shape"), "datatype": case.get("datatype")}},
        sort_keys=True,
    )


def main_aiop():
    global SYNC_AIOP
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    parser.add_argument("--device", default="npu", choices=["npu", "cpu"])
    parser.add_argument("--warmup", type=int, default={int(cfg_dump['warmup'])})
    parser.add_argument("--repeats", type=int, default={int(cfg_dump['repeats'])})
    args = parser.parse_args()
    SYNC_AIOP = args.device == "npu"

    result = {{
        "task": "{spec.name}",
        "device": args.device,
        "status": "error",
        "coverage": {{
            "declared": COVERAGE_AIOP["declared"],
            "executable": COVERAGE_AIOP["executable"],
            "executed": 0,
            "ratio": 0.0,
            "complete": False,
            "missing": list(COVERAGE_AIOP["keys"]),
            "unsupported": len(UNSUPPORTED_CONTRACTS_AIOP),
        }},
        "cases": [],
    }}

    if args.device == "npu" and not HAS_NPU_AIOP:
        result.update(
            status="unavailable",
            error="torch_npu unavailable: " + (NPU_IMPORT_ERROR_AIOP or "device not visible"),
        )
    else:
        if args.device == "npu":
            torch.npu.set_device(0)
        compiled = {{}}
        case_results = []
        for case in CASE_SPECS_AIOP:
            entry = {{
                "name": case["name"],
                "shape": case["shape"],
                "datatype": case["datatype"],
                "kernel": case.get("kernel"),
                "status": "error",
            }}
            if not case.get("available", True):
                entry.update(status="unsupported", error=case.get("error", "kernel variant unavailable"))
                case_results.append(entry)
                continue
            try:
                build_name = case["kernel"]
                if build_name not in compiled:
                    compiled[build_name] = tilelang.compile(
                        globals()[build_name](),
                        target="ascend",
                        out_idx=-1,
                        pass_configs={{"tl.device_compile_flags": DEVICE_FLAGS_AIOP}},
                    )
                kernel = compiled[build_name]
                inputs = make_inputs_aiop(case, args.device)
                expected = globals()[case["golden"]](*[t.cpu() for t in inputs])
                kernel_inputs = _reshape_inputs_aiop(inputs, case)
                with torch.no_grad():
                    outputs = kernel(*kernel_inputs)
                    _sync_aiop()
                outputs = outputs.reshape(expected.shape)
                precision = compare_aiop(outputs, expected, case["precision"])
                performance = measure_aiop(
                    kernel, kernel_inputs, case, args.warmup, args.repeats
                )
                entry.update(
                    status="passed" if precision.get("passed") else "failed",
                    precision=precision,
                    performance=performance,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback
                entry.update(
                    status="error",
                    error=f"{{type(exc).__name__}}: {{exc}}",
                    traceback=traceback.format_exc()[-3000:],
                )
            case_results.append(entry)

        result["cases"] = case_results
        executed = set()
        for entry in case_results:
            if entry["status"] in ("passed", "failed"):
                executed.add(_case_key_aiop(entry))
        declared = set(COVERAGE_AIOP["keys"])
        missing = [key for key in COVERAGE_AIOP["keys"] if key not in executed]
        ratio = (len(executed & declared) / len(declared)) if declared else 0.0
        result["coverage"] = {{
            "declared": len(declared),
            "executable": len(COVERAGE_AIOP["executable_keys"]),
            "executed": len(executed & declared),
            "ratio": ratio,
            "complete": bool(declared) and not missing,
            "missing": missing,
            "unsupported": len(UNSUPPORTED_CONTRACTS_AIOP),
        }}

        statuses = [entry["status"] for entry in case_results]
        precisions = [entry["precision"] for entry in case_results if "precision" in entry]
        performances = [entry["performance"] for entry in case_results if "performance" in entry]
        summary = {{
            "total": len(case_results),
            "passed": statuses.count("passed"),
            "failed": statuses.count("failed"),
            "error": statuses.count("error"),
            "unsupported": statuses.count("unsupported"),
            "unavailable": statuses.count("unavailable"),
            "max_abs_error": max((p.get("max_abs_error", 0.0) for p in precisions), default=0.0),
            "mean_abs_error": (
                sum(p.get("mean_abs_error", 0.0) for p in precisions) / len(precisions)
                if precisions else 0.0
            ),
            "matched_ratio": min((p.get("matched_ratio", 0.0) for p in precisions), default=0.0),
            "latency_ms": statistics.median([p["latency_ms"] for p in performances]) if performances else 0.0,
            "compute_ms": statistics.median([p["compute_ms"] for p in performances]) if performances else 0.0,
            "bandwidth_gbps": statistics.median([p["bandwidth_gbps"] for p in performances]) if performances else 0.0,
            "throughput_gflops": statistics.median([p["throughput_gflops"] for p in performances]) if performances else 0.0,
        }}
        result["summary"] = summary
        result["latency_ms"] = summary["latency_ms"]
        result["max_abs_error"] = summary["max_abs_error"]
        result["matched_ratio"] = summary["matched_ratio"]
        executable_statuses = [s for s in statuses if s != "unsupported"]
        if not executable_statuses:
            result["status"] = "failed"
            result["error"] = "no executable cases for this task"
        else:
            result["status"] = (
                "passed" if all(s == "passed" for s in executable_statuses) else "failed"
            )

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
    raise SystemExit(main_aiop())
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
    kernel_source: str | None = None,
    variants: list[KernelVariant] | None = None,
    enumerate_variants: bool = True,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"test_{spec.name}.py"
    if kernel_source is None:
        sibling = directory / f"{spec.name}.py"
        kernel_source = sibling.read_text(encoding="utf-8") if sibling.is_file() else ""
    if variants is None:
        variants = plan_variants(
            spec,
            plan,
            kernel_source or "",
            golden_override=golden_override,
            enumerate_variants=enumerate_variants,
        )
    cfg_dump = {
        "cann_home": cann_home,
        "device_flags": device_flags,
        "warmup": warmup,
        "repeats": repeats,
    }
    path.write_text(
        render_test(
            spec,
            plan,
            cfg_dump,
            tilelang_repo,
            golden_override=golden_override,
            kernel_source=kernel_source,
            variants=variants,
        ),
        encoding="utf-8",
    )
    return path
