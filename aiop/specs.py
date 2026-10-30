"""Operator task specification and derived workload metrics."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DTYPE_ALIASES = {
    "fp32": "float32",
    "float": "float32",
    "f32": "float32",
    "fp16": "float16",
    "half": "float16",
    "f16": "float16",
    "bf16": "bfloat16",
    "bfloat16_t": "bfloat16",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "u8": "uint8",
    "u16": "uint16",
    "int": "int32",
    "boolean": "bool",
    "fp8_e4m3fn": "float8_e4m3fn",
    "float8_e4m3fn_t": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2",
    "float8_e5m2_t": "float8_e5m2",
    "hifloat8": "hifloat8",
    "hifloat8_t": "hifloat8",
    "int4": "int4x2",
    "int4b_t": "int4x2",
    "fp4x2_e2m1": "fp4x2_e2m1",
    "fp4x2_e1m2": "fp4x2_e1m2",
}

DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "hifloat8": 1,
    "int4x2": 1,
    "fp4x2_e2m1": 1,
    "fp4x2_e1m2": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "uint16": 2,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "float64": 8,
    "int64": 8,
}

A5_DTYPES = set(DTYPE_BYTES)

DEFAULT_PRECISION = {
    "float32": dict(atol=1e-5, rtol=1e-5, max_abs_error=1e-4, matched_ratio=0.9999),
    "float16": dict(atol=1e-3, rtol=1e-3, max_abs_error=1e-2, matched_ratio=0.999),
    "bfloat16": dict(atol=1e-2, rtol=1e-2, max_abs_error=1e-1, matched_ratio=0.999),
    "float8_e4m3fn": dict(atol=1e-1, rtol=1e-1, max_abs_error=2e-1, matched_ratio=0.99),
    "float8_e5m2": dict(atol=1e-1, rtol=1e-1, max_abs_error=2e-1, matched_ratio=0.99),
    "hifloat8": dict(atol=1e-1, rtol=1e-1, max_abs_error=2e-1, matched_ratio=0.99),
    "int8": dict(atol=0.0, rtol=0.0, max_abs_error=0.0, matched_ratio=1.0),
    "int16": dict(atol=0.0, rtol=0.0, max_abs_error=0.0, matched_ratio=1.0),
    "int32": dict(atol=0.0, rtol=0.0, max_abs_error=0.0, matched_ratio=1.0),
    "uint8": dict(atol=0.0, rtol=0.0, max_abs_error=0.0, matched_ratio=1.0),
    "uint16": dict(atol=0.0, rtol=0.0, max_abs_error=0.0, matched_ratio=1.0),
    "bool": dict(atol=0.0, rtol=0.0, max_abs_error=0.0, matched_ratio=1.0),
}

FAMILY_BY_NAME = {
    "relu": "elementwise_unary",
    "abs": "elementwise_unary",
    "exp": "elementwise_unary",
    "neg": "elementwise_unary",
    "sigmoid": "elementwise_unary",
    "tanh": "elementwise_unary",
    "rsqrt": "elementwise_unary",
    "sqrt": "elementwise_unary",
    "log": "elementwise_unary",
    "cast": "elementwise_unary",
    "add": "elementwise_binary",
    "sub": "elementwise_binary",
    "mul": "elementwise_binary",
    "div": "elementwise_binary",
    "maximum": "elementwise_binary",
    "minimum": "elementwise_binary",
    "reduce_sum": "row_reduce",
    "reduce_max": "row_reduce",
    "reduce_min": "row_reduce",
    "reduce_mean": "row_reduce",
    "softmax": "softmax",
    "rmsnorm": "rmsnorm",
    "layernorm": "layernorm",
    "gemm": "gemm",
    "matmul": "gemm",
    "mm": "gemm",
    "gemm_relu": "gemm",
}

UNARY_EXPR = {
    "relu": "T.max({x}, {zero})",
    "abs": "T.abs({x})",
    "exp": "T.exp({x})",
    "neg": "-({x})",
    "sigmoid": "T.sigmoid({x})",
    "tanh": "T.tanh({x})",
    "rsqrt": "T.rsqrt({x})",
    "sqrt": "T.sqrt({x})",
    "log": "T.log({x})",
}

BINARY_EXPR = {
    "add": "{a} + {b}",
    "sub": "{a} - {b}",
    "mul": "{a} * {b}",
    "div": "{a} / {b}",
    "maximum": "T.max({a}, {b})",
    "minimum": "T.min({a}, {b})",
}

REDUCE_OP = {
    "reduce_sum": "sum",
    "reduce_max": "max",
    "reduce_min": "min",
    "reduce_mean": "sum",
}


def normalize_dtype(dtype: str) -> str:
    key = str(dtype).strip().lower()
    key = DTYPE_ALIASES.get(key, key)
    if key not in A5_DTYPES:
        raise ValueError(f"dtype {dtype!r} is not in the Ascend 950 dtype set")
    return key


@dataclass
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    optional: bool = False
    values: str = ""
    requires: tuple[str, ...] = ()
    mutated: bool = False
    contiguous: bool = False
    role: str = "compute"  # compute | metadata | state
    raw_shape: str = ""

    def __post_init__(self) -> None:
        self.shape = tuple(int(v) for v in self.shape)
        self.dtype = normalize_dtype(self.dtype)
        if not self.shape or any(v <= 0 for v in self.shape):
            raise ValueError(f"{self.name}: shape must contain positive static dims, got {self.shape}")
        if isinstance(self.requires, str):
            self.requires = (self.requires,)
        else:
            self.requires = tuple(self.requires or ())
        if not self.raw_shape:
            self.raw_shape = "[" + ", ".join(str(v) for v in self.shape) + "]"

    @property
    def numel(self) -> int:
        n = 1
        for v in self.shape:
            n *= v
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }
        if self.optional:
            data["optional"] = True
        if self.values:
            data["values"] = self.values
        if self.requires:
            data["requires"] = list(self.requires)
        if self.mutated:
            data["mutated"] = True
        if self.contiguous:
            data["contiguous"] = True
        if self.role != "compute":
            data["role"] = self.role
        if self.raw_shape and self.raw_shape != "[" + ", ".join(str(v) for v in self.shape) + "]":
            data["raw_shape"] = self.raw_shape
        return data


@dataclass
class PrecisionSpec:
    atol: float
    rtol: float
    max_abs_error: float
    matched_ratio: float

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclass
class PerfSpec:
    metric: str = "latency"
    target: float | None = None
    unit: str = "ms"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class UnsupportedOperator(ValueError):
    """Raised when a parsed operator cannot be generated by the current templates."""

    def __init__(self, reasons: list[str], spec: "TaskSpec | None" = None) -> None:
        self.reasons = list(reasons) or ["unsupported operator"]
        self.spec = spec
        super().__init__("; ".join(self.reasons))


@dataclass
class TaskSpec:
    name: str
    family: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    formula: str = ""
    precision: PrecisionSpec | None = None
    perf: PerfSpec = field(default_factory=PerfSpec)
    attrs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    version: str = "1"
    supported: bool = True
    unsupported_reasons: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in {
            "elementwise_unary",
            "elementwise_binary",
            "row_reduce",
            "softmax",
            "rmsnorm",
            "layernorm",
            "gemm",
            "attention",
            "unsupported",
        }:
            raise ValueError(f"unsupported family: {self.family}")
        if not self.name.isidentifier():
            raise ValueError(f"op name must be a Python identifier, got {self.name!r}")
        self.unsupported_reasons = [str(r) for r in (self.unsupported_reasons or [])]
        if not self.supported and not self.unsupported_reasons:
            self.unsupported_reasons = ["operator is marked unsupported"]
        if self.precision is None:
            dtype_source = self.outputs[0].dtype if self.outputs else self.inputs[0].dtype
            self.precision = default_precision(dtype_source)

    @property
    def output(self) -> TensorSpec:
        return self.outputs[0]

    @property
    def input(self) -> TensorSpec:
        return self.inputs[0]

    @property
    def metadata_inputs(self) -> list[TensorSpec]:
        return [t for t in self.inputs if t.role == "metadata"]

    @property
    def compute_inputs(self) -> list[TensorSpec]:
        return [t for t in self.inputs if t.role != "metadata"]

    @property
    def reduce_op(self) -> str | None:
        return REDUCE_OP.get(self.name)

    @property
    def unary_expr(self) -> str | None:
        return UNARY_EXPR.get(self.name)

    @property
    def binary_expr(self) -> str | None:
        return BINARY_EXPR.get(self.name)

    def bytes(self) -> int:
        read = sum(t.nbytes for t in self.inputs)
        write = sum(t.nbytes for t in self.outputs)
        return read + write

    def flops(self) -> int:
        if self.family == "gemm":
            m, k = self.inputs[0].shape
            n = self.inputs[1].shape[0]
            return 2 * m * n * k
        total = self.input.numel
        if self.family == "elementwise_binary":
            return 4 * total
        if self.family in ("elementwise_unary", "row_reduce"):
            return 2 * total
        if self.family == "softmax":
            return 8 * total
        if self.family in ("rmsnorm", "layernorm"):
            return 8 * total
        return 0

    def arithmetic_intensity(self) -> float:
        nbytes = max(self.bytes(), 1)
        return self.flops() / nbytes

    def validate(self) -> list[str]:
        issues: list[str] = []
        for tensor in self.inputs + self.outputs:
            if tensor.dtype not in A5_DTYPES:
                issues.append(f"{tensor.name}: unsupported dtype {tensor.dtype}")
        if not self.supported or self.family in ("attention", "unsupported"):
            # Manifest-parsed operators still get structural checks, but their
            # tensor contract is owned by the manifest, not by the templates.
            if not self.inputs:
                issues.append("operator has no inputs")
            return issues
        if self.family == "elementwise_binary" and len(self.inputs) != 2:
            issues.append("elementwise_binary requires exactly two inputs")
        if self.family in ("elementwise_unary", "row_reduce", "softmax") and len(self.inputs) != 1:
            issues.append(f"{self.family} requires exactly one input")
        if not self.outputs:
            issues.append(f"{self.family} requires at least one output")
            return issues
        if self.family == "rmsnorm":
            if len(self.inputs) not in (2,):
                issues.append("rmsnorm requires inputs [x, weight]")
            elif self.inputs[1].shape != (self.inputs[0].shape[-1],):
                issues.append("rmsnorm weight must have shape [D]")
        if self.family == "layernorm":
            if len(self.inputs) not in (2, 3):
                issues.append("layernorm requires inputs [x, weight] or [x, weight, bias]")
        if self.family == "gemm":
            if len(self.inputs) != 2:
                issues.append("gemm requires inputs [A, B]")
            else:
                a, b = self.inputs
                if len(a.shape) != 2 or len(b.shape) != 2:
                    issues.append("gemm inputs must be rank-2")
                elif a.shape[1] != b.shape[1]:
                    issues.append("gemm expects A [M,K] and B [N,K] (transpose_B)")
                elif tuple(self.output.shape) != (a.shape[0], b.shape[0]):
                    issues.append("gemm output must be [M,N]")
        elif self.family == "row_reduce":
            expected = tuple(self.input.shape[:-1])
            if tuple(self.output.shape) != expected:
                issues.append(
                    f"row_reduce output must be {expected}, got {self.output.shape}"
                )
        elif tuple(self.output.shape) != tuple(self.input.shape):
            issues.append(
                f"output shape {self.output.shape} must match input {self.input.shape} for {self.family}"
            )
        return issues

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "formula": self.formula,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "precision": self.precision.to_dict() if self.precision else None,
            "perf": self.perf.to_dict(),
            "attrs": self.attrs,
            "notes": self.notes,
            "version": self.version,
        }
        if not self.supported:
            data["supported"] = False
            data["unsupported_reasons"] = list(self.unsupported_reasons)
        if self.manifest:
            data["manifest"] = self.manifest
        return data

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        inputs = [_tensor_from_dict(t) for t in data["inputs"]]
        outputs = [_tensor_from_dict(t) for t in data["outputs"]]
        prec = data.get("precision")
        precision = PrecisionSpec(**prec) if prec else None
        perf_data = data.get("perf") or {}
        perf = PerfSpec(**perf_data) if perf_data else PerfSpec()
        family = data.get("family") or FAMILY_BY_NAME.get(str(data["name"]).lower(), "")
        if not family:
            family = infer_family(str(data["name"]))
        return cls(
            name=data["name"],
            family=family,
            inputs=inputs,
            outputs=outputs,
            formula=data.get("formula", ""),
            precision=precision,
            perf=perf,
            attrs=data.get("attrs") or {},
            notes=data.get("notes", ""),
            version=str(data.get("version", "1")),
            supported=bool(data.get("supported", True)),
            unsupported_reasons=list(data.get("unsupported_reasons") or []),
            manifest=dict(data.get("manifest") or {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpec":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))


def _tensor_from_dict(data: dict[str, Any]) -> TensorSpec:
    return TensorSpec(
        name=data["name"],
        shape=tuple(data["shape"]),
        dtype=data["dtype"],
        optional=bool(data.get("optional", False)),
        values=str(data.get("values", "")),
        requires=tuple(data.get("requires") or ()),
        mutated=bool(data.get("mutated", False)),
        contiguous=bool(data.get("contiguous", False)),
        role=str(data.get("role", "compute")),
        raw_shape=str(data.get("raw_shape", "")),
    )


def default_precision(dtype: str) -> PrecisionSpec:
    key = normalize_dtype(dtype)
    if key in DEFAULT_PRECISION:
        return PrecisionSpec(**DEFAULT_PRECISION[key])
    if key in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "bool"):
        return PrecisionSpec(atol=0.0, rtol=0.0, max_abs_error=0.0, matched_ratio=1.0)
    return PrecisionSpec(**DEFAULT_PRECISION["float32"])


def infer_family(name: str) -> str:
    key = name.lower()
    if key in FAMILY_BY_NAME:
        return FAMILY_BY_NAME[key]
    for prefix, family in (
        ("reduce_", "row_reduce"),
        ("softmax", "softmax"),
        ("rmsnorm", "rmsnorm"),
        ("layernorm", "layernorm"),
    ):
        if key.startswith(prefix):
            return family
    raise ValueError(f"cannot infer operator family from name {name!r}")


def spec_from_requirement(text: str) -> TaskSpec:
    """Best-effort natural-language -> TaskSpec conversion.

    Supports the common shape: "<op> on [M, N] float32 -> [M, N] float32".
    Anything ambiguous is rejected instead of guessed, so the caller can ask.
    """
    import re

    lowered = text.lower()
    name = None
    for candidate in FAMILY_BY_NAME:
        if re.search(rf"\b{re.escape(candidate)}\b", lowered):
            name = candidate
            break
    if name is None:
        raise ValueError("cannot identify operator name; provide an explicit `name` field")
    family = FAMILY_BY_NAME[name]
    shapes = re.findall(r"\[([0-9,\s]+)\]", text)
    if not shapes:
        raise ValueError("cannot identify tensor shape; provide an explicit spec YAML")
    shape = tuple(int(v) for v in shapes[0].replace(" ", "").split(",") if v)
    dtype_match = re.search(r"\b(float32|float16|bfloat16|bf16|fp16|fp32|int8|int32)\b", lowered)
    dtype = normalize_dtype(dtype_match.group(1)) if dtype_match else "float32"
    out_dtype = dtype
    if family == "gemm":
        if len(shapes) < 2:
            raise ValueError("gemm requires two shapes, e.g. [M,K] and [N,K]")
        bshape = tuple(int(v) for v in shapes[1].replace(" ", "").split(",") if v)
        m, k = shape
        n = bshape[0]
        out = TensorSpec("C", (m, n), "float32")
        inputs = [TensorSpec("A", (m, k), dtype), TensorSpec("B", (n, k), dtype)]
    elif family == "elementwise_binary":
        inputs = [TensorSpec("A", shape, dtype), TensorSpec("B", shape, dtype)]
        out = TensorSpec("C", shape, out_dtype)
    elif family in ("rmsnorm", "layernorm"):
        inputs = [TensorSpec("X", shape, dtype), TensorSpec("W", (shape[-1],), dtype)]
        out = TensorSpec("Y", shape, out_dtype)
    else:
        inputs = [TensorSpec("X", shape, dtype)]
        out = TensorSpec("Y", shape, out_dtype)
    return TaskSpec(
        name=name,
        family=family,
        inputs=inputs,
        outputs=[out],
        formula=text,
    )
