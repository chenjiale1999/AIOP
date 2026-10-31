"""TileOPs manifest ingestion (``src/tileops/manifest/spec/*.yaml``).

The manifest is the source of truth for an operator's external contract: a
polymorphic signature over ``forall`` indices, workload rows that instantiate
that signature, optional tensors whose presence follows a discriminant, and
metadata tensors whose contents come from generators (``values``/``requires``).

This module binds *one* workload row plus *one* dtype case into a concrete,
fully static :class:`~aiop.specs.TaskSpec`, so the rest of the agent pipeline
(plan → generate → static → compile → simulate) stays dtype/shape driven and
never parses expressions again.

Two outcomes are possible:

* the entry maps onto an aiop template family (elementwise, row reduce,
  softmax, norm, gemm) and the bound spec is runnable;
* the entry has no A5 template (attention, MoE, convolution, ...). The bound
  spec is still produced with ``supported=False`` and human-readable
  ``unsupported_reasons``; ``build_plan`` turns that into
  :class:`~aiop.specs.UnsupportedOperator` and the agent reports
  ``blocked_unsupported`` instead of fabricating a kernel.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import PROJECT_ROOT, discover_manifest_dir
from .specs import TaskSpec, TensorSpec, normalize_dtype

# Resolved lazily: a sibling TileOPs checkout, then the repository's bundled
# examples/spec copies. No user-specific fixed directory is assumed.
DEFAULT_MANIFEST_DIR = Path(
    discover_manifest_dir() or (PROJECT_ROOT / "examples" / "spec")
)


class ManifestError(ValueError):
    """The manifest source is malformed or the entry does not exist."""


class BindError(ValueError):
    """A workload row cannot be bound to its entry's signature."""


# --------------------------------------------------------------------------
# op name -> (aiop op name, aiop family). Everything else is unsupported.
# --------------------------------------------------------------------------
OP_FAMILY: dict[str, tuple[str, str]] = {
    "ReluFwdOp": ("relu", "elementwise_unary"),
    "AbsFwdOp": ("abs", "elementwise_unary"),
    "NegFwdOp": ("neg", "elementwise_unary"),
    "ExpFwdOp": ("exp", "elementwise_unary"),
    "SqrtFwdOp": ("sqrt", "elementwise_unary"),
    "RsqrtFwdOp": ("rsqrt", "elementwise_unary"),
    "LogFwdOp": ("log", "elementwise_unary"),
    "SigmoidFwdOp": ("sigmoid", "elementwise_unary"),
    "TanhFwdOp": ("tanh", "elementwise_unary"),
    "AddFwdOp": ("add", "elementwise_binary"),
    "SubFwdOp": ("sub", "elementwise_binary"),
    "MulFwdOp": ("mul", "elementwise_binary"),
    "DivFwdOp": ("div", "elementwise_binary"),
    "MaximumFwdOp": ("maximum", "elementwise_binary"),
    "MinimumFwdOp": ("minimum", "elementwise_binary"),
    "SoftmaxFwdOp": ("softmax", "softmax"),
    "RMSNormFwdOp": ("rmsnorm", "rmsnorm"),
    "LayerNormFwdOp": ("layernorm", "layernorm"),
    "GemmFwdOp": ("gemm", "gemm"),
    "SumFwdOp": ("reduce_sum", "row_reduce"),
    "MeanFwdOp": ("reduce_mean", "row_reduce"),
    "AmaxFwdOp": ("reduce_max", "row_reduce"),
    "AminFwdOp": ("reduce_min", "row_reduce"),
}

FAMILY_LABEL = {
    "elementwise_unary": "elementwise unary",
    "elementwise_binary": "elementwise binary",
    "row_reduce": "last-axis reduction",
    "softmax": "row softmax",
    "rmsnorm": "RMSNorm",
    "layernorm": "LayerNorm",
    "gemm": "persistent GEMM",
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def manifest_files(source: str | Path | None = None) -> list[Path]:
    path = Path(source) if source else Path(discover_manifest_dir() or DEFAULT_MANIFEST_DIR)
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    files = [p for p in files if p.is_file()]
    if not files:
        raise ManifestError(f"no manifest yaml found at {path}")
    return files


_CACHE: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def load_manifest(source: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Merge every manifest file into ``op name -> entry`` (mtime-cached)."""
    files = manifest_files(source)
    key = "|".join(str(path) for path in files)
    stamp = max(path.stat().st_mtime for path in files)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    ops: dict[str, dict[str, Any]] = {}
    for file in files:
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # pragma: no cover - malformed input
            raise ManifestError(f"{file}: {exc}") from exc
        if data is None:
            continue
        if not isinstance(data, dict):
            raise ManifestError(f"{file}: top level must be a mapping of op name -> entry")
        if set(data) <= {"adts"}:  # types.yaml
            continue
        for name, entry in data.items():
            if name == "adts":
                continue
            if not isinstance(entry, dict):
                raise ManifestError(f"{file}: entry {name!r} must be a mapping")
            if name in ops:
                raise ManifestError(f"duplicate op {name!r} in {file}")
            record = dict(entry)
            record.setdefault("family", file.stem)
            record["_source"] = str(file)
            ops[name] = record
    _CACHE[key] = (stamp, ops)
    return ops


def load_operator(source: str | Path | None, op_name: str) -> dict[str, Any]:
    entries = load_manifest(source)
    if op_name not in entries:
        known = ", ".join(sorted(entries)[:8])
        raise ManifestError(f"op {op_name!r} not found in manifest {source} (e.g. {known}, ...)")
    return entries[op_name]


def list_operators(source: str | Path | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, entry in sorted(load_manifest(source).items()):
        rows = entry.get("workloads") or []
        mapped = OP_FAMILY.get(name)
        out.append(
            {
                "op": name,
                "manifest_family": entry.get("family"),
                "status": entry.get("status"),
                "ref_api": entry.get("ref_api"),
                "aiop_name": mapped[0] if mapped else None,
                "aiop_family": mapped[1] if mapped else None,
                "workloads": len(rows),
                "cases": sum(len(r.get("dtype_cases") or [{}]) for r in rows),
                "params": sorted((entry.get("signature") or {}).get("params") or {}),
                "source": entry.get("_source"),
            }
        )
    return out


# --------------------------------------------------------------------------
# signature parsing
# --------------------------------------------------------------------------
def parse_kind(text: Any) -> tuple[str, list[str] | None]:
    """``Dim`` / ``Shape`` / ``DType[a | b]`` / ``Seq[Int]`` -> (kind, members)."""
    raw = str(text).strip()
    if raw == "Dim":
        return "dim", None
    if raw == "Shape":
        return "shape", None
    if raw in ("Seq[Int]", "Seq[Int, ...]") or raw.startswith("Seq["):
        return "seq", None
    if raw.startswith("DType[") and raw.endswith("]"):
        members = [m.strip() for m in raw[len("DType[") : -1].split("|") if m.strip()]
        return "dtype", members
    raise ManifestError(f"unknown forall kind {raw!r}")


def _parse_forall(forall: dict[str, Any] | None) -> dict[str, tuple[str, list[str] | None]]:
    return {name: parse_kind(spec) for name, spec in (forall or {}).items()}


def _is_dtype_type(type_text: Any) -> bool:
    if not isinstance(type_text, str):
        return False
    names = {m.strip().strip("'\"") for m in type_text.split("|")}
    names.discard("None")
    names.discard("bool")  # `bool` is a Python parameter type, not a dtype
    return bool(names) and all(name in _A5_DTYPE_NAMES for name in names)


_A5_DTYPE_NAMES = {
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint16",
    "uint32",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "float8_e4m3fn",
    "float8_e5m2",
    "hifloat8",
}


# --------------------------------------------------------------------------
# expression language (a closed subset of docs/design/manifest.md table 10)
# --------------------------------------------------------------------------
class _Unresolved(Exception):
    """An expression name has no value on the bound branch."""


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _parse(text: str) -> ast.Expression:
    try:
        return ast.parse(str(text).strip(), mode="eval")
    except SyntaxError as exc:  # pragma: no cover - malformed input
        raise ManifestError(f"cannot parse expression {text!r}: {exc.msg}") from exc


def _prod(seq: Iterable[Any]) -> int:
    result = 1
    for value in seq:
        result *= int(value)
    return result


def _broadcast(*shapes: Iterable[Any]) -> tuple[int, ...]:
    if not shapes:
        return ()
    result = tuple(int(v) for v in shapes[0])
    for other in shapes[1:]:
        a = result
        b = tuple(int(v) for v in other)
        if a == b:
            continue
        n = max(len(a), len(b))
        a = (1,) * (n - len(a)) + a
        b = (1,) * (n - len(b)) + b
        out: list[int] = []
        for x, y in zip(a, b):
            if x == 1:
                out.append(y)
            elif y == 1:
                out.append(x)
            elif x == y:
                out.append(x)
            else:
                raise BindError(f"cannot broadcast {result} with {b}")
        result = tuple(out)
    return result


def _norm_axis(axis: int, rank: int) -> int:
    axis = int(axis)
    if rank == 0:
        if axis in (0, -1):
            return 0
        raise BindError(f"axis {axis} is invalid for a rank-0 tensor")
    if not -rank <= axis < rank:
        raise BindError(f"axis {axis} is outside [-{rank}, {rank})")
    return axis % rank


def _reduced(shape: Iterable[Any], dim: Any, keepdim: Any = False, mode: str = "full") -> tuple[int, ...]:
    dims = tuple(int(v) for v in shape)
    if dim is None:
        if mode == "reject":
            raise _Unresolved("reduced(..., dim=None, mode='reject')")
        axes = list(range(len(dims)))
    elif isinstance(dim, int):
        axes = [dim]
    else:
        axes = list(dim)
    norm = {_norm_axis(a, len(dims)) for a in axes}
    if keepdim:
        return tuple(1 if i in norm else v for i, v in enumerate(dims))
    return tuple(v for i, v in enumerate(dims) if i not in norm)


def _coalesce_dtype(first: Any, second: Any) -> Any:
    if first is None or first == "":
        return second
    return first


def _repeat(value: Any, count: Any) -> list[int]:
    return [int(value)] * int(count)


def _prefix_sum(seq: Iterable[Any]) -> list[int]:
    out = [0]
    for value in seq:
        out.append(out[-1] + int(value))
    return out


def _exclusive_prefix_sum(seq: Iterable[Any]) -> list[int]:
    out = [0]
    for value in list(seq)[:-1]:
        out.append(out[-1] + int(value))
    return out


def _ceil_div(numerator: Any, denominator: Any) -> int:
    return -(-int(numerator) // int(denominator))


def _balanced_sizes(total: Any, count: Any) -> list[int]:
    count = int(count)
    if count <= 0:
        raise BindError(f"balanced_sizes needs a positive count, got {count}")
    base, remainder = divmod(int(total), count)
    return [base + 1] * remainder + [base] * (count - remainder)


def _per_axis(value: Any, index: Any, ndim: Any, fallback: Any = None) -> Any:
    """``per_axis(kernel_size, 0, 2)``: one axis of a per-dimension sequence."""
    if isinstance(value, (list, tuple)):
        items = [int(v) for v in value]
        if len(items) == int(ndim):
            return items[int(index)]
        if len(items) == 1:
            return items[0]
        raise BindError(f"per_axis: {len(items)} values for {ndim} axes")
    if isinstance(value, (int, float)):
        return int(value)
    if fallback is not None:
        return fallback
    raise _Unresolved(f"per_axis({value!r}, ...)")


def _pool_out(
    size: Any,
    kernel: Any,
    stride: Any,
    padding: Any,
    dilation: Any = 1,
    ceil_mode: Any = False,
) -> int:
    span = int(size) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1
    out = _ceil_div(span, stride) + 1 if ceil_mode else span // int(stride) + 1
    if ceil_mode and (out - 1) * int(stride) >= int(size) + int(padding):
        out -= 1
    return out


def _conv_out(size: Any, kernel: Any, stride: Any, padding: Any, dilation: Any) -> int:
    span = int(size) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1
    return span // int(stride) + 1


def _valid_axes(dim: Any, rank: Any) -> bool:
    if dim is None:
        return True
    axes = [dim] if isinstance(dim, int) else list(dim)
    try:
        for axis in axes:
            _norm_axis(axis, int(rank))
    except BindError:
        return False
    return True


def _unique_axes(dim: Any, rank: Any) -> bool:
    if dim is None:
        return True
    axes = [dim] if isinstance(dim, int) else list(dim)
    try:
        norm = [_norm_axis(axis, int(rank)) for axis in axes]
    except BindError:
        return False
    return len(set(norm)) == len(norm)


def _representable(value: Any, dtype: Any) -> bool:
    name = str(dtype)
    if name.startswith("float") or name in ("bfloat16", "hifloat8"):
        return value is not None
    if name.startswith(("int", "uint")):
        return float(value).is_integer()
    return True


def _promote_int_to_float(dtype: Any) -> str:
    name = str(dtype)
    return "float32" if name.startswith(("int", "uint", "bool")) else name


_FUNCS = {
    "prod": _prod,
    "len": len,
    "broadcast": _broadcast,
    "reduced": _reduced,
    "coalesce_dtype": _coalesce_dtype,
    "promote_int_to_float": _promote_int_to_float,
    "repeat": _repeat,
    "as_tensor": lambda seq: list(seq),
    "prefix_sum": _prefix_sum,
    "exclusive_prefix_sum": _exclusive_prefix_sum,
    "ceil_div": _ceil_div,
    "ceil": math.ceil,
    "floor": math.floor,
    "balanced_sizes": _balanced_sizes,
    "per_axis": _per_axis,
    "pool.out": _pool_out,
    "conv.out": _conv_out,
    "valid_axes": _valid_axes,
    "unique_axes": _unique_axes,
    "representable": _representable,
    "sum": lambda seq: sum(int(v) for v in seq),
    "max": max,
    "min": min,
    "all": lambda seq: all(bool(v) for v in seq),
    "any": lambda seq: any(bool(v) for v in seq),
    "abs": abs,
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def evaluate(node: ast.AST, env: dict[str, Any], presence: dict[str, bool]) -> Any:
    """Evaluate a manifest expression against bound indices and presence."""
    if isinstance(node, ast.Expression):
        return evaluate(node.body, env, presence)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "inf":
            return math.inf
        if node.id in env:
            return env[node.id]
        if node.id in _A5_DTYPE_NAMES:  # a constant dtype literal
            return node.id
        raise _Unresolved(node.id)
    if isinstance(node, ast.Attribute):
        dotted = _dotted(node)
        if dotted in _FUNCS:  # namespaced primitive used as a value
            return dotted
        if node.attr == "value":
            base = evaluate(node.value, env, presence)
            if base is None:
                raise _Unresolved(dotted)
            return base
        if node.attr == "kind":
            base = evaluate(node.value, env, presence)
            if isinstance(base, dict) and "kind" in base:
                return base["kind"]
            raise _Unresolved(dotted)
        raise _Unresolved(dotted)
    if isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand, env, presence)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Not):
            return not value
        raise _Unresolved(ast.unparse(node))
    if isinstance(node, ast.BinOp):
        left = evaluate(node.left, env, presence)
        right = evaluate(node.right, env, presence)
        for ast_op, fn in _BINOPS.items():
            if isinstance(node.op, ast_op):
                return fn(left, right)
        raise _Unresolved(ast.unparse(node))
    if isinstance(node, ast.BoolOp):
        values = node.values
        if isinstance(node.op, ast.And):
            result: Any = True
            for sub in values:
                result = evaluate(sub, env, presence)
                if not result:
                    return result
            return result
        result = False
        for sub in values:
            result = evaluate(sub, env, presence)
            if result:
                return result
        return result
    if isinstance(node, ast.Compare):
        left = evaluate(node.left, env, presence)
        for op, comparator in zip(node.ops, node.comparators):
            right = evaluate(comparator, env, presence)
            matched = False
            for ast_op, fn in _CMPOPS.items():
                if isinstance(op, ast_op):
                    matched = True
                    if not fn(left, right):
                        return False
                    break
            if not matched:
                raise _Unresolved(ast.unparse(node))
            left = right
        return True
    if isinstance(node, ast.IfExp):
        branch = node.body if evaluate(node.test, env, presence) else node.orelse
        return evaluate(branch, env, presence)
    if isinstance(node, ast.Tuple):
        return tuple(evaluate(elt, env, presence) for elt in node.elts)
    if isinstance(node, ast.List):
        return [evaluate(elt, env, presence) for elt in node.elts]
    if isinstance(node, ast.Subscript):
        base = evaluate(node.value, env, presence)
        index = evaluate(node.slice, env, presence)
        return base[index]
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = _dotted(node.func)
        else:
            raise _Unresolved(ast.unparse(node))
        if name == "present":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                raise ManifestError("present(x) takes a single name")
            target = node.args[0].id
            if target in presence:
                return bool(presence[target])
            if target in env:
                return env[target] is not None
            return False
        fn = _FUNCS.get(name)
        if fn is None:
            raise _Unresolved(f"unknown primitive {name}()")
        args = [evaluate(arg, env, presence) for arg in node.args]
        kwargs = {kw.arg: evaluate(kw.value, env, presence) for kw in node.keywords if kw.arg}
        return fn(*args, **kwargs)
    raise _Unresolved(ast.unparse(node))


def eval_text(text: Any, env: dict[str, Any], presence: dict[str, bool]) -> Any:
    return evaluate(_parse(str(text)), env, presence)


# --------------------------------------------------------------------------
# shapes and dtypes
# --------------------------------------------------------------------------
def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    tail = "".join(current)
    if tail.strip():
        parts.append(tail)
    return [part.strip() for part in parts if part.strip()]


class _LazyEnv(dict):
    """Env whose type-family formals are evaluated only where the branch uses them."""

    def __init__(
        self,
        base: dict[str, Any],
        exprs: dict[str, ast.Expression],
        presence: dict[str, bool],
    ) -> None:
        super().__init__(base)
        self._exprs = exprs
        self._presence = presence

    def __contains__(self, key: object) -> bool:
        return key in self._exprs or super().__contains__(key)

    def __getitem__(self, key: str) -> Any:
        if key in self._exprs:
            expr = self._exprs.pop(key)
            value = evaluate(expr, self, self._presence)
            dict.__setitem__(self, key, value)
            return value
        return super().__getitem__(key)


def _select_type_case(family: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    match = family.get("match")
    if isinstance(match, (list, tuple)):
        names = [str(name).strip() for name in match]
    elif match:
        names = [name.strip() for name in str(match).split(",")]
    else:
        names = []
    if len(names) == 1:
        key: Any = env[names[0]]
    else:
        key = tuple(env[name] for name in names)
    for case in family.get("cases") or []:
        when = case.get("when")
        if isinstance(when, dict):
            raise BindError("algebraic-data-type type families are not bound by aiop")
        if isinstance(when, list):
            when = tuple(when)
        if when == key:
            return case
    raise BindError(f"no type-family case matches {key!r}")


def eval_shape(
    text: Any,
    entry: dict[str, Any],
    env: dict[str, Any],
    presence: dict[str, bool],
) -> tuple[int, ...]:
    raw = str(text).strip()
    types = (entry.get("signature") or {}).get("types") or {}
    for name, family in types.items():
        match = re.fullmatch(rf"{re.escape(name)}\[(.*)\]", raw, re.S)
        if not match:
            continue
        args = _split_top_level(match.group(1))
        formals = list((family.get("params") or {}).keys())
        if len(args) != len(formals):
            raise BindError(
                f"{name}[...] takes {len(formals)} arguments, got {len(args)}"
            )
        exprs = {formal: _parse(arg) for formal, arg in zip(formals, args)}
        local = _LazyEnv(env, exprs, presence)
        case = _select_type_case(family, local)
        return eval_shape(case.get("is"), entry, local, presence)
    return _eval_axes(raw, env, presence)


def _eval_axes(text: str, env: dict[str, Any], presence: dict[str, bool]) -> tuple[int, ...]:
    inner = text.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    axes: list[int] = []
    for part in _split_top_level(inner):
        if part.startswith("*"):
            value = evaluate(_parse(part[1:]), env, presence)
            axes.extend(int(v) for v in value)
        else:
            axes.append(int(evaluate(_parse(part), env, presence)))
    return tuple(axes)


def eval_dtype(text: Any, env: dict[str, Any], presence: dict[str, bool]) -> str:
    value = evaluate(_parse(str(text)), env, presence)
    if value is None:
        raise BindError(f"dtype expression {text!r} resolved to None")
    return normalize_dtype(str(value))


def _flag(value: Any, env: dict[str, Any], presence: dict[str, bool]) -> bool:
    if value is None or isinstance(value, bool):
        return bool(value)
    try:
        return bool(eval_text(value, env, presence))
    except _Unresolved:
        return False


# --------------------------------------------------------------------------
# binding
# --------------------------------------------------------------------------
def _select_workload(rows: list[dict[str, Any]], selector: Any) -> tuple[int, dict[str, Any]]:
    if isinstance(selector, int):
        if not 0 <= selector < len(rows):
            raise BindError(f"workload index {selector} outside 0..{len(rows) - 1}")
        return selector, rows[selector]
    for index, row in enumerate(rows):
        if row.get("label") == selector:
            return index, row
    raise BindError(f"no workload labelled {selector!r}")


def _resolve_presence(
    inputs: dict[str, Any],
    env: dict[str, Any],
    some: set[str],
) -> dict[str, bool]:
    presence: dict[str, bool] = {}
    pending = dict(inputs)
    for _ in range(len(pending) + 2):
        progressed = False
        for name, spec in list(pending.items()):
            optional = spec.get("optional")
            if optional is None or optional is False:
                presence[name] = True
            elif optional is True:
                presence[name] = name in some
            else:
                try:
                    presence[name] = bool(eval_text(optional, env, presence))
                except _Unresolved:
                    continue
            pending.pop(name)
            progressed = True
        if not pending or not progressed:
            break
    if pending:
        raise BindError(
            "cannot resolve tensor presence for "
            + ", ".join(sorted(pending))
            + " (check the row's `some` list)"
        )
    return presence


_GENERATOR_LENGTH = {
    # generator -> (length mode, offset added to len(seq))
    "as_tensor": ("len", 0),
    "prefix_sum": ("len", 1),
    "exclusive_prefix_sum": ("len", 1),
    "chunk_offsets": ("len", 1),
    "packed_positions": ("sum", 0),
    "token_indices": ("sum", 0),
    "chunk_indices": ("chunks", 0),
}

_SINGLE_AXIS = re.compile(r"^\[\s*([A-Za-z_]\w*)\s*(?:\+\s*(\d+))?\s*\]$")
_FIRST_AXIS = re.compile(r"^\[\s*([A-Za-z_]\w*)\s*(?:\+\s*(\d+))?\s*,")


def _generator_length(
    values: Any,
    seqs: dict[str, list[int]],
    env: dict[str, Any],
    presence: dict[str, bool],
) -> int | None:
    """Length of a metadata generator's result when its argument is a Seq index."""
    if not values:
        return None
    try:
        node = _parse(str(values))
    except ManifestError:
        return None
    if isinstance(node, ast.Expression):
        node = node.body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    mode = _GENERATOR_LENGTH.get(node.func.id)
    if mode is None or not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, ast.Name) or first.id not in seqs:
        return None
    seq = seqs[first.id]
    if mode[0] == "len":
        return len(seq) + mode[1]
    if mode[0] == "sum":
        return sum(seq)
    # chunked: one row per chunk of one request
    if len(node.args) < 2:
        return None
    try:
        chunk = int(evaluate(node.args[1], env, presence))
    except (_Unresolved, ValueError, TypeError):
        return None
    if chunk <= 0:
        return None
    return sum(_ceil_div(length, chunk) for length in seq)


def _solve_dims(
    inputs_def: dict[str, Any],
    kinds: dict[str, tuple[str, list[str] | None]],
    env: dict[str, Any],
    presence: dict[str, bool],
) -> None:
    """Unification (docs/design/manifest.md table 19) for unbound ``Dim`` indices.

    Two facts solve a dim that a row does not write:

    * a metadata generator over a ``Seq[Int]`` fixes the length of the tensor
      carrying it (``prefix_sum(seq)`` -> ``len(seq)+1``, ``packed_positions(seq)``
      -> ``sum(seq)``);
    * a varlen batch index ``B`` is the number of requests, i.e. ``len(seq)``.
    """
    seqs = {name: env[name] for name, kind in kinds.items() if kind[0] == "seq" and name in env}

    def _bind(name: str, value: int) -> None:
        if (
            name in kinds
            and kinds[name][0] == "dim"
            and name not in env
            and isinstance(value, int)
            and value > 0
        ):
            env[name] = value

    for spec in inputs_def.values():
        produced = _generator_length(spec.get("values"), seqs, env, presence)
        if produced is None:
            continue
        shape = str(spec.get("shape", "")).strip()
        match = _SINGLE_AXIS.match(shape) or _FIRST_AXIS.match(shape)
        if not match:
            continue
        _bind(match.group(1), produced - int(match.group(2) or 0))

    if not seqs:
        return
    lengths = {len(value) for value in seqs.values()}
    if len(lengths) != 1:
        return
    length = lengths.pop()
    for name, kind in kinds.items():
        if kind[0] == "dim" and name not in env and name.upper() in ("B", "N"):
            env[name] = length


def _apply_let(
    lets: dict[str, Any],
    env: dict[str, Any],
    presence: dict[str, bool],
) -> None:
    pending = dict(lets)
    for _ in range(len(pending) + 1):
        progressed = False
        for name, expr in list(pending.items()):
            try:
                env[name] = eval_text(expr, env, presence)
            except (_Unresolved, BindError, ValueError, TypeError, KeyError, IndexError):
                continue
            pending.pop(name)
            progressed = True
        if not pending or not progressed:
            break


def _check_shape_rules(
    rules: list[Any],
    env: dict[str, Any],
    presence: dict[str, bool],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rule in rules or []:
        text = str(rule)
        try:
            value: Any = bool(eval_text(text, env, presence))
        except (_Unresolved, BindError, ValueError, TypeError, KeyError, IndexError):
            value = None
        out.append({"rule": text, "holds": value})
    return out


def _template_reasons(
    op_name: str,
    family: str,
    inputs: list[TensorSpec],
    outputs: list[TensorSpec],
    env: dict[str, Any],
    row: dict[str, Any],
) -> list[str]:
    """Restrictions the aiop templates impose beyond the manifest contract."""
    reasons: list[str] = []
    label = FAMILY_LABEL.get(family, family)
    if not inputs:
        return ["manifest entry has no call-time inputs"]
    if len(outputs) != 1:
        reasons.append(
            f"entry returns {len(outputs)} outputs; the {label} template writes exactly one"
        )
        return reasons
    src = inputs[0]
    dst = outputs[0]
    if family in ("elementwise_unary", "softmax") and src.dtype != dst.dtype:
        reasons.append(
            f"in-place {label} template keeps the input dtype ({src.dtype}); entry asks for {dst.dtype}"
        )
    if family == "elementwise_unary" and src.dtype not in ("float16", "bfloat16", "float32"):
        reasons.append(f"{label} template computes in fp32 and requires a float input, got {src.dtype}")
    if family == "elementwise_binary":
        if len(inputs) != 2:
            reasons.append(f"entry has {len(inputs)} inputs; the binary template takes exactly two")
        elif inputs[0].shape != inputs[1].shape:
            reasons.append(
                f"operands {list(inputs[0].shape)} and {list(inputs[1].shape)} broadcast; "
                "the A5 elementwise template needs identical shapes"
            )
        elif inputs[0].dtype != inputs[1].dtype:
            reasons.append("the A5 elementwise template needs both operands in one dtype")
        alpha = env.get("alpha", 1)
        if alpha not in (1, 1.0, None):
            reasons.append(f"alpha={alpha} is a fused scale the elementwise template does not apply")
    if family == "row_reduce":
        dim = env.get("dim")
        rank = len(src.shape)
        axes = [rank - 1] if dim is None else ([dim] if isinstance(dim, int) else list(dim))
        norm = sorted({_norm_axis(a, rank) for a in axes})
        if norm != [rank - 1]:
            reasons.append(
                f"reduction axis {dim} is not the last axis; the row template reduces the last axis only"
            )
        if env.get("keepdim"):
            reasons.append("keepdim=True is outside the row-reduce template")
    if family == "softmax":
        dim = env.get("dim")
        if dim is not None:
            rank = len(src.shape)
            if _norm_axis(dim, rank) != rank - 1:
                reasons.append("softmax over a non-last axis needs a layout transform the template lacks")
            elif dim < 0 and dim != -1:
                reasons.append("only dim=-1 is generated")
    if family in ("rmsnorm", "layernorm"):
        normalized = env.get("normalized_shape")
        normalized = tuple(normalized) if isinstance(normalized, (list, tuple)) else ()
        if len(normalized) != 1:
            reasons.append(
                f"normalized_shape={list(normalized)} has rank {len(normalized)}; "
                "the norm template normalizes one trailing axis"
            )
        if len(inputs) < 2:
            reasons.append(
                "optional weight/bias are absent in this row; the A5 norm template always applies them"
            )
        elif len(inputs) >= 2 and inputs[1].dtype != src.dtype:
            reasons.append("weight dtype must match the input dtype for the norm template")
    if family == "gemm":
        if env.get("trans_a"):
            reasons.append("trans_a=True is not emitted by the persistent GEMM template")
        if not env.get("trans_b", True):
            reasons.append("trans_b=False (B is [K, N]) is not emitted; the L1 path needs [N, K]")
        if len(inputs) != 2:
            reasons.append(f"entry has {len(inputs)} inputs; GEMM takes A and B")
    return reasons


def _raw_shape_text(spec: dict[str, Any]) -> str:
    return str(spec.get("shape", "")).strip()


def bind_operator(
    source: str | Path | None,
    op_name: str,
    workload: int | str = 0,
    dtype_case: int = 0,
) -> TaskSpec:
    """Bind one workload row (and one dtype case) of ``op_name`` into a TaskSpec."""
    entry = load_operator(source, op_name)
    signature = entry.get("signature") or {}
    kinds = _parse_forall(signature.get("forall"))
    params_def: dict[str, Any] = dict(signature.get("params") or {})
    rows: list[dict[str, Any]] = list(entry.get("workloads") or [])
    if not rows:
        raise ManifestError(f"{op_name}: entry has no workload rows")

    row_index, row = _select_workload(rows, workload)
    dtype_cases = list(row.get("dtype_cases") or [])
    if dtype_cases:
        if not 0 <= dtype_case < len(dtype_cases):
            raise BindError(f"{op_name}: dtype_case {dtype_case} outside 0..{len(dtype_cases) - 1}")
        dtype_assign = dict(dtype_cases[dtype_case])
    else:
        if dtype_case:
            raise BindError(f"{op_name}: workload row has no dtype_cases")
        dtype_assign = {}
    some = set(row.get("some") or [])

    env: dict[str, Any] = {}
    presence: dict[str, bool] = {}

    # 1. construction parameters (a dtype parameter may also sit in dtype_cases)
    for name, spec in params_def.items():
        if name in dtype_assign:
            env[name] = normalize_dtype(str(dtype_assign[name]))
        elif name in row:
            value = row[name]
            env[name] = normalize_dtype(str(value)) if _is_dtype_type(spec.get("type")) else value
        elif "default" in spec:
            env[name] = spec["default"]
        else:
            raise BindError(f"{op_name}: workload row is missing required parameter {name!r}")

    # 2. forall indices
    for name, (kind, _members) in kinds.items():
        if kind == "dtype":
            # A dtype index is relevant only on branches where it appears, so a
            # row may leave one out (e.g. the cos/sin cache dtype without rope).
            if name in dtype_assign:
                env[name] = normalize_dtype(str(dtype_assign[name]))
            elif name in row:
                env[name] = normalize_dtype(str(row[name]))
        elif kind == "dim":
            if name in row:
                env[name] = int(row[name])
        elif kind == "shape":
            if name in row:
                value = row[name]
                env[name] = tuple(int(v) for v in value) if isinstance(value, (list, tuple)) else (int(value),)
        elif kind == "seq":
            if name in row:
                value = row[name]
                env[name] = eval_text(value, env, presence) if isinstance(value, str) else list(value)

    inputs_def: dict[str, Any] = dict(signature.get("inputs") or {})
    outputs_def: dict[str, Any] = dict(signature.get("outputs") or {})
    _solve_dims(inputs_def, kinds, env, presence)
    _apply_let(signature.get("let") or {}, env, presence)

    # 3. tensors
    presence = _resolve_presence(inputs_def, env, some)

    def _tensor(name: str, spec: dict[str, Any], *, is_output: bool) -> TensorSpec:
        raw_shape = _raw_shape_text(spec)
        try:
            shape = eval_shape(raw_shape, entry, env, presence)
            dtype = eval_dtype(spec.get("dtype"), env, presence)
        except _Unresolved as exc:
            raise BindError(f"{op_name}.{name}: unbound name {exc} in shape/dtype {raw_shape!r}") from exc
        if not shape:
            shape = (1,)  # rank-0 tensors are a single element
        if any(int(v) <= 0 for v in shape):
            raise BindError(f"{op_name}.{name}: non-positive extent in {list(shape)}")
        role = "compute"
        if spec.get("values"):
            role = "metadata"
        elif _flag(spec.get("mutated"), env, presence):
            role = "state"
        elif is_output and spec.get("buffer"):
            role = "state"
        requires = spec.get("requires") or ()
        if isinstance(requires, str):
            requires = (requires,)
        return TensorSpec(
            name=name,
            shape=tuple(shape),
            dtype=dtype,
            optional=not presence.get(name, True),
            values=str(spec.get("values") or ""),
            requires=tuple(requires),
            mutated=_flag(spec.get("mutated"), env, presence),
            contiguous=bool(spec.get("contiguous")),
            role=role,
            raw_shape=raw_shape,
        )

    inputs = [_tensor(name, spec, is_output=False) for name, spec in inputs_def.items() if presence.get(name)]
    outputs = [_tensor(name, spec, is_output=True) for name, spec in outputs_def.items()]

    mapped = OP_FAMILY.get(op_name)
    reasons: list[str] = []
    if mapped is None:
        family = "attention" if str(entry.get("family")) == "attention" else "unsupported"
        reasons.append(
            f"{op_name} is a `{entry.get('family')}` operator; aiop has no A5 template family for it"
        )
        name = op_name
    else:
        name, family = mapped
        reasons = _template_reasons(op_name, family, inputs, outputs, env, row)

    attrs: dict[str, Any] = {
        key: value
        for key, value in env.items()
        if key not in kinds and key not in {t.name for t in inputs + outputs}
    }
    for extra in ("is_causal", "sm_scale", "softcap", "window_size_left", "window_size_right"):
        if extra in row:
            attrs[extra] = row[extra]
    shape_rules = _check_shape_rules(signature.get("shape_rules") or [], env, presence)

    spec = TaskSpec(
        name=name,
        family=family,
        inputs=inputs,
        outputs=outputs,
        formula=str(entry.get("ref_api") or ""),
        attrs=attrs,
        notes=f"TileOPs manifest entry {op_name} ({entry.get('status')}), row {row_index}",
        supported=not reasons,
        unsupported_reasons=reasons,
        manifest={
            "op": op_name,
            "manifest_family": entry.get("family"),
            "status": entry.get("status"),
            "ref_api": entry.get("ref_api"),
            "source": entry.get("_source"),
            "workload_index": row_index,
            "workload_label": row.get("label"),
            "dtype_case": dtype_case,
            "dtype_assignment": {k: str(v) for k, v in dtype_assign.items()},
            "some": sorted(some),
            "present_inputs": [t.name for t in inputs],
            "indices": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in env.items()
                if key in kinds
            },
            "shape_rules": shape_rules,
            "metadata_values": {t.name: t.values for t in inputs if t.values},
            "metadata_requires": {t.name: list(t.requires) for t in inputs if t.requires},
        },
    )
    return spec


def describe_operator(source: str | Path | None, op_name: str) -> dict[str, Any]:
    """Raw entry plus workload labels, for `aiop manifest show`."""
    entry = load_operator(source, op_name)
    mapped = OP_FAMILY.get(op_name)
    rows = entry.get("workloads") or []
    return {
        "op": op_name,
        "manifest_family": entry.get("family"),
        "status": entry.get("status"),
        "ref_api": entry.get("ref_api"),
        "aiop_name": mapped[0] if mapped else None,
        "aiop_family": mapped[1] if mapped else None,
        "signature": entry.get("signature") or {},
        "workloads": [
            {"index": index, "label": row.get("label"), "dtype_cases": row.get("dtype_cases") or []}
            for index, row in enumerate(rows)
        ],
        "entry": entry,
    }
