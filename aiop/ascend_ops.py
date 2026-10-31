"""Authoritative Ascend op scope derived from ``tilelang/ascend/language``.

The A5 agent must only emit ops that the Ascend dialect actually defines. This
module scans the dialect source directory, builds the legal op registry (the
declared ``tilelang.ascend.language`` surface plus the local ``T.simd`` set),
and validates generated code that imports the dialect.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ASCEND_IMPORT = "tilelang.ascend.language"
SIMD_NAMESPACE = "simd"

# Common LLM near-misses for the reducer lifecycle. These are 1:1 identifier
# renames; the canonical contract still has to be satisfied by the call shape.
KNOWN_OP_ALIASES = {
    "reduce_init": "reducer_init",
    "init_reducer": "reducer_init",
    "init_reduce": "reducer_init",
    "reduce_update": "reducer_update",
    "update_reducer": "reducer_update",
    "reduce_finalize": "finalize_reducer",
    "reducer_finalize": "finalize_reducer",
    "finalize_reduce": "finalize_reducer",
    "alloc_reduce": "alloc_reducer",
    "reduce_alloc": "alloc_reducer",
}


@dataclass(frozen=True)
class OpInfo:
    """One public definition found in the Ascend dialect directory."""

    name: str
    module: str
    kind: str  # "function" | "class"
    signature: str
    doc: str
    lineno: int
    namespace: str  # "T" or "T.simd"

    @property
    def qualname(self) -> str:
        return f"{self.namespace}.{self.name}"


@dataclass(frozen=True)
class ScopeViolation:
    """A dialect usage that falls outside the directory-defined op set."""

    op: str
    kind: str  # "unknown" | "simd_unknown" | "namespace"
    line: int
    column: int
    detail: str


@dataclass
class OpScope:
    """Legal Ascend op surface plus provenance."""

    language_dir: Path
    declared: frozenset[str]
    simd: frozenset[str]
    local: dict[str, OpInfo] = field(default_factory=dict)
    source: str = "runtime"  # "runtime" | "static"

    def is_declared(self, name: str) -> bool:
        return name in self.declared

    def is_simd(self, name: str) -> bool:
        return name in self.simd

    def is_local(self, name: str) -> bool:
        return name in self.local

    def suggest(self, name: str, pool: Iterable[str], n: int = 3) -> list[str]:
        import difflib

        candidates = list(pool)
        alias = KNOWN_OP_ALIASES.get(name)
        if alias and alias in candidates:
            return [alias]
        key = tuple(sorted(name.split("_")))
        permuted = [candidate for candidate in candidates if tuple(sorted(candidate.split("_"))) == key]
        if permuted:
            return permuted[:n]
        return difflib.get_close_matches(name, candidates, n=n, cutoff=0.6)

    def to_dict(self) -> dict[str, object]:
        return {
            "language_dir": str(self.language_dir),
            "source": self.source,
            "declared_count": len(self.declared),
            "local_count": len(self.local),
            "simd_count": len(self.simd),
        }


_SCOPE_CACHE: dict[tuple[str, tuple[float, ...]], OpScope] = {}


def _format_args(args: ast.arguments) -> str:
    return re.sub(r"^(self|cls)\s*,?\s*", "", ast.unparse(args))


def _signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    class_map: dict[str, ast.ClassDef] | None = None,
    depth: int = 0,
) -> str:
    if isinstance(node, ast.ClassDef):
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return f"{node.name}({_format_args(item.args)})"
        if class_map and depth < 3:
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in class_map:
                    parent = class_map[base.id]
                    inherited = _signature(parent, class_map, depth + 1)
                    return f"{node.name}{inherited[len(parent.name):]}"
        return f"{node.name}()"
    return f"{node.name}({_format_args(node.args)})"


def scan_directory(root: str | Path) -> dict[str, OpInfo]:
    """Return every public function/class defined under ``root``."""
    root = Path(root)
    class_map: dict[str, ast.ClassDef] = {}
    parsed: list[tuple[Path, ast.Module]] = []
    for path in sorted(root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parsed.append((path, tree))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                class_map[node.name] = node

    found: dict[str, OpInfo] = {}
    for path, tree in parsed:
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_"):
                continue
            namespace = "T.simd" if path.name == "simd.py" else "T"
            info = OpInfo(
                name=node.name,
                module=path.name,
                kind="class" if isinstance(node, ast.ClassDef) else "function",
                signature=_signature(node, class_map),
                doc=ast.get_docstring(node) or "",
                lineno=node.lineno,
                namespace=namespace,
            )
            if info.name in found:
                raise ValueError(f"duplicate Ascend op {info.name!r} in {found[info.name].module} and {path.name}")
            found[info.name] = info
    return found


def _declared_from_runtime(local: dict[str, OpInfo]) -> frozenset[str]:
    import tilelang.ascend.language as T

    names = set(getattr(T, "__all__", ()))
    # A local definition that is directly reachable on T also belongs to the
    # dialect surface (submodule names and metadata do not).
    names.update(name for name in local if hasattr(T, name))
    return frozenset(names)


def _module_all(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                return list(ast.literal_eval(node.value))
            except (ValueError, TypeError):
                return []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            try:
                return list(ast.literal_eval(node.value))
            except (ValueError, TypeError):
                return []
    return []


def _declared_from_static(root: Path, local: dict[str, OpInfo]) -> frozenset[str]:
    """Best-effort fallback used only when the runtime import is unavailable."""
    names = set(local)
    init = root / "__init__.py"
    if init.is_file():
        tree = ast.parse(init.read_text(encoding="utf-8"))
        names.update(_module_all(tree))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name)
    return frozenset(names)


def _latest_mtimes(root: Path) -> tuple[float, ...]:
    stamps = []
    for path in sorted(root.glob("*.py")):
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            stamps.append(0.0)
    return tuple(stamps)


def load_scope(cfg, root: str | Path | None = None, *, prefer_runtime: bool = True) -> OpScope:
    """Build (and cache) the legal op scope for the configured tilelang repo."""
    from .config import discover_tilelang_repo
    from .runtime import tilelang_a5

    if root is not None:
        language_dir = Path(root)
    else:
        repo = discover_tilelang_repo() or str(cfg.paths.tilelang_repo or "")
        if repo:
            language_dir = Path(repo) / "tilelang" / "ascend" / "language"
        else:
            try:
                import tilelang

                language_dir = Path(tilelang.__file__).resolve().parent / "ascend" / "language"
            except Exception:  # noqa: BLE001 - reported below
                language_dir = Path("tilelang") / "ascend" / "language"
    if not language_dir.is_dir():
        raise FileNotFoundError(f"Ascend dialect directory not found: {language_dir}")
    key = (str(language_dir.resolve()), _latest_mtimes(language_dir))
    cached = _SCOPE_CACHE.get(key)
    if cached is not None:
        return cached

    local = scan_directory(language_dir)
    simd = frozenset(name for name, info in local.items() if info.module == "simd.py")
    if prefer_runtime:
        try:
            tilelang_a5.bootstrap(cfg)
            declared = _declared_from_runtime(local)
            source = "runtime"
        except Exception:  # noqa: BLE001 - degrade to the static surface
            declared = _declared_from_static(language_dir, local)
            source = "static"
    else:
        declared = _declared_from_static(language_dir, local)
        source = "static"
    scope = OpScope(language_dir=language_dir, declared=declared, simd=simd, local=local, source=source)
    _SCOPE_CACHE[key] = scope
    return scope


def ascend_aliases(tree: ast.AST) -> set[str]:
    """Names bound to ``tilelang.ascend.language`` by an import statement."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == ASCEND_IMPORT and alias.asname:
                    aliases.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module == "tilelang.ascend":
            for alias in node.names:
                if alias.name == "language":
                    aliases.add(alias.asname or "language")
    return aliases


def _chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return list(reversed(parts))


def scan_tree(tree: ast.AST, scope: OpScope, aliases: set[str]) -> list[ScopeViolation]:
    """Validate every dialect attribute access in ``tree`` against ``scope``."""
    if not aliases:
        return []
    violations: list[ScopeViolation] = []
    seen: set[tuple[str, int, int]] = set()

    def add(op: str, kind: str, node: ast.Attribute, detail: str) -> None:
        key = (op, node.lineno, node.col_offset)
        if key in seen:
            return
        seen.add(key)
        violations.append(ScopeViolation(op=op, kind=kind, line=node.lineno, column=node.col_offset, detail=detail))

    chains: list[tuple[ast.Attribute, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _chain(node)
        if chain and chain[0] in aliases and len(chain) >= 2:
            chains.append((node, chain))
    # An inner ``T.a`` node shares its position with the outer ``T.a.b`` node;
    # report only the longest chain so one expression yields one finding.
    longer_positions = {(node.lineno, node.col_offset) for node, chain in chains if len(chain) > 2}

    for node, chain in chains:
        if len(chain) == 2 and (node.lineno, node.col_offset) in longer_positions:
            continue
        if len(chain) == 2:
            op = chain[1]
            if op == SIMD_NAMESPACE:
                continue
            if op in ("tile", "ascend"):
                add(f"{chain[0]}.{op}", "namespace", node, "A3 sub-namespace is not part of the Ascend dialect")
            elif not scope.is_declared(op):
                add(f"{chain[0]}.{op}", "unknown", node, "op is not part of the tilelang.ascend.language surface")
            continue
        sub = chain[1]
        if sub == SIMD_NAMESPACE:
            op = chain[2]
            if not scope.is_simd(op):
                add(f"{chain[0]}.simd.{op}", "simd_unknown", node, "op is not defined in tilelang/ascend/language/simd.py")
            continue
        full = ".".join(chain[:3])
        add(full, "namespace", node, "only T.<op> and T.simd.<op> access is allowed by the Ascend dialect")

    return violations


def normalize_known_aliases(source: str) -> tuple[str, list[dict[str, str]]]:
    """Rewrite known 1:1 reducer aliases in ``T.<name>`` positions."""

    fixes: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(2)
        target = KNOWN_OP_ALIASES.get(name)
        if target is None:
            return match.group(0)
        fixes.append({"from": f"T.{name}", "to": f"T.{target}"})
        return f"{match.group(1)}{target}"

    rewritten = re.sub(r"(\bT\.)([A-Za-z_][A-Za-z0-9_]*)", replace, source)
    return rewritten, fixes


def scan_source(source: str, scope: OpScope, filename: str = "<generated>") -> list[ScopeViolation]:
    """Parse ``source`` and validate dialect usage; syntax errors yield nothing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return scan_tree(tree, scope, ascend_aliases(tree))


def scan_path(path: str | Path, scope: OpScope) -> list[ScopeViolation]:
    text = Path(path).read_text(encoding="utf-8")
    return scan_source(text, scope, str(path))


def describe_scope(scope: OpScope) -> str:
    return (
        f"{len(scope.declared)} declared ops / {len(scope.local)} directory-defined ops / "
        f"{len(scope.simd)} T.simd ops ({scope.source})"
    )
