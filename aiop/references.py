"""Dual reference baselines: verified Ascend A5 kernels and their CUDA peers.

Every generated operator has to be aligned with two implementations at the same
time:

* the CUDA implementation of the same operator (functional semantics, I/O
  contract, compute logic, precision expectations), and
* a verified TileLang Ascend A5 implementation (code structure, naming,
  comments, hardware adaptation, TileLang Ascend API usage).

The TileOPs reference checkout is discovered dynamically (environment,
config, ``ref_reops/TileOPs`` or a sibling checkout). Nothing here compiles or
imports the reference kernels; sources are indexed with regular expressions so
discovery stays cheap and works without torch/tilelang for the reference
device.
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, Config

_CLASS_RE = re.compile(r"^class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.MULTILINE)
_REGISTER_RE = re.compile(
    r"register_kernel_type\(\s*[\"']([^\"']+)[\"']\s*,\s*[\"']([^\"']+)[\"']\s*,\s*([A-Za-z_]\w*)\s*\)"
)


def _norm(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.lower())


def op_stem(op: str) -> str:
    """``ReluFwdOp`` → ``Relu``, ``GQADenseFwdOp`` → ``GQADense``, ``relu`` → ``relu``."""
    stem = op
    for suffix in ("FwdOp", "BwdOp", "Op", "Fwd", "Bwd", "Kernel"):
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)]
    return stem


@dataclass
class Reference:
    path: str
    rel_path: str
    class_name: str
    source: str
    op_keys: list[str] = field(default_factory=list)

    @property
    def module(self) -> str:
        return self.rel_path.replace("/", ".").removesuffix(".py")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rel_path": self.rel_path,
            "class": self.class_name,
            "module": self.module,
            "op_keys": list(self.op_keys),
            "lines": self.source.count("\n") + 1,
            "chars": len(self.source),
        }


@dataclass
class ReferencePair:
    op: str
    stem: str
    repo: str
    cuda: Reference | None = None
    ascend: Reference | None = None
    ascend_exact: bool = False
    target_rel_dir: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def has_repo(self) -> bool:
        return bool(self.repo)

    @property
    def missing(self) -> list[str]:
        missing: list[str] = []
        if self.cuda is None:
            missing.append("cuda")
        if self.ascend is None:
            missing.append("ascend")
        return missing

    def target_path(self, repo_root: str | Path | None = None) -> str:
        root = Path(repo_root or self.repo)
        rel_dir = self.target_rel_dir or "elementwise"
        return str(root / "src" / "tileops" / "kernels" / "ascend" / rel_dir / f"{self.stem.lower()}.py")

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "stem": self.stem,
            "repo": self.repo,
            "cuda": self.cuda.to_dict() if self.cuda else None,
            "ascend": self.ascend.to_dict() if self.ascend else None,
            "ascend_exact": self.ascend_exact,
            "target_rel_dir": self.target_rel_dir,
            "target_path": self.target_path() if self.repo else "",
            "missing": self.missing,
            "notes": list(self.notes),
        }


def _has_kernels(root: Path) -> bool:
    return (root / "src" / "tileops" / "kernels").is_dir()


def discover_tileops_repo(cfg: Config | None = None) -> str:
    """Locate the TileOPs checkout that holds both CUDA and Ascend kernels."""
    candidates: list[str] = []
    for key in ("AIOP_TILEOPS_REPO", "TILEOPS_REPO"):
        value = os.environ.get(key)
        if value:
            candidates.append(os.path.expandvars(os.path.expanduser(value)))
    if cfg is not None:
        candidates.append(cfg.references.tileops_repo)
    # The bundled reference checkout is the documented baseline; a sibling or
    # manifest-derived checkout is only a fallback.
    candidates.extend(
        [
            str(PROJECT_ROOT / "ref_reops" / "TileOPs"),
            str(Path.cwd() / "ref_reops" / "TileOPs"),
        ]
    )
    if cfg is not None and cfg.paths.tileops_manifest:
        path = Path(cfg.paths.tileops_manifest)
        for parent in [path, *path.parents]:
            if _has_kernels(parent):
                candidates.append(str(parent))
                break
    candidates.extend(
        [
            str(PROJECT_ROOT.parent / "TileOPs"),
            str(Path.cwd() / "TileOPs"),
        ]
    )
    for candidate in candidates:
        if candidate and _has_kernels(Path(candidate)):
            return str(Path(candidate).resolve())
    return ""


@functools.lru_cache(maxsize=64)
def manifest_has_op(manifest_dir: str, op: str) -> bool:
    """True when the TileOPs manifest declares this operator key."""
    root = Path(manifest_dir)
    if not root.is_dir():
        return False
    pattern = re.compile(rf"^{re.escape(op)}\s*:", re.MULTILINE)
    for path in root.rglob("*.y*ml"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            return True
    return False


def _iter_python(root: Path):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        yield path, text


@dataclass
class _Index:
    classes: list[tuple[Path, str, str, str, list[str]]] = field(default_factory=list)
    # (path, class_name, bases, source, registrations[op keys])


def _index(repo: Path) -> _Index:
    """Index class definitions and Ascend registrations once per checkout."""
    cached = _INDEX_CACHE.get(str(repo))
    if cached is not None:
        return cached
    index = _Index()
    kernels = repo / "src" / "tileops" / "kernels"
    for path, text in _iter_python(kernels):
        regs = [
            (match.group(1), match.group(2), match.group(3))
            for match in _REGISTER_RE.finditer(text)
        ]
        by_class: dict[str, list[str]] = {}
        for op_key, target, cls_name in regs:
            if target == "ascend":
                by_class.setdefault(cls_name, []).append(op_key)
        for match in _CLASS_RE.finditer(text):
            cls_name = match.group(1)
            bases = match.group(2)
            index.classes.append(
                (path, cls_name, bases, text, by_class.get(cls_name, []))
            )
    _INDEX_CACHE[str(repo)] = index
    return index


_INDEX_CACHE: dict[str, _Index] = {}


def _rel(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _is_kernel_class(cls_name: str, bases: str) -> bool:
    """Real kernel implementations end with ``Kernel``; interfaces do not."""
    return cls_name != "Kernel" and cls_name.endswith("Kernel") and (
        "Kernel" in bases or bases.strip() == ""
    )


def _pick_cuda(index: _Index, repo: Path, stem: str) -> Reference | None:
    wanted = _norm(stem)
    scored: list[tuple[tuple[int, int, int, int], Reference]] = []
    for path, cls_name, bases, text, _ in index.classes:
        rel_path = _rel(path, repo)
        if "/ascend/" in f"/{rel_path}" or rel_path.startswith("ascend/"):
            continue
        if path.name in ("call_spec.py", "kernel_base.py"):
            continue
        if not _is_kernel_class(cls_name, bases):
            continue
        name = _norm(cls_name)
        if not name.startswith(wanted):
            continue
        rank = 0 if name in (wanted + "kernel", wanted + "fwdkernel") else 1
        score = (rank, len(name), len(rel_path), 0 if path.stem.lower() == stem.lower() else 1)
        scored.append(
            (
                score,
                Reference(
                    path=str(path),
                    rel_path=rel_path,
                    class_name=cls_name,
                    source=text,
                ),
            )
        )
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


FAMILY_DIR_HINTS = {
    "attention": ("gemm", "reduction", "elementwise"),
    "gemm": ("gemm", "reduction", "elementwise"),
    "rmsnorm": ("reduction", "elementwise"),
    "layernorm": ("reduction", "elementwise"),
    "softmax": ("reduction", "elementwise"),
    "row_reduce": ("reduction", "elementwise"),
    "moe": ("moe", "elementwise"),
    "elementwise_unary": ("elementwise", "reduction"),
    "elementwise_binary": ("elementwise", "reduction"),
}


def _pick_ascend(
    index: _Index,
    repo: Path,
    op: str,
    stem: str,
    cuda_rel_dir: str,
    family: str = "",
) -> tuple[Reference | None, bool]:
    """Return (reference, exact_op_match)."""
    wanted_op = _norm(op)
    wanted_stem = _norm(stem)
    hints = FAMILY_DIR_HINTS.get(family, ())
    exact: list[Reference] = []
    same_dir: list[tuple[int, Reference]] = []
    prefix: list[tuple[tuple[int, int], Reference]] = []
    other: list[tuple[tuple[int, int], Reference]] = []
    for path, cls_name, bases, text, op_keys in index.classes:
        rel_path = _rel(path, repo)
        if not rel_path.startswith("ascend/") and "/kernels/ascend/" not in f"/{rel_path}":
            continue
        if not _is_kernel_class(cls_name, bases):
            continue
        name = _norm(cls_name)
        ref = Reference(
            path=str(path),
            rel_path=rel_path,
            class_name=cls_name,
            source=text,
            op_keys=op_keys,
        )
        if any(
            _norm(key) == wanted_op or _norm(op_stem(key)) == wanted_stem
            for key in op_keys
        ):
            exact.append(ref)
            continue
        inner = rel_path
        if inner.startswith("src/tileops/kernels/"):
            inner = inner[len("src/tileops/kernels/") :]
        parent = Path(inner).parent
        ref_dir = "" if str(parent) == "ascend" else str(parent.relative_to(Path("ascend")))
        if ref_dir == cuda_rel_dir:
            same_dir.append((0 if name.startswith(wanted_stem) else 1, ref))
        elif name.startswith("ascend" + wanted_stem):
            prefix.append(((0, len(name)), ref))
        else:
            hint_rank = 0
            for position, hint in enumerate(hints):
                if ref_dir.split("/")[0] == hint:
                    hint_rank = position
                    break
            else:
                hint_rank = len(hints)
            other.append(((1 + hint_rank, len(name)), ref))
    if exact:
        # Prefer the registration whose file name matches the operator stem.
        exact.sort(key=lambda ref: (0 if ref.path.rsplit("/", 1)[-1].startswith(stem.lower()) else 1))
        return exact[0], True
    if same_dir:
        same_dir.sort(key=lambda item: item[0])
        return same_dir[0][1], False
    if prefix:
        prefix.sort(key=lambda item: item[0])
        return prefix[0][1], False
    if other:
        other.sort(key=lambda item: item[0])
        return other[0][1], False
    return None, False


def find_reference_pair(
    spec,
    cfg: Config | None = None,
    *,
    repo_root: str | Path | None = None,
) -> ReferencePair:
    """Resolve the CUDA peer and the Ascend A5 baseline for ``spec``."""
    cfg = cfg or Config()
    repo = str(repo_root or discover_tileops_repo(cfg) or "")
    stem = op_stem(spec.name)
    pair = ReferencePair(op=spec.name, stem=stem, repo=repo)
    if not repo:
        pair.notes.append(
            "TileOPs reference checkout not found; set AIOP_TILEOPS_REPO to enable "
            "dual-baseline alignment"
        )
        return pair
    root = Path(repo)
    index = _index(root)
    pair.cuda = _pick_cuda(index, root, stem)
    cuda_rel_dir = ""
    if pair.cuda is not None:
        prefix = "src/tileops/kernels/"
        rel = pair.cuda.rel_path
        inner = rel[len(prefix) :] if rel.startswith(prefix) else rel
        cuda_parent = Path(inner).parent
        cuda_rel_dir = "" if str(cuda_parent) == "." else str(cuda_parent)
    pair.target_rel_dir = cuda_rel_dir or "elementwise"
    pair.ascend, pair.ascend_exact = _pick_ascend(
        index, root, spec.name, stem, cuda_rel_dir, spec.family
    )
    if pair.cuda is None:
        pair.notes.append(f"no CUDA kernel class matching {stem!r} was found")
    if pair.ascend is None:
        pair.notes.append("no verified Ascend A5 kernel was found as a style baseline")
    elif not pair.ascend_exact:
        pair.notes.append(
            f"no exact Ascend kernel for {spec.name!r}; using "
            f"{pair.ascend.class_name} ({pair.ascend.rel_path}) as the style baseline"
        )
    return pair


def _clip(source: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(source) <= limit:
        return source, False
    return source[:limit], True


def render_reference_context(pair: ReferencePair, max_chars: int = 40000) -> str:
    """Markdown block injected into design/LLM/draft prompts."""
    if not pair.repo or (pair.cuda is None and pair.ascend is None):
        return ""
    lines = [
        "# DUAL REFERENCE BASELINES (mandatory)",
        "",
        "Every generated Ascend A5 operator must align with both implementations below.",
        "Do not invent APIs, names or semantics that contradict them.",
        "",
    ]
    if pair.cuda is not None:
        source, clipped = _clip(pair.cuda.source, max_chars)
        lines += [
            "## A. CUDA peer (functional semantics / I/O / compute / precision)",
            f"- file: `{pair.cuda.rel_path}`",
            f"- class: `{pair.cuda.class_name}`",
            "",
            "```python",
            source.rstrip(),
            "```",
        ]
        if clipped:
            lines.append(f"(source truncated at {max_chars} chars; read `{pair.cuda.path}` for the rest)")
        lines.append("")
    if pair.ascend is not None:
        source, clipped = _clip(pair.ascend.source, max_chars)
        kind = "exact operator" if pair.ascend_exact else "style baseline (same family)"
        lines += [
            "## B. Verified Ascend A5 peer (structure / naming / comments / hardware adaptation)",
            f"- file: `{pair.ascend.rel_path}`",
            f"- class: `{pair.ascend.class_name}`",
            f"- role: {kind}",
            "",
            "```python",
            source.rstrip(),
            "```",
        ]
        if clipped:
            lines.append(f"(source truncated at {max_chars} chars; read `{pair.ascend.path}` for the rest)")
        lines.append("")
    lines += [
        "## C. Alignment rules enforced by the agent",
        "1. CUDA peer defines semantics, input/output names, shapes, dtypes and compute order.",
        "2. Ascend peer defines module/class naming, registration, comments and hardware adaptation.",
        "3. Use only the TileLang Ascend A5 API; obey UB 208 KiB / L1 512 KiB / L0C 256 KiB,",
        "   one-dimensional T.Kernel grid, and T.SimtVF/T.SimdVF thread domains.",
        "4. Output parity against the CUDA semantics is gated at atol <= 1e-5 (fp32)",
        "   and at the dtype resolution floor for fp16/bf16.",
        "5. The module must integrate into TileOPs: class `Ascend<Stem>Kernel` plus",
        f'   `register_kernel_type("{pair.op}", "ascend", Ascend<Stem>Kernel)`.',
        "",
    ]
    return "\n".join(lines)


def reference_summary(pair: ReferencePair) -> str:
    """Short section appended to DESIGN.md (paths + checklist, no source dumps)."""
    if not pair.repo or (pair.cuda is None and pair.ascend is None):
        return ""
    lines = [
        "## 基准实现对齐（双基准）",
        "",
        f"- 参考仓库：`{pair.repo}`",
        f"- CUDA 基准：`{pair.cuda.rel_path}` → `{pair.cuda.class_name}`" if pair.cuda else "- CUDA 基准：未找到",
        f"- Ascend 基准：`{pair.ascend.rel_path}` → `{pair.ascend.class_name}`"
        + ("" if pair.ascend_exact else "（同族风格基准）")
        if pair.ascend
        else "- Ascend 基准：未找到",
        f"- 目标集成路径：`{pair.target_path()}`",
        "",
        "对齐检查：语义/IO 以 CUDA 基准为准；结构/命名/注册/注释以 Ascend 基准为准；",
        "输出与 CUDA 语义基准的 fp32 绝对误差 ≤ 1e-5，fp16/bf16 不低于 dtype 分辨率下限。",
        "",
    ]
    return "\n".join(lines)


def write_references(task_dir: str | Path, pair: ReferencePair) -> Path:
    import json

    path = Path(task_dir) / "references.json"
    path.write_text(json.dumps(pair.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_reference_context(task_dir: str | Path, text: str) -> Path:
    path = Path(task_dir) / "reference_context.md"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path
