"""Op-scope enforcement and Ascend reference coverage tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiop import ascend_ops
from aiop.checks.static import check_api
from aiop.config import PROJECT_ROOT, load_config

LEGAL_KERNEL = """\
import tilelang.ascend.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((128,), "float32")):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_shared((128,), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=128):
                for i in T.Parallel(128):
                    x_ub[i] = T.max(x_ub[i], T.float32(0))
            with T.SimdVF():
                mask = T.simd.pset(32)
                r = T.simd.vld(x_ub[0])
                T.simd.vsts(x_ub[0], r, mask)
    return main
"""


def _kernel(tmp_path: Path, body: str, name: str = "kernel.py") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _rules(path: Path) -> list[str]:
    report = check_api(path, load_config())
    return [finding.rule for finding in report.findings]


def _messages(path: Path) -> list[str]:
    report = check_api(path, load_config())
    return [finding.message for finding in report.findings]


def test_legal_ascend_ops_pass(tmp_path):
    assert _rules(_kernel(tmp_path, LEGAL_KERNEL)) == []


def test_legal_simd_ops_pass(tmp_path):
    source = """\
import tilelang.ascend.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((256,), "float32")):
        with T.SimdVF():
            mask = T.simd.pset(32)
            a = T.simd.vld(X[0])
            b = T.simd.vld(X[128])
            c = T.simd.vadd(a, b, mask)
            idx = T.simd.vci(0, "uint32")
            g = T.simd.vgatherb(X[0], idx, mask)
            T.simd.vscatter(c, X[0], idx, mask)
            T.simd.vsts(X[0], g, mask)
    return main
"""
    assert _rules(_kernel(tmp_path, source)) == []


@pytest.mark.parametrize(
    "call",
    ["T.tma_copy(X, X)", "T.wgmma_gemm(X, X, X)", "T.alloc_tmem((8,), 'float32')"],
)
def test_cuda_only_ops_are_blocked(tmp_path, call):
    source = f"""\
import tilelang.ascend.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((8,), "float32")):
        with T.Kernel(1) as bx:
            {call}
    return main
"""
    rules = _rules(_kernel(tmp_path, source))
    assert "API001" in rules


@pytest.mark.parametrize(
    "call",
    [
        "T.printf('x=%d', bx)",
        "T.tile.vadd(X, X, X)",
        "T.alloc_L1((8,), 'float32')",
        "T.alloc_ub((8,), 'float32')",
        "T.sync_all()",
        "T.set_flag('V', 'MTE3', 0)",
    ],
)
def test_a3_legacy_ops_are_blocked(tmp_path, call):
    source = f"""\
import tilelang.ascend.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((8,), "float32")):
        with T.Kernel(1) as bx:
            {call}
    return main
"""
    assert "API003" in _rules(_kernel(tmp_path, source))


@pytest.mark.parametrize(
    "call",
    ["T.copy_op.copy(X, X)", "T.kernel.Kernel(1)", "T.simd.not_a_real_op(X, X)"],
)
def test_namespace_misuse_is_blocked(tmp_path, call):
    source = f"""\
import tilelang.ascend.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((8,), "float32")):
        with T.Kernel(1) as bx:
            {call}
    return main
"""
    rules = _rules(_kernel(tmp_path, source))
    assert "API004" in rules


def test_import_alias_is_enforced(tmp_path):
    source = """\
from tilelang.ascend import language as TL

def build():
    @TL.prim_func
    def main(X: TL.Tensor((8,), "float32")):
        with TL.Kernel(1) as bx:
            TL.printf("x")
    return main
"""
    assert "API003" in _rules(_kernel(tmp_path, source))


def test_aliased_legal_call_passes(tmp_path):
    source = """\
from tilelang.ascend import language as TL

def build():
    @TL.prim_func
    def main(X: TL.Tensor((8,), "float32")):
        with TL.Kernel(1) as bx:
            TL.copy(X, X)
    return main
"""
    assert _rules(_kernel(tmp_path, source)) == []


def test_non_ascend_import_is_not_scoped(tmp_path):
    source = """\
import tilelang.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((8,), "float32")):
        T.tma_copy(X, X)
    return main
"""
    assert _rules(_kernel(tmp_path, source)) == []


def test_error_messages_name_the_replacement(tmp_path):
    source = """\
import tilelang.ascend.language as T

def build():
    @T.prim_func
    def main(X: T.Tensor((8,), "float32")):
        with T.Kernel(1) as bx:
            T.printf("x")
            T.tile.vadd(X, X, X)
    return main
"""
    messages = " | ".join(_messages(_kernel(tmp_path, source)))
    assert "T.print" in messages
    assert "T.simd" in messages


def test_scope_matches_directory_scan():
    cfg = load_config()
    scope = ascend_ops.load_scope(cfg)
    on_disk = ascend_ops.scan_directory(scope.language_dir)
    assert set(scope.local) == set(on_disk)
    assert len(scope.local) == len(on_disk)
    assert len(scope.simd) == sum(1 for info in on_disk.values() if info.module == "simd.py")
    assert scope.source in {"runtime", "static"}
    if scope.source == "runtime":
        assert {"T.Kernel", "T.copy", "T.gemm", "T.alloc_l1", "T.dual_copy"} <= {
            f"T.{name}" for name in scope.declared
        }


def test_reference_document_covers_every_op():
    cfg = load_config()
    scope = ascend_ops.load_scope(cfg)
    doc = PROJECT_ROOT / "knowledge" / "13-ascend-op-reference.md"
    assert doc.is_file(), "run scripts/gen_ascend_op_kb.py"
    text = doc.read_text(encoding="utf-8")
    missing = [info.qualname for info in scope.local.values() if f"`{info.qualname}`" not in text]
    assert missing == []
    for info in scope.local.values():
        signature = f"{info.qualname}{info.signature[len(info.name):]}"
        assert f"`{signature}`" in text, f"missing signature for {info.qualname}"
    for section in ("核心功能", "入参", "返回值", "A5 硬件适配", "性能建议", "示例", "注意事项"):
        assert section in text


def test_generator_check_passes():
    import importlib.util
    import sys

    path = PROJECT_ROOT / "scripts" / "gen_ascend_op_kb.py"
    spec = importlib.util.spec_from_file_location("gen_ascend_op_kb", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scope = ascend_ops.load_scope(load_config())
    assert module.check(PROJECT_ROOT / "knowledge" / "13-ascend-op-reference.md", scope) == 0


def test_manifest_entries_are_complete():
    cfg = load_config()
    manifest = json.loads((PROJECT_ROOT / "reports" / "ascend_op_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ops"]
    for op in manifest["ops"]:
        for key in ("name", "qualname", "module", "kind", "signature"):
            assert op[key], f"{op.get('qualname')}: empty {key}"
        for key in ("summary", "returns", "hardware", "example"):
            assert op[key].strip(), f"{op['qualname']}: empty {key}"
        assert op["perf"], f"{op['qualname']}: empty perf"


def test_manifest_matches_directory_scan():
    cfg = load_config()
    scope = ascend_ops.load_scope(cfg)
    manifest = json.loads((PROJECT_ROOT / "reports" / "ascend_op_manifest.json").read_text(encoding="utf-8"))
    names = {op["name"] for op in manifest["ops"]}
    assert names == set(scope.local)
    assert manifest["local_count"] == len(scope.local)
    assert manifest["simd_count"] == len(scope.simd)
    assert manifest["declared_count"] == len(scope.declared)
