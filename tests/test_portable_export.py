"""Standalone export: one test file with the kernel embedded."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import torch

from aiop import cli
from aiop.config import discover_manifest_dir, discover_tilelang_repo
from aiop.generation.templates import write_kernel
from aiop.manifest import bind_operator, load_operator
from aiop.runtime.hardware import build_plan
from aiop.simulation.casematrix import input_key, plan_variants
from aiop.simulation.testgen import write_test
from aiop.specs import TaskSpec, TensorSpec


def _spec() -> TaskSpec:
    return TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )


def test_export_writes_single_file_with_embedded_kernel(tmp_path):
    spec = _spec()
    plan = build_plan(spec)
    task_dir = tmp_path / "workspace" / "relu-test"
    task_dir.mkdir(parents=True)
    spec.save(task_dir / "spec.yaml")
    write_kernel(spec, plan, task_dir)

    out = tmp_path / "bundle"
    code = cli.main(["export", str(task_dir), "--out", str(out)])
    assert code == 0

    test = out / "test_relu.py"
    assert test.is_file()
    source = test.read_text(encoding="utf-8")
    # The kernel is inlined at module top level, not stored as a literal.
    assert "def build_0():" in source
    assert "KERNEL_SOURCE" not in source
    assert "import base64" not in source
    assert "import types" not in source
    assert "linecache" not in source
    assert "AIOP_TILELANG_REPO" in source
    assert "ASCEND_HOME_PATH" in source
    assert "/home/" not in source

    # No external kernel file may be referenced or imported.
    assert "KERNEL_PATH" not in source
    assert "spec_from_file_location" not in source
    assert "importlib" not in source
    assert "from relu import" not in source
    assert "import relu\n" not in source
    # Case matrix and golden reference are embedded in the same file, and the
    # test-side symbols carry the _aiop suffix to avoid clashing with kernel code.
    assert "CASE_SPECS_AIOP =" in source
    assert "def golden_aiop_0(" in source
    assert "def compare_aiop(" in source
    assert "def main_aiop(" in source
    assert "torch.relu(x)" in source
    # Structured metrics are emitted for every case.
    assert "max_rel_error" in source
    assert "mean_rel_error" in source
    assert "bandwidth_gbps" in source
    assert "throughput_gflops" in source
    assert "compute_ms" in source

    # Copy the test alone to a fresh directory: it must still load the kernel
    # from the embedded source (no sibling relu.py).
    isolated = tmp_path / "server"
    isolated.mkdir()
    copied = isolated / test.name
    copied.write_text(source, encoding="utf-8")
    module_spec = importlib.util.spec_from_file_location("portable_relu_test", copied)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert callable(module.build_0)
    first = module.CASE_SPECS_AIOP[0]
    assert first["kind"] == "as_is"
    assert first["shape"] == [[128, 1000]]
    assert first["datatype"] == ["float32"]
    assert module.COVERAGE_AIOP["declared"] == 1
    accuracy = module.compare_aiop(torch.ones(4), torch.ones(4), first["precision"])
    for key in (
        "max_abs_error",
        "mean_abs_error",
        "max_rel_error",
        "mean_rel_error",
        "matched_ratio",
        "mismatched_elements",
        "total_elements",
    ):
        assert key in accuracy

    # TileLang's eager builder resolves the kernel body via inspect; inlined
    # source makes this work without any linecache tricks.
    assert inspect.getsource(module.build_0)


def test_string_precision_becomes_float_in_exported_test(tmp_path):
    data = _spec().to_dict()
    data["precision"] = {
        "atol": "1e-05",
        "rtol": "1e-05",
        "max_abs_error": "0.0001",
        "matched_ratio": "0.9999",
    }
    spec = TaskSpec.from_dict(data)
    plan = build_plan(spec)
    test = write_test(
        spec,
        plan,
        tmp_path,
        cann_home="/x",
        device_flags=[],
        warmup=0,
        repeats=1,
        tilelang_repo="/repo",
        kernel_source="def build():\n    return None\n",
    )
    source = test.read_text(encoding="utf-8")
    assert "'atol': 1e-05" in source
    assert "'rtol': 1e-05" in source
    assert "'max_abs_error': 0.0001" in source
    assert "'matched_ratio': 0.9999" in source

    module_spec = importlib.util.spec_from_file_location("runner_string_precision", test)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    tensor = torch.ones(8)
    assert module.compare_aiop(
        tensor, tensor, module.CASE_SPECS_AIOP[0]["precision"]
    )["passed"]


def test_manifest_export_enumerates_all_shape_dtype_variants(tmp_path):
    manifest = Path(discover_manifest_dir())
    if not manifest.is_dir():
        import pytest

        pytest.skip("TileOPs manifest not available")
    spec = bind_operator(manifest, "ReluFwdOp", workload=0, dtype_case=0)
    plan = build_plan(spec)
    write_kernel(spec, plan, tmp_path)
    kernel_source = (tmp_path / f"{spec.name}.py").read_text(encoding="utf-8")

    variants = plan_variants(
        spec, plan, kernel_source, enumerate_variants=True
    )
    entry = load_operator(manifest, "ReluFwdOp")
    expected = set()
    for workload_index, row in enumerate(entry.get("workloads") or []):
        for dtype_index in range(len(row.get("dtype_cases") or [{}])):
            expected.add(
                input_key(
                    bind_operator(
                        manifest,
                        "ReluFwdOp",
                        workload=workload_index,
                        dtype_case=dtype_index,
                    )
                )
            )
    assert expected
    assert {input_key(variant.spec) for variant in variants} == expected
    assert all(variant.available for variant in variants)

    test = write_test(
        spec,
        plan,
        tmp_path,
        cann_home="/x",
        device_flags=[],
        warmup=0,
        repeats=1,
        tilelang_repo=discover_tilelang_repo(),
        kernel_source=kernel_source,
        variants=variants,
    )
    source = test.read_text(encoding="utf-8")
    for index in range(len(variants)):
        assert f"def build_{index}():" in source
        assert f"def golden_aiop_{index}(" in source
    module_spec = importlib.util.spec_from_file_location("runner_matrix", test)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.COVERAGE_AIOP["declared"] == len(expected)
    assert {tuple(map(tuple, c["shape"])) for c in module.CASE_SPECS_AIOP} == {
        tuple(shape for shape, _ in key) for key in expected
    }


def test_unsupported_manifest_contracts_are_kept_in_case_matrix(tmp_path):
    manifest = Path(discover_manifest_dir())
    if not manifest.is_dir():
        import pytest

        pytest.skip("TileOPs manifest not available")
    spec = bind_operator(manifest, "AddFwdOp", workload=0, dtype_case=0)
    plan = build_plan(spec)
    write_kernel(spec, plan, tmp_path)
    kernel_source = (tmp_path / f"{spec.name}.py").read_text(encoding="utf-8")
    variants = plan_variants(spec, plan, kernel_source, enumerate_variants=True)
    assert any(not variant.available for variant in variants)

    test = write_test(
        spec,
        plan,
        tmp_path,
        cann_home="/x",
        device_flags=[],
        warmup=0,
        repeats=1,
        tilelang_repo=discover_tilelang_repo(),
        kernel_source=kernel_source,
        variants=variants,
    )
    module_spec = importlib.util.spec_from_file_location("runner_unsupported", test)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.UNSUPPORTED_CONTRACTS_AIOP
    assert all(not item.get("available", True) for item in module.UNSUPPORTED_CONTRACTS_AIOP)
    declared = module.COVERAGE_AIOP["declared"]
    executable = module.COVERAGE_AIOP["executable"]
    assert declared > executable
    combos = {
        (tuple(map(tuple, case["shape"])), tuple(case["datatype"]))
        for case in module.CASE_SPECS_AIOP
    }
    assert len(combos) == declared
