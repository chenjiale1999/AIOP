import ast

import pytest

from aiop.checks.static import check_api, check_semantics, check_syntax
from aiop.config import load_config
from aiop.generation.templates import render, write_kernel
from aiop.runtime.hardware import build_plan
from aiop.simulation.testgen import write_test
from aiop.specs import TaskSpec, TensorSpec


CASES = [
    TaskSpec("relu", "elementwise_unary", [TensorSpec("X", (128, 1000), "float32")], [TensorSpec("Y", (128, 1000), "float32")]),
    TaskSpec("add", "elementwise_binary", [TensorSpec("A", (1000003,), "float32"), TensorSpec("B", (1000003,), "float32")], [TensorSpec("C", (1000003,), "float32")]),
    TaskSpec("reduce_sum", "row_reduce", [TensorSpec("X", (128, 1000), "float32")], [TensorSpec("Y", (128,), "float32")]),
    TaskSpec("softmax", "softmax", [TensorSpec("X", (100, 777), "float16")], [TensorSpec("Y", (100, 777), "float16")]),
    TaskSpec("rmsnorm", "rmsnorm", [TensorSpec("X", (256, 1000), "bfloat16"), TensorSpec("W", (1000,), "bfloat16")], [TensorSpec("Y", (256, 1000), "bfloat16")]),
    TaskSpec("layernorm", "layernorm", [TensorSpec("X", (128, 1000), "float32"), TensorSpec("W", (1000,), "float32"), TensorSpec("B", (1000,), "float32")], [TensorSpec("Y", (128, 1000), "float32")]),
    TaskSpec("gemm", "gemm", [TensorSpec("A", (256, 256), "bfloat16"), TensorSpec("B", (448, 256), "bfloat16")], [TensorSpec("C", (256, 448), "float32")]),
    TaskSpec("gemm_relu", "gemm", [TensorSpec("A", (256, 128), "bfloat16"), TensorSpec("B", (224, 128), "bfloat16")], [TensorSpec("C", (256, 224), "float32")]),
]


@pytest.mark.parametrize("spec", CASES, ids=[c.name for c in CASES])
def test_render_is_valid_python(spec):
    plan = build_plan(spec)
    source = render(spec, plan)
    ast.parse(source)
    assert "def build()" in source
    assert "@T.prim_func" in source
    assert "T.Kernel(" in source
    assert "TODO" not in source


@pytest.mark.parametrize("spec", CASES, ids=[c.name for c in CASES])
def test_static_checks_pass(spec, tmp_path):
    plan = build_plan(spec)
    kernel = write_kernel(spec, plan, tmp_path)
    cfg = load_config()
    assert check_syntax(kernel).ok
    assert check_api(kernel, cfg).ok
    assert check_semantics(spec, plan, kernel).ok


def test_unknown_api_is_rejected(tmp_path):
    kernel = tmp_path / "bad.py"
    kernel.write_text(
        "import tilelang.ascend.language as T\n"
        "def build():\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((8,), 'float32')):\n"
        "        with T.Kernel(1) as bx:\n"
        "            T.not_a_real_api(X)\n"
        "    return main\n",
        encoding="utf-8",
    )
    report = check_api(kernel, load_config())
    assert not report.ok
    assert any(f.rule == "API001" for f in report.findings)


def test_threads_on_kernel_is_rejected(tmp_path):
    kernel = tmp_path / "bad2.py"
    kernel.write_text(
        "import tilelang.ascend.language as T\n"
        "def build():\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((8,), 'float32')):\n"
        "        with T.Kernel(1, threads=128) as bx:\n"
        "            T.copy(X, X)\n"
        "    return main\n",
        encoding="utf-8",
    )
    report = check_api(kernel, load_config())
    assert any(f.rule == "API002" for f in report.findings)


def test_test_script_is_self_contained(tmp_path):
    spec = CASES[3]
    plan = build_plan(spec)
    write_kernel(spec, plan, tmp_path)
    test = write_test(
        spec,
        plan,
        tmp_path,
        cann_home="/usr/local/Ascend/cann-9.2.0",
        device_flags=["-D__INTEL_LLVM_COMPILER=1"],
        warmup=1,
        repeats=1,
        tilelang_repo="/home/cjl/workspace/tilelang",
    )
    source = test.read_text(encoding="utf-8")
    ast.parse(source)
    for marker in ("[PRECISION_PASS]", "[PRECISION_FAIL]", "--json", "golden("):
        assert marker in source


@pytest.mark.parametrize("spec", CASES, ids=[c.name for c in CASES])
def test_generated_golden_executes_on_cpu(spec, tmp_path):
    import importlib.util

    plan = build_plan(spec)
    write_kernel(spec, plan, tmp_path)
    test = write_test(
        spec,
        plan,
        tmp_path,
        cann_home="/usr/local/Ascend/cann-9.2.0",
        device_flags=[],
        warmup=0,
        repeats=1,
        tilelang_repo="/home/cjl/workspace/tilelang",
    )
    module_spec = importlib.util.spec_from_file_location(f"runner_{spec.name}_{spec.family}", test)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    inputs = module.make_inputs("cpu")
    output = module.golden(*inputs)
    assert tuple(output.shape) == tuple(spec.output.shape)
    assert output.numel() == spec.output.numel
