import math

import pytest

from aiop.runtime.hardware import AIC_CORES, AIV_CORES, UB_BUDGET_BYTES, build_plan, check_plan, largest_divisor
from aiop.specs import TaskSpec, TensorSpec, default_precision, infer_family, spec_from_requirement


def test_dtype_normalization_and_precision():
    spec = TaskSpec(
        "add",
        "elementwise_binary",
        [TensorSpec("A", (16, 16), "fp32"), TensorSpec("B", (16, 16), "float")],
        [TensorSpec("C", (16, 16), "f32")],
    )
    assert spec.input.dtype == "float32"
    assert spec.output.dtype == "float32"
    assert spec.validate() == []
    assert spec.precision.matched_ratio == default_precision("float32").matched_ratio


def test_row_reduce_shape_contract():
    good = TaskSpec(
        "reduce_sum",
        "row_reduce",
        [TensorSpec("X", (8, 33), "float32")],
        [TensorSpec("Y", (8,), "float32")],
    )
    assert good.validate() == []
    bad = TaskSpec.from_dict(
        {
            "name": "reduce_sum",
            "family": "row_reduce",
            "inputs": [{"name": "X", "shape": [8, 33], "dtype": "float32"}],
            "outputs": [{"name": "Y", "shape": [8, 1], "dtype": "float32"}],
        }
    )
    assert any("row_reduce output" in msg for msg in bad.validate())


def test_gemm_contract():
    spec = TaskSpec(
        "gemm",
        "gemm",
        [TensorSpec("A", (64, 32), "bfloat16"), TensorSpec("B", (128, 32), "bfloat16")],
        [TensorSpec("C", (64, 128), "float32")],
    )
    assert spec.validate() == []
    assert spec.flops() == 2 * 64 * 128 * 32
    assert spec.arithmetic_intensity() > 0


def test_plan_elementwise_exact_tiling():
    spec = TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )
    plan = build_plan(spec)
    assert plan.variant == "simt_ub_staged"
    assert plan.blocks <= AIV_CORES
    assert 128000 % (plan.blocks * plan.tile) == 0
    assert check_plan(spec, plan)["errors"] == []


def test_plan_elementwise_tail_guard():
    spec = TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (1000003,), "float32")],
        [TensorSpec("Y", (1000003,), "float32")],
    )
    plan = build_plan(spec)
    assert plan.variant == "simt_direct_global"
    assert plan.tail_mode == "guarded"
    assert plan.blocks == AIV_CORES


def test_plan_row_budget_and_guard():
    spec = TaskSpec(
        "layernorm",
        "layernorm",
        [
            TensorSpec("X", (100, 777), "float32"),
            TensorSpec("W", (777,), "float32"),
            TensorSpec("B", (777,), "float32"),
        ],
        [TensorSpec("Y", (100, 777), "float32")],
    )
    plan = build_plan(spec)
    assert plan.rows == 1
    assert plan.tail_mode == "guarded"
    assert plan.ub_used <= UB_BUDGET_BYTES


def test_plan_gemm_divisibility_and_budget():
    spec = TaskSpec(
        "gemm",
        "gemm",
        [TensorSpec("A", (512, 512), "bfloat16"), TensorSpec("B", (1024, 512), "bfloat16")],
        [TensorSpec("C", (512, 1024), "float32")],
    )
    plan = build_plan(spec)
    assert 512 % plan.block_m == 0
    assert 1024 % plan.block_n == 0
    assert 512 % plan.block_k == 0
    assert plan.blocks <= AIC_CORES
    assert plan.block_m * plan.block_n * 4 <= 256 * 1024


def test_largest_divisor():
    assert largest_divisor(128000, 4096) == 4000
    assert largest_divisor(1000003, 4096) == 1


def test_infer_family_and_requirement():
    assert infer_family("softmax") == "softmax"
    assert infer_family("reduce_max") == "row_reduce"
    spec = spec_from_requirement("relu on [8, 16] float32")
    assert spec.name == "relu"
    assert spec.input.shape == (8, 16)
