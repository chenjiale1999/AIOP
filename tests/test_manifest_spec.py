"""TileOPs manifest ingestion: signature binding, unification and gating."""

import pytest

from aiop.manifest import (
    OP_FAMILY,
    BindError,
    ManifestError,
    bind_operator,
    describe_operator,
    list_operators,
    load_manifest,
)
from aiop.runtime.hardware import build_plan
from aiop.specs import TaskSpec, TensorSpec, UnsupportedOperator


def test_manifest_merges_every_family_file():
    ops = load_manifest()
    assert len(ops) > 150
    for name in ("MLAPagedFwdOp", "GQADenseFwdOp", "RMSNormFwdOp", "GemmFwdOp", "SoftmaxFwdOp"):
        assert name in ops, name
        assert ops[name]["_source"].endswith(".yaml")


def test_list_operators_reports_aiop_mapping():
    rows = {row["op"]: row for row in list_operators()}
    assert rows["RMSNormFwdOp"]["aiop_family"] == "rmsnorm"
    assert rows["GemmFwdOp"]["aiop_family"] == "gemm"
    assert rows["MLAPagedFwdOp"]["aiop_family"] is None
    assert rows["MLAPagedFwdOp"]["manifest_family"] == "attention"
    assert rows["MLAPagedFwdOp"]["workloads"] > 0


def test_bind_mla_paged_derives_batch_and_keeps_metadata():
    spec = bind_operator(None, "MLAPagedFwdOp", workload=0, dtype_case=0)
    tensors = {tensor.name: tensor for tensor in spec.inputs}
    # cache_lens carries 64 per-request lengths, so B = 64 even though the row
    # never writes B.
    assert tensors["q"].shape == (64, 1, 128, 576)
    assert tensors["kv_cache"].shape == (4096, 64, 576)
    assert tensors["kv_cache"].contiguous is True
    assert tensors["block_table"].shape == (64, 64)
    assert tensors["cache_seqlens"].shape == (64,)
    assert tensors["cache_seqlens"].role == "metadata"
    assert tensors["cache_seqlens"].values == "as_tensor(cache_lens)"
    assert tensors["cache_seqlens"].requires == ("in_range(S_q, W * PS + 1)",)
    assert [t.name for t in spec.outputs] == ["o", "lse"]
    assert spec.outputs[0].shape == (64, 1, 128, 512)
    assert spec.supported is False
    assert spec.family == "attention"
    assert spec.manifest["workload_label"] == "ds-v3-4k"


def test_optional_tensor_presence_follows_discriminant():
    without = bind_operator(None, "MLAPagedFwdOp", workload=0, dtype_case=0)
    assert "kv_scale" not in {t.name for t in without.inputs}
    with_scale = bind_operator(None, "MLAPagedFwdOp", workload=3, dtype_case=0)
    names = {t.name for t in with_scale.inputs}
    assert "kv_scale" in names
    assert with_scale.inputs[0].dtype == "bfloat16"  # T
    assert {t.name: t.dtype for t in with_scale.inputs}["kv_cache"] == "float8_e4m3fn"
    assert with_scale.manifest["dtype_assignment"] == {"T": "bfloat16", "KV": "float8_e4m3fn"}


def test_token_totals_are_unified_from_the_sequence_total():
    spec = bind_operator(None, "NSAVarlenFwdOp", workload=0, dtype_case=0)
    # seq_lens = repeat(2048, 4) -> T_q = 8192, N = 4.
    shapes = {tensor.name: tensor.shape for tensor in spec.inputs}
    assert shapes["q"] == (8192, 16, 64)
    assert shapes["offsets"] == (5,)
    assert shapes["block_counts"] == (8192, 1)


def test_bind_rmsnorm_reaches_a_template_family():
    spec = bind_operator(None, "RMSNormFwdOp", workload=0, dtype_case=1)
    assert spec.supported is True
    assert (spec.name, spec.family) == ("rmsnorm", "rmsnorm")
    assert [t.shape for t in spec.inputs] == [(2048, 4096), (4096,)]
    assert spec.inputs[0].raw_shape == "[*B, *normalized_shape]"
    assert spec.attrs["eps"] == 1.0e-6
    assert build_plan(spec).variant == "simt_row_reducer"
    assert spec.manifest["workload_label"] == "llama-8b-prefill"


def test_bind_layernorm_keeps_optional_bias():
    spec = bind_operator(None, "LayerNormFwdOp", workload=6, dtype_case=0)
    # llama-13b-prefill passes weight only.
    assert [t.name for t in spec.inputs] == ["x", "weight"]
    assert spec.supported is True
    spec = bind_operator(None, "LayerNormFwdOp", workload=0, dtype_case=0)
    assert [t.name for t in spec.inputs] == ["x", "weight", "bias"]


def test_bind_softmax_rejects_output_dtype_override():
    # row 3 forces dtype=float32 on a bfloat16 input: the row template cannot cast.
    spec = bind_operator(None, "SoftmaxFwdOp", workload=3, dtype_case=0)
    assert spec.supported is False
    assert any("dtype" in reason for reason in spec.unsupported_reasons)
    plain = bind_operator(None, "SoftmaxFwdOp", workload=0, dtype_case=0)
    assert plain.supported is True
    assert plain.output.shape == (32, 32, 4096)


def test_bind_reduction_only_on_the_last_axis():
    last = bind_operator(None, "SumFwdOp", workload=0, dtype_case=0)
    assert last.supported is True
    assert last.output.shape == (2048,)
    first = bind_operator(None, "SumFwdOp", workload=2, dtype_case=0)
    assert first.supported is False
    assert any("last axis" in reason for reason in first.unsupported_reasons)


def test_bind_gemm_uses_the_type_family_layout():
    supported = bind_operator(None, "GemmFwdOp", workload=1, dtype_case=0)
    a, b = supported.inputs
    assert (a.shape, b.shape) == ((128, 7168), (2112, 7168))
    assert supported.output.shape == (128, 2112)
    assert supported.output.dtype == "bfloat16"
    plan = build_plan(supported)
    assert plan.out_cast is True
    assert plan.variant == "persistent_gemm_epilogue"
    # trans_b=False stores B as [K, N]; the L1 path needs [N, K].
    transposed = bind_operator(None, "GemmFwdOp", workload=0, dtype_case=0)
    assert transposed.supported is False
    assert any("trans_b" in reason for reason in transposed.unsupported_reasons)


def test_elementwise_binary_requires_identical_shapes():
    spec = bind_operator(None, "AddFwdOp", workload=0, dtype_case=0)
    assert spec.supported is True
    assert spec.name == "add" and spec.family == "elementwise_binary"
    assert build_plan(spec).family == "elementwise_binary"


def test_clamp_broadcast_bounds_and_family_branch():
    spec = bind_operator(None, "ClampTensorFwdOp", workload=0, dtype_case=0)
    assert spec.output.shape == (4096, 4096)
    assert spec.supported is False  # three-input clamp has no aiop template


def test_unknown_op_and_bad_selector_are_reported():
    with pytest.raises(ManifestError):
        bind_operator(None, "NoSuchFwdOp")
    with pytest.raises(BindError):
        bind_operator(None, "RMSNormFwdOp", workload=999)
    with pytest.raises(BindError):
        bind_operator(None, "RMSNormFwdOp", workload="no-such-label")


def test_unsupported_spec_blocks_planning():
    spec = bind_operator(None, "GQADenseFwdOp", workload=0, dtype_case=0)
    with pytest.raises(UnsupportedOperator) as exc:
        build_plan(spec)
    assert "attention" in str(exc.value)
    assert exc.value.spec is spec


def test_manifest_fields_survive_a_spec_round_trip(tmp_path):
    spec = bind_operator(None, "GQADenseFwdOp", workload=0, dtype_case=0)
    path = tmp_path / "spec.json"
    spec.save(path)
    import json

    restored = TaskSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert restored.family == "attention"
    assert restored.supported is False
    assert restored.unsupported_reasons == spec.unsupported_reasons
    assert restored.manifest["op"] == "GQADenseFwdOp"
    assert [t.name for t in restored.inputs] == [t.name for t in spec.inputs]
    assert restored.outputs[0].shape == spec.outputs[0].shape


def test_every_attention_entry_binds():
    """The attention family is the contract the templates do not cover yet; every
    row must still bind (or fail with a message naming what is missing)."""
    entries = load_manifest()
    failures = []
    for name, entry in entries.items():
        if entry.get("family") != "attention":
            continue
        for index, row in enumerate(entry.get("workloads") or []):
            for case in range(len(row.get("dtype_cases") or [{}])):
                try:
                    spec = bind_operator(None, name, workload=index, dtype_case=case)
                except (BindError, ManifestError) as exc:
                    failures.append(f"{name}[{index}]: {exc}")
                    continue
                assert spec.inputs, name
                if not spec.supported:
                    assert spec.unsupported_reasons
    assert not failures, failures[:5]


def test_describe_operator_exposes_rows():
    data = describe_operator(None, "MLAPagedFwdOp")
    assert data["aiop_family"] is None
    assert data["workloads"][0]["label"] == "ds-v3-4k"
    assert "signature" in data and "forall" in data["signature"]


def test_op_family_table_matches_bindable_entries():
    entries = load_manifest()
    for op_name, (_aiop_name, family) in OP_FAMILY.items():
        assert op_name in entries, op_name
        spec = bind_operator(None, op_name, workload=0, dtype_case=0)
        assert spec.family == family, op_name


@pytest.mark.slow
def test_manifest_rmsnorm_compiles_end_to_end(tmp_path):
    from aiop.agent.orchestrator import OperatorAgent
    from aiop.config import load_config

    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.simulation.backend = "static"
    spec = bind_operator(None, "RMSNormFwdOp", workload=0, dtype_case=0)
    result = OperatorAgent(cfg).run(spec, simulate=False)
    assert result["status"] in ("success", "success_static_only"), result["status"]
    assert result["compile"]["ok"] is True
    assert result["plan"]["family"] == "rmsnorm"


@pytest.mark.slow
def test_manifest_gemm_output_cast_compiles(tmp_path):
    from aiop.agent.orchestrator import OperatorAgent
    from aiop.config import load_config

    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.simulation.backend = "static"
    spec = bind_operator(None, "GemmFwdOp", workload=1, dtype_case=0)
    result = OperatorAgent(cfg).run(spec, simulate=False)
    assert result["status"] in ("success", "success_static_only"), result["status"]
    assert result["plan"]["out_cast"] is True
    kernel = (tmp_path / "workspace" / result["task_id"] / "gemm.py").read_text()
    assert "T.dual_copy" in kernel


@pytest.mark.slow
def test_manifest_attention_reports_blocked_unsupported(tmp_path):
    from aiop.agent.orchestrator import OperatorAgent
    from aiop.config import load_config

    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    spec = bind_operator(None, "GQADenseFwdOp", workload=0, dtype_case=0)
    result = OperatorAgent(cfg).run(spec, simulate=False)
    assert result["status"] == "blocked_unsupported"
    assert result["verified"] is False
    assert result["unsupported_reasons"]
    task_dir = tmp_path / "workspace" / result["task_id"]
    assert (task_dir / "DESIGN.md").is_file()
    assert not (task_dir / "gqa_dense_fwd.py").exists()
    assert "cache_seqlens" not in (task_dir / "DESIGN.md").read_text()


def test_round_trip_keeps_optional_metadata_flags():
    metadata = TensorSpec(
        "cache_seqlens",
        (8,),
        "int32",
        role="metadata",
        values="as_tensor(cache_lens)",
        requires=("in_range(S_q, W * PS + 1)",),
        raw_shape="[B]",
    )
    data = metadata.to_dict()
    assert data["role"] == "metadata"
    assert data["requires"] == ["in_range(S_q, W * PS + 1)"]
    assert data["values"] == "as_tensor(cache_lens)"
    spec = TaskSpec(
        "rmsnorm",
        "rmsnorm",
        [TensorSpec("x", (8, 16), "float16"), metadata],
        [TensorSpec("y", (8, 16), "float16")],
        supported=False,
        unsupported_reasons=["metadata inputs are not fed to the norm template"],
        manifest={"op": "RMSNormFwdOp", "workload_label": "synthetic"},
    )
    restored = TaskSpec.from_dict(spec.to_dict())
    assert restored.supported is False
    assert restored.unsupported_reasons == spec.unsupported_reasons
    assert restored.inputs[1].role == "metadata"
    assert restored.inputs[1].requires == ("in_range(S_q, W * PS + 1)",)
    assert restored.inputs[1].raw_shape == "[B]"
    assert restored.manifest["workload_label"] == "synthetic"
    assert restored.family == "rmsnorm"
