# gemm — aiop 验证报告

- status: **success_static_only**
- verified: `False`
- generator: `template`
- task: `sample-gemm_bf16`

## 计划

```json
{
  "family": "gemm",
  "variant": "persistent_gemm",
  "blocks": 2,
  "tile": 0,
  "rows": 0,
  "threads": 0,
  "tile_extent": 0,
  "num_stages": 2,
  "iterations": 1,
  "block_m": 256,
  "block_n": 224,
  "block_k": 128,
  "epilogue": "none",
  "tail_mode": "exact",
  "buffer_bytes": {}
}
```

## 编译（L1/L2）

- ok: `True`
- category: `none`
- CCE source: `/home/cjl/workspace/aiop/workspace/sample-gemm_bf16/compile_result.cce.c`
- audit: `{"chars": 2063, "notify": 5, "wait": 5, "vf_calls": 0, "dma_copies": 0, "scalar_global_refs": 3, "has_printf": false, "balanced_sync": true}`

## 仿真（L3）

- status: `not_run`
- backend: `n/a`
- metrics: `{}`

## 上板（L4）

- status: `not_run`
- host: ``
- metrics: `{}`

## 事实与缺口

- L1/L2 verified: lowering + bisheng device compile passed (1.9s).

## 产物

- design: `/home/cjl/workspace/aiop/workspace/sample-gemm_bf16/DESIGN.md`
- spec: `/home/cjl/workspace/aiop/workspace/sample-gemm_bf16/spec.yaml`
- kernel: `/home/cjl/workspace/aiop/workspace/sample-gemm_bf16/gemm.py`
- test: `/home/cjl/workspace/aiop/workspace/sample-gemm_bf16/test_gemm.py`
- compile_report: `/home/cjl/workspace/aiop/workspace/sample-gemm_bf16/compile_result.json`
