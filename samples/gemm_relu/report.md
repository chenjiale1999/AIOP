# gemm_relu — aiop 验证报告

- status: **success_static_only**
- verified: `False`
- generator: `template`
- task: `sample-gemm_relu`

## 计划

```json
{
  "family": "gemm",
  "variant": "persistent_gemm_epilogue",
  "blocks": 1,
  "tile": 0,
  "rows": 0,
  "threads": 0,
  "tile_extent": 0,
  "num_stages": 2,
  "iterations": 1,
  "block_m": 256,
  "block_n": 224,
  "block_k": 128,
  "epilogue": "relu",
  "tail_mode": "exact",
  "buffer_bytes": {
    "c_ub": 114688
  }
}
```

## 编译（L1/L2）

- ok: `True`
- category: `none`
- CCE source: `/home/cjl/workspace/aiop/workspace/sample-gemm_relu/compile_result.cce.c`
- audit: `{"chars": 2592, "notify": 3, "wait": 3, "vf_calls": 1, "dma_copies": 1, "scalar_global_refs": 5, "has_printf": false, "balanced_sync": true}`

## 仿真（L3）

- status: `not_run`
- backend: `n/a`
- metrics: `{}`

## 上板（L4）

- status: `not_run`
- host: ``
- metrics: `{}`

## 事实与缺口

- L1/L2 verified: lowering + bisheng device compile passed (2.5s).

## 产物

- design: `/home/cjl/workspace/aiop/workspace/sample-gemm_relu/DESIGN.md`
- spec: `/home/cjl/workspace/aiop/workspace/sample-gemm_relu/spec.yaml`
- kernel: `/home/cjl/workspace/aiop/workspace/sample-gemm_relu/gemm_relu.py`
- test: `/home/cjl/workspace/aiop/workspace/sample-gemm_relu/test_gemm_relu.py`
- compile_report: `/home/cjl/workspace/aiop/workspace/sample-gemm_relu/compile_result.json`
