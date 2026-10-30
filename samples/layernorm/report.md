# layernorm — aiop 验证报告

- status: **success_static_only**
- verified: `False`
- generator: `template`
- task: `sample-layernorm`

## 计划

```json
{
  "family": "layernorm",
  "variant": "simt_row_reducer",
  "blocks": 64,
  "tile": 0,
  "rows": 2,
  "threads": 128,
  "tile_extent": 1024,
  "num_stages": 2,
  "iterations": 1,
  "block_m": 0,
  "block_n": 0,
  "block_k": 0,
  "epilogue": "none",
  "tail_mode": "exact",
  "buffer_bytes": {
    "x": 16384,
    "acc": 16,
    "w": 4000
  }
}
```

## 编译（L1/L2）

- ok: `True`
- category: `none`
- CCE source: `/home/cjl/workspace/aiop/workspace/sample-layernorm/compile_result.cce.c`
- audit: `{"chars": 3646, "notify": 1, "wait": 1, "vf_calls": 1, "dma_copies": 2, "scalar_global_refs": 4, "has_printf": false, "balanced_sync": true}`

## 仿真（L3）

- status: `not_run`
- backend: `n/a`
- metrics: `{}`

## 上板（L4）

- status: `not_run`
- host: ``
- metrics: `{}`

## 事实与缺口

- L1/L2 verified: lowering + bisheng device compile passed (2.0s).

## 产物

- design: `/home/cjl/workspace/aiop/workspace/sample-layernorm/DESIGN.md`
- spec: `/home/cjl/workspace/aiop/workspace/sample-layernorm/spec.yaml`
- kernel: `/home/cjl/workspace/aiop/workspace/sample-layernorm/layernorm.py`
- test: `/home/cjl/workspace/aiop/workspace/sample-layernorm/test_layernorm.py`
- compile_report: `/home/cjl/workspace/aiop/workspace/sample-layernorm/compile_result.json`
