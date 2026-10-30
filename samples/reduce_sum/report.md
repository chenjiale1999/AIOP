# reduce_sum — aiop 验证报告

- status: **success_static_only**
- verified: `False`
- generator: `template`
- task: `sample-reduce_sum`

## 计划

```json
{
  "family": "row_reduce",
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
    "acc": 16
  }
}
```

## 编译（L1/L2）

- ok: `True`
- category: `none`
- CCE source: `/home/cjl/workspace/aiop/workspace/sample-reduce_sum/compile_result.cce.c`
- audit: `{"chars": 1395, "notify": 0, "wait": 0, "vf_calls": 1, "dma_copies": 0, "scalar_global_refs": 1, "has_printf": false, "balanced_sync": true}`

## 仿真（L3）

- status: `not_run`
- backend: `n/a`
- metrics: `{}`

## 上板（L4）

- status: `not_run`
- host: ``
- metrics: `{}`

## 事实与缺口

- L1/L2 verified: lowering + bisheng device compile passed (2.1s).

## 产物

- design: `/home/cjl/workspace/aiop/workspace/sample-reduce_sum/DESIGN.md`
- spec: `/home/cjl/workspace/aiop/workspace/sample-reduce_sum/spec.yaml`
- kernel: `/home/cjl/workspace/aiop/workspace/sample-reduce_sum/reduce_sum.py`
- test: `/home/cjl/workspace/aiop/workspace/sample-reduce_sum/test_reduce_sum.py`
- compile_report: `/home/cjl/workspace/aiop/workspace/sample-reduce_sum/compile_result.json`
