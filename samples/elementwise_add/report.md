# add — aiop 验证报告

- status: **success_static_only**
- verified: `False`
- generator: `template`
- task: `sample-elementwise_add`

## 计划

```json
{
  "family": "elementwise_binary",
  "variant": "simt_ub_staged",
  "blocks": 64,
  "tile": 4096,
  "rows": 0,
  "threads": 256,
  "tile_extent": 0,
  "num_stages": 2,
  "iterations": 4,
  "block_m": 0,
  "block_n": 0,
  "block_k": 0,
  "epilogue": "none",
  "tail_mode": "exact",
  "buffer_bytes": {
    "tile0": 32768,
    "tile1": 32768,
    "tile2": 32768
  }
}
```

## 编译（L1/L2）

- ok: `True`
- category: `none`
- CCE source: `/home/cjl/workspace/aiop/workspace/sample-elementwise_add/compile_result.cce.c`
- audit: `{"chars": 2826, "notify": 8, "wait": 8, "vf_calls": 1, "dma_copies": 3, "scalar_global_refs": 3, "has_printf": false, "balanced_sync": true}`

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

- design: `/home/cjl/workspace/aiop/workspace/sample-elementwise_add/DESIGN.md`
- spec: `/home/cjl/workspace/aiop/workspace/sample-elementwise_add/spec.yaml`
- kernel: `/home/cjl/workspace/aiop/workspace/sample-elementwise_add/add.py`
- test: `/home/cjl/workspace/aiop/workspace/sample-elementwise_add/test_add.py`
- compile_report: `/home/cjl/workspace/aiop/workspace/sample-elementwise_add/compile_result.json`
