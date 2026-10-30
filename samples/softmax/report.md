# softmax — aiop 验证报告

- status: **success_static_only**
- verified: `False`
- generator: `template`
- task: `sample-softmax`

## 计划

```json
{
  "family": "softmax",
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
- CCE source: `/home/cjl/workspace/aiop/workspace/sample-softmax/compile_result.cce.c`
- audit: `{"chars": 2507, "notify": 0, "wait": 0, "vf_calls": 1, "dma_copies": 0, "scalar_global_refs": 2, "has_printf": false, "balanced_sync": true}`

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

- design: `/home/cjl/workspace/aiop/workspace/sample-softmax/DESIGN.md`
- spec: `/home/cjl/workspace/aiop/workspace/sample-softmax/spec.yaml`
- kernel: `/home/cjl/workspace/aiop/workspace/sample-softmax/softmax.py`
- test: `/home/cjl/workspace/aiop/workspace/sample-softmax/test_softmax.py`
- compile_report: `/home/cjl/workspace/aiop/workspace/sample-softmax/compile_result.json`
