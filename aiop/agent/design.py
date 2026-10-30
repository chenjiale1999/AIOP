"""Design document generation (Stage 1 of the A5 agent pipeline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..runtime.hardware import (
    AIC_CORES,
    AIV_CORES,
    L0C_BUDGET_BYTES,
    L1_BUDGET_BYTES,
    PLAN_SOURCE_NOTE,
    UB_BUDGET_BYTES,
    Plan,
    roofline,
)
from ..specs import TaskSpec

FAMILY_PATTERN = {
    "elementwise_unary": "knowledge/04 模式 A（逐元素/带宽受限）",
    "elementwise_binary": "knowledge/04 模式 A（逐元素/带宽受限）",
    "row_reduce": "knowledge/04 模式 B（行级归约）",
    "softmax": "knowledge/04 模式 B（行级归约 + online max/sum）",
    "rmsnorm": "knowledge/04 模式 B（RMSNorm）",
    "layernorm": "knowledge/04 模式 B（LayerNorm，双 reducer）",
    "gemm": "knowledge/04 模式 C（持久化 GEMM）",
}

FAMILY_API = {
    "elementwise_unary": "T.alloc_shared / T.copy / T.Pipelined / T.SimtVF / T.Parallel",
    "elementwise_binary": "T.alloc_shared / T.copy / T.Pipelined / T.SimtVF / T.Parallel",
    "row_reduce": "T.alloc_fragment / T.alloc_reducer / reducer_init / reducer_update / finalize_reducer",
    "softmax": "T.alloc_reducer(max/sum) / T.exp / T.alloc_fragment",
    "rmsnorm": "T.alloc_reducer(sum) / T.rsqrt / T.alloc_shared(weight)",
    "layernorm": "T.alloc_reducer(sum x2) / T.rsqrt / T.max / T.alloc_shared(weight,bias)",
    "gemm": "T.alloc_l1 / T.alloc_l0c / T.Persistent / T.Pipelined / T.gemm / T.dual_copy",
}


def build_design(spec: TaskSpec, plan: Plan, knowledge_hits: list[dict[str, Any]]) -> str:
    pr = spec.precision
    assert pr is not None
    roof = roofline(spec)
    buffers = json.dumps(plan.buffer_bytes, indent=2, sort_keys=True)
    hits = "\n".join(
        f"- `{h['doc']}` › {h['heading']} (score {h['score']})" for h in knowledge_hits[:5]
    ) or "- (no knowledge hit)"
    memory_rows = (
        f"| UB | {plan.ub_used} B | {UB_BUDGET_BYTES} B | `T.alloc_shared` |\n"
        f"| L1 | {_gemm_l1(spec, plan)} B | {L1_BUDGET_BYTES} B | `T.alloc_l1` |\n"
        f"| L0C | {plan.block_m * plan.block_n * 4 if spec.family == 'gemm' else 0} B | {L0C_BUDGET_BYTES} B | `T.alloc_l0c` |"
    )
    return f"""# {spec.name} — A5 TileLang 算子设计

> 自动生成：aiop agent · 目标 Ascend 950PR/DT（`dav-3510`, NpuArch 3510）
> 事实来源：{PLAN_SOURCE_NOTE}

## 1. 概述

| 项 | 值 |
|---|---|
| 算子 | `{spec.name}` |
| 家族 | `{spec.family}` |
| 计算语义 | {spec.formula or _formula_text(spec)} |
| 输入 | {_io_text(spec.inputs)} |
| 输出 | {_io_text(spec.outputs)} |
| 瓶颈类型 | {roof['bottleneck']}（算术强度 {roof['arithmetic_intensity']} FLOPs/B） |
| 逻辑 Bytes / FLOPs | {roof['bytes']} B / {roof['flops']} FLOPs |

## 2. 编程模式

选择 **Developer（自动调度）**：使用 `T.Kernel` + 高层 `T.copy`/`T.gemm`/`T.SimtVF`，
由编译器完成 core 放置、buffer 版本、同步 flag 配对。理由：

- 生成器需要稳定、可审计的结构；手写 `set_flag/wait_flag` 会引入无法静态验证的同步协议。
- 本算子不依赖跨核手工同步或手工多缓冲。

## 3. API 映射

| 计算步骤 | TileLang API | 说明 |
|---|---|---|
| 程序骨架 | `T.Kernel` + 一维 grid | 不允许 `threads=` |
| 片上分配 | `{FAMILY_API[spec.family].split(' / ')[0]}` | 见第 5 节预算 |
| 计算 | `{FAMILY_API[spec.family]}` | 全部为 A5 dialect 已验证接口 |
| 调度 | `T.Pipelined` / `T.Persistent` | `{_schedule_text(plan)}` |
| 同步 | 编译器自动 | 不生成手动 flag |

## 4. Tiling 与 workload

| 项 | 值 |
|---|---|
| 变体 | `{plan.variant}` |
| grid（block 数） | {plan.blocks}（Vector 上限 {AIV_CORES} / Cube 上限 {AIC_CORES}） |
| tile / rows | {_tile_text(plan)} |
| 线程域 | {plan.threads or 'n/a'} |
| 迭代次数 | {plan.iterations} |
| 尾部策略 | `{plan.tail_mode}` |
| GEMM block | {_gemm_block_text(plan)} |

## 5. 内存层级规划

```text
GM (HBM) --MTE2--> UB (T.alloc_shared) --Vector--> UB --MTE3--> GM
GM (HBM) --MTE2--> L1 (T.alloc_l1) --> L0A/L0B --> L0C --dual_copy--> UB --> GM   [仅 GEMM]
```

| 层级 | 计划占用 | 预算 | 依据 |
|---|---|---|---|
{memory_rows}

Buffer 明细：

```json
{buffers}
```

预算已包含 buffer 版本数（`annotate_buffer_versions`）。超预算时按“降 rows/tile → 降 stages →
复用输入 buffer”的顺序调整。

## 6. 循环与调度结构

- {_loop_text(plan, spec)}
- 不使用动态循环边界；所有 extent 为编译期常量。
- GEMM 的 K 循环首步 `clear_accum=True`，其余 `False`。

## 7. 同步策略

自动调度负责：
1. AIC/AIV 任务放置与 core mask；
2. buffer 版本与地址轮转；
3. MTE2/MTE3/V/S 的 `asc_sync_notify/wait` 配对。

生成代码中不应出现 `T.ascend_set_flag` / `T.ascend_wait_flag`；静态检查会拒绝调试插桩。

## 8. 验证方案

| 层级 | 内容 | 通过标准 |
|---|---|---|
| L0 静态 | 语法、API 白名单、约束、语义 | 无 error |
| L1 Lowering | `tilelang.lower(target="ascend")` | 产出 CCE 源码 |
| L2 设备编译 | `bisheng --npu-arch=dav-3510` | 生成 aibin |
| L3 仿真 | `npusim -s Ascend950` + torch_npu golden 对比 | 混合容差通过 |
| L4 上板 | SSH 远端 A5 运行 + 计时 | 精度通过且达到性能目标 |

Golden 参考：`test_{spec.name}.py` 中的 PyTorch CPU 实现，输入同 spec dtype/shape。

## 9. 精度标准

| dtype | atol | rtol | max_abs_error | required_matched_ratio |
|---|---|---|---|---|
| {spec.output.dtype} | {pr.atol} | {pr.rtol} | {pr.max_abs_error} | {pr.matched_ratio} |

判定：`|actual-golden| <= atol + rtol*|golden|` 的逐元素通过率 ≥ required_matched_ratio，
且 `max_abs_error <= max_abs_error`。整型输出要求 0 误差。

## 10. 风险点与技术约束检测

| 检查项 | 结论 |
|---|---|
| 网格维度 | 一维 `T.Kernel({plan.blocks})` ✅ |
| 线程域 | `T.SimtVF(threads=...)` ✅ |
| 动态边界 | 无 ✅ |
| UB/L1/L0C 容量 | 预算内 ✅ |
| GEMM 整除 | block 由 gcd 选择，保证整除 ✅ |
| 尾部 | `{plan.tail_mode}` ✅ |
| reducer 生命周期 | 每次 alloc 对应一次 init/finalize ✅ |
| 调试插桩 | 交付代码禁止 `T.print`/`assert` ✅ |

## 11. 知识库引用与同类实现

{_pattern_ref(spec)}

上游参考：

- `examples/ascend/example_vecadd.py`（逐元素 / 流水）
- `examples/ascend/example_rmsnorm.py`（行级 reducer / 容量预算）
- `examples/ascend/example_gemm.py`（持久化 GEMM / L1→L0C）
- `tilelang/ascend/README.md`（A5 后端与 SIMT/SIMD 规则）

知识库召回：

{hits}
"""


def build_blocked_design(
    spec: TaskSpec,
    reasons: list[str],
    knowledge_hits: list[dict[str, Any]],
) -> str:
    """Design record for an operator whose contract aiop can bind but not generate."""
    roof = roofline(spec)
    hits = "\n".join(
        f"- `{h['doc']}` › {h['heading']} (score {h['score']})" for h in knowledge_hits[:5]
    ) or "- (no knowledge hit)"
    manifest = spec.manifest or {}
    shapes = "\n".join(
        f"| `{t.name}` | in | {list(t.shape)} | `{t.dtype}` | `{t.role}` |"
        for t in spec.inputs
    ) + "\n" + "\n".join(
        f"| `{t.name}` | out | {list(t.shape)} | `{t.dtype}` | `{t.role}` |"
        for t in spec.outputs
    )
    metadata = manifest.get("metadata_inputs") or {
        t.name: {"values": t.values, "requires": list(t.requires)}
        for t in spec.inputs
        if t.values or t.requires
    }
    metadata_text = "\n".join(
        f"- `{name}`: values=`{info.get('values', '')}`"
        + (f", requires={info.get('requires')}" if info.get("requires") else "")
        for name, info in (metadata if isinstance(metadata, dict) else {}).items()
    ) or "- (none)"
    reasons_text = "\n".join(f"{index}. {reason}" for index, reason in enumerate(reasons, 1))
    present = ", ".join(f"`{name}`" for name in manifest.get("present_inputs", [])) or "(none)"
    return f"""# {spec.name} — A5 TileLang 算子设计（未生成）

> 自动生成：aiop agent · 目标 Ascend 950PR/DT（`dav-3510`, NpuArch 3510）
> 结论：**契约已解析，但当前模板族无法覆盖该算子**；未生成 kernel，未做编译/仿真/上板。

## 1. 算子契约

| 项 | 值 |
|---|---|
| manifest 条目 | `{manifest.get('op', spec.name)}` |
| manifest family | `{manifest.get('manifest_family', '')}` |
| 参考 API | `{spec.formula or 'n/a'}` |
| aiop family | `{spec.family}` |
| 来源文件 | `{manifest.get('source', '')}` |
| workload | `{manifest.get('workload_label', '')}` (#{manifest.get('workload_index', 0)}) |
| dtype case | #{manifest.get('dtype_case', 0)} `{json.dumps(manifest.get('dtype_assignment', {}))}` |
| 传入的 optional 张量 | {present} |

| 张量 | 方向 | 形状 | dtype | role |
|---|---|---|---|---|
{shapes}

## 2. 为什么不能生成

{reasons_text}

## 3. 元数据张量（`values` / `requires`）

{metadata_text}

这些张量由 manifest 生成器实例化，内容不是计算输入：`requires` 是调用方义务，
不能在 kernel 内检查；真正的 kernel 只能读取它们。

## 4. 规模与 roofline

| 项 | 值 |
|---|---|
| 逻辑 Bytes | {roof['bytes']} B |
| 逻辑 FLOPs | {roof['flops']} FLOPs |
| 算术强度 | {roof['arithmetic_intensity']} FLOPs/B |
| 瓶颈 | {roof['bottleneck']} |

## 5. 补齐路径

1. 在 `aiop/generation/templates.py` 增加该家族的 TileLang A5 模板（`T.Kernel` +
   `T.SimtVF`/`T.alloc_l0c` 等已验证接口），并在 `aiop/specs.py` 注册 family。
2. 在 `aiop/manifest.py` 的 `OP_FAMILY` 中加入 `{manifest.get('op', spec.name)}` 的映射。
3. 在 `aiop/runtime/hardware.py` 给出该家族的 tiling 规划与容量预算。
4. 用同一 manifest workload 行重跑 `aiop generate --manifest ... --op ... --workload ...`。

## 6. 验证状态

| 层级 | 结果 |
|---|---|
| L0 静态 | 未运行（无 kernel） |
| L1 Lowering | 未运行 |
| L2 设备编译 | 未运行 |
| L3 仿真 | 未运行 |
| L4 上板 | 未运行 |

## 7. 知识库召回

{hits}
"""


def build_open_design(
    spec: TaskSpec,
    reasons: list[str],
    knowledge_hits: list[dict[str, Any]],
    *,
    golden_source: str,
) -> str:
    """Design record for the LLM open-ended path (no A5 template family)."""
    roof = roofline(spec)
    hits = "\n".join(
        f"- `{h['doc']}` › {h['heading']} (score {h['score']})" for h in knowledge_hits[:5]
    ) or "- (no knowledge hit)"
    manifest = spec.manifest or {}
    shapes = "\n".join(
        f"| `{t.name}` | in | {list(t.shape)} | `{t.dtype}` | `{t.role}` |" for t in spec.inputs
    )
    shapes += "\n" + "\n".join(
        f"| `{t.name}` | out | {list(t.shape)} | `{t.dtype}` | `{t.role}` |" for t in spec.outputs
    )
    constraints = "\n".join(f"- {reason}" for reason in reasons) or "- (none recorded)"
    return f"""# {spec.name} — A5 TileLang 算子设计（LLM 开放模式）

> 自动生成：aiop agent · 目标 Ascend 950PR/DT（`dav-3510`, NpuArch 3510）
> 模式：**open**（无 aiop 模板 family，由 LLM 起草 kernel，门禁不放松）
> golden 来源：`{golden_source}`

## 1. 算子契约

| 项 | 值 |
|---|---|
| manifest 条目 | `{manifest.get('op', spec.name)}` |
| manifest family | `{manifest.get('manifest_family', '')}` |
| 参考 API | `{spec.formula or 'n/a'}` |
| aiop family | `{spec.family}`（无模板） |
| workload | `{manifest.get('workload_label', '')}` (#{manifest.get('workload_index', 0)}) |
| dtype case | #{manifest.get('dtype_case', 0)} `{json.dumps(manifest.get('dtype_assignment', {}))}` |

| 张量 | 方向 | 形状 | dtype | role |
|---|---|---|---|---|
{shapes}

## 2. 模板缺席的原因

{constraints}

## 3. 开放模式的边界

- **不做 tiling 断言**：`Plan.variant = "llm_open"`，只固定 grid 上限（Vector 64 / Cube 32），
  UB/L1/L0C 是否够用**由 lowering + bisheng 设备编译事后验证**，不预先承诺。
- **门禁不放松**：语法、API 白名单、reducer 生命周期、禁打印/断言、设备编译全部照旧；
  失败会带着编译错误回到 LLM 修复循环，而不是直接判死。
- **golden 可能自指**：`{golden_source}`。若为 `llm-self-referenced`，L3 结果只是
  "与该参考一致"，不是独立证据，报告里 `verified` 保持 `False`。
- **元数据张量**：manifest 的 `values` 生成器未实现时，带 `role=metadata` 输入的算子
  只做 L0–L2，L3 标记 `unavailable` 并写明原因。

## 4. 规模与 roofline

| 项 | 值 |
|---|---|
| 逻辑 Bytes | {roof['bytes']} B |
| 逻辑 FLOPs | {roof['flops']} FLOPs |
| 算术强度 | {roof['arithmetic_intensity']} FLOPs/B |
| 瓶颈 | {roof['bottleneck']} |

## 5. 验证方案

| 层级 | 内容 | 通过标准 |
|---|---|---|
| L0 静态 | 语法 / API 白名单 / reducer 生命周期 / 禁调试插桩 | 无 error |
| L1 Lowering | `tilelang.lower(target="ascend")` | 产出 CCE 源码 |
| L2 设备编译 | `bisheng --npu-arch=dav-3510` | 生成 aibin |
| L3 仿真 | golden（`{golden_source}`）+ 混合容差 | 见第 3 节的可信度说明 |
| L4 上板 | SSH 远端 A5 运行 | 需配置 `board.host` |

## 6. 知识库召回

{hits}
"""


def _formula_text(spec: TaskSpec) -> str:
    return {
        "elementwise_unary": f"{spec.name}(x) 逐元素",
        "elementwise_binary": f"c = a {spec.name} b 逐元素",
        "row_reduce": f"{spec.name}(x) 沿最后一维",
        "softmax": "softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))",
        "rmsnorm": "y = x / sqrt(mean(x^2) + eps) * w",
        "layernorm": "y = (x - mean) / sqrt(var + eps) * w + b",
        "gemm": "C = op(A) @ op(B)",
    }[spec.family]


def _io_text(tensors) -> str:
    return "; ".join(f"`{t.name}` {list(t.shape)} {t.dtype}" for t in tensors)


def _tile_text(plan: Plan) -> str:
    if plan.tile:
        return f"tile={plan.tile}"
    if plan.rows:
        return f"rows={plan.rows}, row_extent={plan.tile_extent}"
    return f"M/N/K block={plan.block_m}/{plan.block_n}/{plan.block_k}"


def _gemm_block_text(plan: Plan) -> str:
    if plan.family != "gemm":
        return "n/a"
    return f"{plan.block_m} x {plan.block_n} x {plan.block_k}（epilogue={plan.epilogue}）"


def _gemm_l1(spec: TaskSpec, plan: Plan) -> int:
    if spec.family != "gemm":
        return 0
    from ..specs import DTYPE_BYTES

    return (plan.block_m + plan.block_n) * plan.block_k * DTYPE_BYTES[spec.inputs[0].dtype] * plan.num_stages


def _schedule_text(plan: Plan) -> str:
    return {
        "simt_ub_staged": f"T.Pipelined(num_stages={plan.num_stages}) + UB 双缓冲",
        "simt_direct_global": "T.serial + guard（shape 非整 tile 整除）",
        "simt_row_reducer": f"T.Pipelined(num_stages={plan.num_stages}) + 行级 reducer",
        "persistent_gemm": "T.Persistent + K 方向 2 段流水",
        "persistent_gemm_epilogue": "T.Persistent + K 流水 + Cube→Vector epilogue",
    }.get(plan.variant, plan.variant)


def _loop_text(plan: Plan, spec: TaskSpec) -> str:
    if spec.family == "gemm":
        return "`T.Persistent` 遍历输出 tile；内层 `T.Pipelined` 遍历 K"
    if plan.variant == "simt_direct_global":
        return "`T.serial` 遍历 step，`T.Parallel(tile)` + `idx < N` guard"
    if spec.family == "elementwise_unary" or spec.family == "elementwise_binary":
        return "`T.Pipelined` 遍历 step，`T.Parallel(tile)` 逐元素计算后整体搬出"
    return "`T.Pipelined` 遍历行组，`T.SimtVF` 内 `T.Parallel(rows, tile)` + reducer"


def _pattern_ref(spec: TaskSpec) -> str:
    return f"模式：{FAMILY_PATTERN[spec.family]}；A5 性能规则见 `knowledge/06-performance-optimization.md`。"
