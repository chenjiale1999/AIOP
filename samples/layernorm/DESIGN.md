# layernorm — A5 TileLang 算子设计

> 自动生成：aiop agent · 目标 Ascend 950PR/DT（`dav-3510`, NpuArch 3510）
> 事实来源：src/ascend/transform/auto_schedule.cc (UB/L1/L0C/L0A/L0B) + examples/ascend/example_{vecadd,rmsnorm,gemm}.py (grid sizes)

## 1. 概述

| 项 | 值 |
|---|---|
| 算子 | `layernorm` |
| 家族 | `layernorm` |
| 计算语义 | y = (x - mean) / sqrt(var + eps) * w + b |
| 输入 | `X` [128, 1000] float32; `W` [1000] float32; `B` [1000] float32 |
| 输出 | `Y` [128, 1000] float32 |
| 瓶颈类型 | memory（算术强度 0.9922 FLOPs/B） |
| 逻辑 Bytes / FLOPs | 1032000 B / 1024000 FLOPs |

## 2. 编程模式

选择 **Developer（自动调度）**：使用 `T.Kernel` + 高层 `T.copy`/`T.gemm`/`T.SimtVF`，
由编译器完成 core 放置、buffer 版本、同步 flag 配对。理由：

- 生成器需要稳定、可审计的结构；手写 `set_flag/wait_flag` 会引入无法静态验证的同步协议。
- 本算子不依赖跨核手工同步或手工多缓冲。

## 3. API 映射

| 计算步骤 | TileLang API | 说明 |
|---|---|---|
| 程序骨架 | `T.Kernel` + 一维 grid | 不允许 `threads=` |
| 片上分配 | `T.alloc_reducer(sum x2)` | 见第 5 节预算 |
| 计算 | `T.alloc_reducer(sum x2) / T.rsqrt / T.max / T.alloc_shared(weight,bias)` | 全部为 A5 dialect 已验证接口 |
| 调度 | `T.Pipelined` / `T.Persistent` | `T.Pipelined(num_stages=2) + 行级 reducer` |
| 同步 | 编译器自动 | 不生成手动 flag |

## 4. Tiling 与 workload

| 项 | 值 |
|---|---|
| 变体 | `simt_row_reducer` |
| grid（block 数） | 64（Vector 上限 64 / Cube 上限 32） |
| tile / rows | rows=2, row_extent=1024 |
| 线程域 | 128 |
| 迭代次数 | 1 |
| 尾部策略 | `exact` |
| GEMM block | n/a |

## 5. 内存层级规划

```text
GM (HBM) --MTE2--> UB (T.alloc_shared) --Vector--> UB --MTE3--> GM
GM (HBM) --MTE2--> L1 (T.alloc_l1) --> L0A/L0B --> L0C --dual_copy--> UB --> GM   [仅 GEMM]
```

| 层级 | 计划占用 | 预算 | 依据 |
|---|---|---|---|
| UB | 20400 B | 212992 B | `T.alloc_shared` |
| L1 | 0 B | 524288 B | `T.alloc_l1` |
| L0C | 0 B | 262144 B | `T.alloc_l0c` |

Buffer 明细：

```json
{
  "acc": 16,
  "w": 4000,
  "x": 16384
}
```

预算已包含 buffer 版本数（`annotate_buffer_versions`）。超预算时按“降 rows/tile → 降 stages →
复用输入 buffer”的顺序调整。

## 6. 循环与调度结构

- `T.Pipelined` 遍历行组，`T.SimtVF` 内 `T.Parallel(rows, tile)` + reducer
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

Golden 参考：`test_layernorm.py` 中的 PyTorch CPU 实现，输入同 spec dtype/shape。

## 9. 精度标准

| dtype | atol | rtol | max_abs_error | required_matched_ratio |
|---|---|---|---|---|
| float32 | 1e-05 | 1e-05 | 0.0001 | 0.9999 |

判定：`|actual-golden| <= atol + rtol*|golden|` 的逐元素通过率 ≥ required_matched_ratio，
且 `max_abs_error <= max_abs_error`。整型输出要求 0 误差。

## 10. 风险点与技术约束检测

| 检查项 | 结论 |
|---|---|
| 网格维度 | 一维 `T.Kernel(64)` ✅ |
| 线程域 | `T.SimtVF(threads=...)` ✅ |
| 动态边界 | 无 ✅ |
| UB/L1/L0C 容量 | 预算内 ✅ |
| GEMM 整除 | block 由 gcd 选择，保证整除 ✅ |
| 尾部 | `exact` ✅ |
| reducer 生命周期 | 每次 alloc 对应一次 init/finalize ✅ |
| 调试插桩 | 交付代码禁止 `T.print`/`assert` ✅ |

## 11. 知识库引用与同类实现

模式：knowledge/04 模式 B（LayerNorm，双 reducer）；A5 性能规则见 `knowledge/06-performance-optimization.md`。

上游参考：

- `examples/ascend/example_vecadd.py`（逐元素 / 流水）
- `examples/ascend/example_rmsnorm.py`（行级 reducer / 容量预算）
- `examples/ascend/example_gemm.py`（持久化 GEMM / L1→L0C）
- `tilelang/ascend/README.md`（A5 后端与 SIMT/SIMD 规则）

知识库召回：

- `.opencode/agents/aiop-op-orchestrator.md` › aiop-op-orchestrator / 门禁校验（你亲自执行） (score 21.3792)
- `.agent/skills/aiop-op-pipeline/SKILL.md` › SKILL / 3. 设计阶段（必须产出 DESIGN.md） (score 20.4558)
- `.opencode/agents/aiop-op-designer.md` › aiop-op-designer / 必做步骤 (score 19.2275)
- `.opencode/agents/README.md` › README / 使用示例 (score 18.2769)
- `.opencode/agents/aiop-op-tuner.md` › aiop-op-tuner / 反模式 (score 17.8204)
