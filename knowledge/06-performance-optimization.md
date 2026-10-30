# A5 算子性能优化最佳实践

面向 Ascend 950（`dav-3510`）+ TileLang Ascend dialect。本文只写**能落到生成器决策**的规则：
先判瓶颈类型，再选 tiling / 流水 / 核类型，最后用同一口径测量。

## 1. 先判瓶颈：Roofline 分类

| 类型 | 判据 | 典型算子 | 优化主线 |
|---|---|---|---|
| 带宽受限 | 算术强度 `FLOPs / Bytes` 远低于机器平衡点 | elementwise、cast、reduce、softmax、norm | 减少搬运次数、满带宽、合并读写 |
| 算力受限 | 大 `M/N/K` 的矩阵乘 | GEMM、attention 主体 | Cube 利用率、L0 复用、K 方向流水 |
| 延迟受限 | 单次工作量小、核数用不满 | 小 shape、逐行小 reduction | 持久化、合并 tile、减少启动与 VF 调用 |

生成器必须先把 spec 归入三类之一，并在 `DESIGN.md` 写明依据（FLOPs、Bytes、shape）。

- Bytes 口径：每个输入读一次、每个输出写一次；attention 等中间结果留在片上，不计 HBM。
- FLOPs 口径：GEMM 为 `2*M*N*K`；elementwise 按实际算术计。
- **不要用理论峰值反推单算子效率**：模型 cycle、求解 II、编译耗时、设备时延是四种量。

## 2. 片上容量预算（硬数字）

| 存储 | 容量 | 来源 | 生成器用法 |
|---|---|---|---|
| UB（`T.alloc_shared`） | 物理 248 KB；含 SIMT VF 时调度上限 216 KB（32 KB 线程上下文） | `src/ascend/transform/auto_schedule.cc` `GetSharedMemoryLimit` | 预算按 **208 KB** 留余量（官方 rmsnorm 示例同口径） |
| L1（`T.alloc_l1`） | 512 KB | 同上 `GetL1MemoryLimit` | GEMM A/B tile + 版本数一起算 |
| L0C（`T.alloc_l0c`） | 256 KB | `GetL0CMemoryLimit` | `block_M*block_N*4B*版本` 必须放得下 |
| L0A / L0B | 各 64 KB | `GetL0AMemoryLimit` / `GetL0BMemoryLimit` | L0 staging 时按 `block_K` 切 |
| SS（`T.alloc_ss`） | 3 KB 共享池 | `aidoc` / 01 §2 | 只放标量，不放 tile |

预算公式（生成器必须执行）：

```text
UB_bytes  = Σ(alloc_shared 元素数 × dtype 大小 × buffer 版本数) + 布局 padding
L1_bytes  = Σ(alloc_l1 元素数 × dtype 大小 × 版本数)
L0C_bytes = block_M × block_N × 4 × 版本数
```

超预算时的调整顺序：降 `rows`/`tile` → 降 `num_stages` → 降 `block_N` → 复用输入 buffer（见
`example_rmsnorm.py` 的 `reuse_input`）。**不要**用 `T.annotate_unlimit_memory` 当修复。

## 3. 核类型与并行度

| 计算 | 核 | TileLang 入口 | 网格规模参考 |
|---|---|---|---|
| 逐元素 / 归约 / softmax / norm | AIV（Vector） | `T.SimtVF(threads=...)`、`T.SimdVF()` | 官方示例用 **64** 个 block |
| 矩阵乘 | AIC（Cube） | `T.gemm` | 官方 GEMM 示例用 **32** 个 block |
| Cube + epilogue 融合 | AIC + 2×AIV | `T.gemm` + `T.dual_copy` + `T.SimtVF` | `T.Kernel(32)`，`dual_copy` 分给两个 AIV |

来源：`examples/ascend/example_vecadd.py`（64）、`example_rmsnorm.py`（64）、
`example_gemm.py`（32）、`tilelang/ascend/README.md` Quick Start（32）。

规则：

1. 纯 Vector 算子不要用 `T.dual_copy` 伪装双 AIV；它要求同一 kernel 有 Cube 工作。
2. 混合核在 3510 上以 `__mix__(1,2)` 启动，两个 AIV 写同一 buffer 时字节区间必须不相交。
3. 小 shape 用 `T.Persistent` 合并 tile，避免启动开销与空核。
4. `T.SimtVF` 线程数不超过 tile 元素数；行归约时线程域尽量落在一行内（官方 rmsnorm 的
   `threads = min(threads, TILE)`）。

## 4. 搬运与对齐

| 规则 | 说明 |
|---|---|
| 32B 对齐 | DMA 基本粒度；行宽非 32B 时按字节处理，必要时 `pad_value=`（仅 GM→UB） |
| 填充 lane | 不会被自动初始化；消费前必须显式写值 |
| 尾部 | 逻辑范围不等于向上圆整后的物理范围；输出只写逻辑范围 |
| L2 提示 | GEMM 权重 GM→L1 用 `l2_cache_ctrl="NOTALLOC_KEEP"`；输出 UB→GM 用 `NOTALLOC_PW` |
| 多缓冲 | `T.annotate_buffer_versions({buf: 2})`；版本数计入容量 |
| 融合 | epilogue 能放进同一 kernel 就不要二次 launch；GEMM 后用 `dual_copy` 到 UB 再做 Vector |

**重要**：在 `T.Parallel` body 内直接读写 GM 会命中 polyhedral 向量化拒绝点
（`05-limits-and-pitfalls.md` §2），退化为逐元素标量访问。仅当 shape 无法整 tile 化、
且正确性优先时使用；常规路径必须先 `T.copy` 到 UB。

## 5. 流水与调度

- `T.Pipelined(extent, num_stages=2|3)`：GEMM 的 K 循环、逐元素的多 step 循环首选。
- `num_stages` 增加会成倍占用 UB/L1；先保证容量，再谈重叠。
- 持久化：`T.Persistent([tiles], num_cores, bx)`；需要 swizzle 时用
  `T.AscendTileScheduler` 系列（GROUP_M + N-first snake）。
- 手动 `T.Stage` 只在自动调度确有 overlap 问题时启用；不要与 `T.Pipelined` 的
  order/stage 参数混用。
- 编译期自动插同步；不要手写 `__syncthreads` 等价物。

## 6. SIMT / SIMD 选择

| 场景 | 选择 | 理由 |
|---|---|---|
| 逐元素、简单归约、可读性优先 | `T.SimtVF` + `T.Parallel` | 编译器负责线程映射与同步 |
| 需要特定指令 / mask / 舍入 / 打包布局 | `T.SimdVF` + `T.simd.*` | 精确控制 2048-bit 向量与谓词 |
| 行内归约 | `T.alloc_reducer`（sum/max/min） | 硬件延迟归约，避免跨线程手写规约 |
| 需要 SIMD 打包写 NZ | `T.simd.vsstb` / `vsts` | 注意半寄存器与 PART_EVEN 的配对语义 |

向量宽度 2048 bit = 64×fp32 = 128×fp16/bf16；`T.simd.pset(32)` 表示 32-bit 全 lane 有效。

## 7. 可搜索的调优参数（Autotune 边界）

| 参数 | 搜索空间建议 | 约束 |
|---|---|---|
| `block_M/block_N/block_K` | 32/64/128/224/256 的可行组合 | 必须整除对应维度；L0C/L1 预算 |
| `core_num` | 16/32/64 | Vector 64、Cube 32 为上限参考 |
| `T.SimtVF(threads=)` | 128/256/512 | ≤ tile 元素数，且不跨行破坏归约 |
| `num_stages` | 2/3/4 | UB/L1 容量 |
| `l2_cache_ctrl` | 默认 / KEEP / PW | 只对支持的搬运路径 |
| SIMD/SIMT 双实现 | 二选一 | 语义必须一致，golden 同时校验 |

搜索顺序：先固定正确性配置 → 扫 tile 几何 → 扫 stages → 最后扫缓存策略。
每轮只改一个维度，记录 `latency/bandwidth/TFLOPS` 与配置，避免不可复现的组合爆炸。

## 8. 测量协议

```text
1. 预热 ≥ 5 次（排除首次 JIT / 首次 DMA 配置）
2. 稳态重复 ≥ 20 次，报告中位数与 P90，不用单次最优
3. 关闭调试插桩（printf/dump/assert），否则测量值无效
4. 固定 kernel revision、输入 seed、设备频率策略、CANN/TileLang 版本
5. 带宽 = 逻辑 Bytes / latency；算力 = 逻辑 FLOPs / latency
6. 对比 baseline 时同时固定缓存清理协议
```

## 9. 生成器检查清单

- [ ] 已给出瓶颈分类与依据（FLOPs/Bytes）。
- [ ] UB/L1/L0C/L0A/L0B 预算已计算，含版本数与 padding。
- [ ] 核类型与网格规模符合 64 AIV / 32 AIC 约束。
- [ ] 非整除 shape 有明确尾部策略（整 tile 化 / guard / 明确拒绝）。
- [ ] `T.Parallel` body 没有 GM 直接访问、原子、内嵌循环、跨址邻居读。
- [ ] 归约 init/finalize 同作用域，reducer op 在白名单内。
- [ ] 多缓冲版本数已计入容量。
- [ ] 性能结论附带测量协议与配置。
