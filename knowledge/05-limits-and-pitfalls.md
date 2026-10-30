# 已知限制与生成前自检

本文件只收录「会导致生成失败或结果错误」的硬限制。每条都能指到源码或上游文档。

## 1. 语言与 API 层硬限制

| # | 限制 | 后果 | 规避 |
|---|---|---|---|
| 1 | `T.Kernel` 不接受 `threads=` | 立即 `TypeError` | 线程域写进 `T.SimtVF(threads=...)` |
| 2 | 只支持一维网格 `T.Kernel(N)` | 多维 `ValueError` | 自行算 tile 索引 `row = tile // n_tiles` |
| 3 | L1 输入 GEMM 要求 `transpose_A=False`、`transpose_B=True` | 走错 lowering / 结果错 | 权重按 `[N, K]` 存；不要套用到 L0 路径 |
| 4 | `T.dual_copy` 前提是同一 kernel 有 Cube 侧工作 | 纯 Vector 核无法靠它变双 AIV | 用 `T.MixedKernel(..., sids=2)` 且确有 Cube 工作 |
| 5 | L0C→UB 双拷贝要求源/目的 dtype 一致、rank≥2；N-split 需 N 被 32 整除 | 形状/类型错误 | 分割前按 `04` 的模式 C 检查尾维 |
| 6 | reducer 句柄不可读、不可 `T.clear`/`T.fill`、不可原地 finalize；init/finalize 必须同作用域 | 语义拒绝 | 读 `finalize_reducer` 的目的 fragment |
| 7 | reducer 仅支持 `sum`/`max`/`min` | 其他 op 报错 | 位运算归约用 `T.reduce_bitand/bitor/bitxor`（仅整型） |
| 8 | `T.warp_reduce_*` 硬件指令仅支持 fp16/fp32/int32/uint32 | dtype 不支持 | 其他 dtype 走 shuffle/UB 回退，或换算法 |
| 9 | `T.alloc_ss` 仅标量、shape 必须静态、3072B 池 | VF/DMA/原子不可用 | 大数据走 UB |
| 10 | `pad_value`/`data_select` 仅 GM→UB，且只能用 8/16/32-bit dtype | 其他路径被忽略 | 其他路径显式写填充值 |
| 11 | 填充 lane 不会自动初始化 | 脏数据进入计算 | 消费前显式写值 |
| 12 | `T.annotate_unlimit_memory` 不扩物理内存 | 超容量仍启动失败 | 当作排查工具，不是修复 |
| 13 | 手动多缓冲（`annotate_manual_multi_buffer`）有限支持 | 复杂控制流下同步错误 | 优先编译器托管版本 |

## 2. 向量化拒绝点（P2/P3 polyhedral 路径）

完整 40 条逐条契约见
`/home/cjl/workspace/tilelang/docs/compiler_internals/p2_unsupported_catalog.md`
（`plan.py` 21 / `value.py` 9 / `body.py` 5 / `emit.py` 4 / `access.py` 1）。

**适用范围**：这些拒绝点属于 `AscendPolyhedralVectorize` 这一条 lowering 路径；只有被它接管的
`T.Parallel` 结构才受约束。被拒时的正确动作是**改写 kernel 结构或改用显式 `T.simd.*`**，
而不是反复调参。

| 类别 | 代表拒绝点 | 典型触发写法 |
|---|---|---|
| 访存 | 访存秩不匹配 | `BufferLoad/Store` 下标个数与 buffer rank 不一致 |
| body | 原子调用 | `T.Parallel` body 内出现 `atomic_*` |
| body | 两臂写不同地址 | if/else 两臂写同一 buffer 的不同位置 |
| body | 同址 if/else 但单臂本身带谓词 | 嵌套 else-if 导致没有单一 select 条件 |
| body | body 内嵌循环 | `T.Parallel` 内再套 `for` |
| value | 除数不是编译期正奇数常量 | `FloorDiv/FloorMod` 用变量或偶数做除数 |
| value | GM 操作数 | `T.Parallel` 内直接读 GM，未先搬到 UB |
| value | 不支持的 intrinsic / 原子 | body 值表达式里出现不支持的 call |
| plan | 循环下界非 0 | `for i in T.Parallel(...)` 起点不为 0 |
| plan | 没有并行轴 / 主体无写 | 纯串行循环或只读 body |
| plan | 所有轴 true-random | 访存索引在所有轴上都不是仿射 |
| plan | stencil 跨址读 | 并行轴上读 `i-1`/`i+1` 这类邻居地址 |
| plan | 不支持的归约算子 | 归约操作用了白名单外的语义 |
| plan | 条件归约 / 旧值多次出现 | 归约体被 if 包裹，或累加值被读多次 |
| plan | 同段混位宽 / store 元素类型不支持 | 一个并行段里混用不同位宽 |
| emit | 不能标量化 / 无发射器 / 缺 per-lane 索引向量 / 循环外语句非均匀 | 计划通过但发射阶段无对应指令 |

## 3. 示例级功能限制

`flash_attention`（`examples/ascend/flash_attention/README.md`）：

- `head_dim` 固定 128（softmax 直接写 NZ 布局，硬编码 D=128）。
- `block_q` 必须为偶数；`block_kv == 2 * VL == 128`（64 个 fp32 SIMD lane）。
- `q_len % block_q == 0`、`kv_len % block_kv == 0`。
- 输入必须 bf16、fp32 累加；输出 dtype 可配（MHA 直接 fp32，GQA 写 bf16）。

## 4. 容量 / 启动失败排查顺序

1. 记录**生成的 allocation 大小**（含 layout padding 与 buffer 版本数）。
2. 确认 VF 模式（SIMT/SIMD）、线程数、buffer 版本注解。
3. 确认 `compile_flags` 与所选 SDK 的资源规则。
4. 区分三类失败：**编译器容量拒绝** / **厂商编译失败** / **设备启动失败**；
   一次只改一个配置再复现。
5. `annotate_unlimit_memory` 只用于确认是否为调度容量约束，不作为最终修复。

## 5. 性能结论的坑

- 调试接口（`printf`、`asc_dump`、`assert`）会显著改变性能，含调试调用的测量结果不能代表
  kernel 性能；调试完必须移除或关闭开关。
- 模型 cycle、求解出的 II、编译耗时、设备实测时延是四种量，不能互相替代。
- Cube 与 Vector 的利用率分母不同，两者百分比相加不等于 100%。
- 相同生成源码/ELF 不代表运行期地址、cache、输入、设备状态一致；对比实验要逐项固定。
- 清理 TileLang cache 不会重建过期的参考 C++ 扩展，必须同时冻结两个 revision。

## 6. 生成前自检清单

- [ ] 目标锁 `dav-3510`，所有 dtype 在 950 支持列表内（`01` §5）。
- [ ] 网格是一维；线程域只在 `T.SimtVF(threads=...)` 内。
- [ ] 每个 `T.alloc_*` 的容量（含 padding、版本数）在硬件限制内。
- [ ] GEMM 走的是哪种输入路径？transpose 组合与权重布局是否匹配。
- [ ] 分块 K 循环首步 `clear_accum=True`，其余 `False`。
- [ ] 跨 AIV 写同一 buffer 时字节区间不相交，或用 `dual_copy`。
- [ ] 所有归约都有明确的 init/finalize 作用域，且不在 reducer 句柄上做读写。
- [ ] 尾部（tail）、动态长度、零长度、非连续输入有处理路径。
- [ ] 32B 对齐：需要填充时显式写填充值，不依赖自动初始化。
- [ ] 原子累加前已初始化目标；确定性要求是否允许原子。
- [ ] `T.Parallel` body 内没有 GM 直接访问、原子、内嵌循环、跨址邻居读（避免落入 §2 拒绝点）。
- [ ] 输出 shape 与参考实现逐元素可比，且覆盖全部定义域。
- [ ] 编译被拒时先看 `TL_LOWER_TRACE` 的 pass 边界，再决定改结构还是换写法。
- [ ] 调试接口已在最终版本移除。
