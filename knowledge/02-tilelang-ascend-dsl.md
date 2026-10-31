# TileLang Ascend DSL 速查

入口固定为 `import tilelang.ascend.language as T`（Ascend dialect 在通用层之上叠加/覆盖）。
签名以 `<tilelang-repo>/tilelang/ascend/language/` 为准，本文只做检索用速查。

## 1. 程序骨架

```python
import torch, torch_npu, tilelang
import tilelang.ascend.language as T

@tilelang.jit(target="ascend")
def kernel(A, B, block_M: int = 256, block_N: int = 224, block_K: int = 128):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)
    C = T.empty((M, N), T.float32)
    with T.Kernel(32) as bx:
        ...
    return C
```

导出参数用 `T.const("M, N, K")` 声明符号维度；`out_idx` 控制多输出的返回位置。
`tilelang.compile(func, target="ascend")` 与 `@tilelang.jit(target="ascend")` 等价。

## 2. 启动域

| API | 语义 | 约束 |
|---|---|---|
| `T.Kernel(n)` | 1-D 核网格，`bx` 即 `blockIdx.x` | **不接受 `threads=`**；只允许一维 |
| `T.MixedKernel(n, sids=2)` | AIC+AIV 混合核，额外绑定 `sid` | `sids` 取 1/2；3510 物理上按 `__mix__(1,2)` 启动 |
| `T.Cube()` / `T.Vector()` | 显式核作用域 | 属"完全手动"路径，需关闭自动调度 |
| `T.SimtVF(threads=N)` | 线程并行区（SIMT） | `threads` 可为最多三维；编译器按数据冒险自动插 `asc_syncthreads()` |
| `T.SimdVF()` | 寄存器级向量区（SIMD） | 无线程概念；映射 2048-bit CCE 向量指令 |

线程内省：`T.get_thread_binding(dim=0)`、`T.get_thread_bindings()`、
`T.get_thread_extent(dim=0)`、`T.get_thread_extents()`。

2-D 分解需自行计算：`row = tile // n_tiles; col = tile % n_tiles`。

## 3. 内存分配

| API | 物理位置 | 默认 scope |
|---|---|---|
| `T.alloc_shared(shape, dtype)` | UB | `shared.dyn` |
| `T.alloc_l1(shape, dtype)` | L1 / CBuf | `shared.l1` |
| `T.alloc_l0a(shape, dtype)` / `alloc_l0b` / `alloc_l0c` | L0A/L0B/L0C | `shared.l0a/l0b/l0c` |
| `T.alloc_l0a_sf(buf, sf_dtype="uint16", sf_shape=None)` | L0 MX scale 槽 | — |
| `T.alloc_ss(shape, dtype="uint32")` | AIC+2AIV 共享标量 | 3072B 池，shape 必须静态 |
| `T.alloc_fragment(shape, dtype)` | 寄存器 fragment | 常用于 SIMT/SIMD |

版本与容量注解：

- `T.annotate_buffer_versions({buf: 2})` 或 `{buf: (2, "counter"|"iteration"|"auto")}`。
- `{buf: 1}` 表示退出多缓冲；`{buf: (1, "auto")}` 保留资格但只给 1 个物理版本。
- `T.annotate_manual_multi_buffer(...)`：手动多缓冲，**有限支持**，优先用编译器托管。
- `T.annotate_unlimit_memory("shared", "shared.l1")`：解除调度容量约束，不扩物理内存。

## 4. 数据搬运

### 4.1 `T.copy(src, dst, **kw)`

| 关键字 | 适用路径 | 说明 |
|---|---|---|
| `transpose=True` | GM→L1 | 走 `dn2nz`（非 `nd2nz`），并交换逻辑轴；**不是通用转置** |
| `l0_pitch` | L1→L0、L0C→UB/GM | 以 16 元素 M/N block 为单位的矩阵 pitch；A/C 用 `ceildiv(M,16)`，B 用 `ceildiv(N,16)` |
| `l2_cache_ctrl` | GM↔L1、UB↔GM | 整数或策略名（大小写不敏感、后缀匹配），如 `"NOTALLOC_KEEP"` |
| `unit_flag_ctrl` | Cube 相关 | 与后续累加器 drain 配对 |
| `sub_blockid` | 软件双拷贝 | 路由到某个 AIV sub-block |
| `pad_value=v` | **仅 GM→UB** | 行宽向上圆整到 32B 并用 v 填充尾部；等价于先 `ascend_set_copy_pad_value(v)` |
| `data_select=True` | **仅 GM→UB** | 只做填充，复用当前 pad 寄存器值；与 `pad_value` 互斥 |
| `coalesced_width` / `annotations` / `loop_layout` | — | 通用参数 |

L2 策略默认值（`copy_op.py`）：

| 路径 | 默认 |
|---|---|
| `T.copy` GM→L1 / GM→UB | `NORMAL_FV`(0) |
| `T.copy` UB→GM | `NOTALLOC_CI`(4) |
| `T.dual_copy` UB→GM | `NORMAL_FV`(0) |

UB→UB 且目标带 `make_ascend_compact_nz_layout` 时，降低为 ND→NZ scatter；
目标分配需多留一行 padding，拷贝区域要排除该行（`T.copy(src, dst[:rows, :])`）。

### 4.2 `T.dual_copy(src, dst, *, unit_flag_ctrl=None, l2_cache_ctrl=0)`

把逻辑 tile 拆给两个 AIV。**前提是同一 kernel 里确有 Cube 侧工作**，不能用来把纯 Vector
kernel 变成双 AIV 启动。

| 路径 | 实现 | 约束 |
|---|---|---|
| L0C→UB | 硬件双目的拷贝 | 源/目的 dtype 一致，rank≥2；N-split 时 N 需被 32 整除；控制流需对两个 sub-core 一致 |
| GM→UB、UB→GM、UB→L1 | 按 `cthread` sid 的软件拷贝 | 支持一维 2:1；UB→L1 自动做 ND→NZ |

拆分判定：rank≥2 时**恰好一个尾维**呈 2:1，较大者为完整 tile。
`[M,N] ↔ [M/2,N]` 为 M-split，`[M,N] ↔ [M,N/2]` 为 N-split。

### 4.3 其他 DMA 辅助

- `T.ascend_set_copy_pad_value(value, dtype=...)`：写 pad 寄存器（`PIPE_S`），
  `pad_value=` 会自动插该调用。
- `T.ascend_nd2nz_post_copy(...)`：ND→NZ 后处理。

## 5. 计算

```python
T.gemm(A, B, C, transpose_A=False, transpose_B=False,
       policy=GemmWarpPolicy.Square, clear_accum=False,
       unit_flag_ctrl=None, annotations=None)
```

- L1 输入路径：`transpose_A=False`、`transpose_B=True`。
- L0 输入：A/B 区域决定有效 MAD 的 M/N/K，尾维必须从 0 开始且为紧凑 tile；
  分配可因对齐而 padding，但 copy 区域要覆盖物理搬运。
- 分块 K 循环：首步 `clear_accum=True`，后续 `False`。
- `T.gemm_blockscaled(A, B, C, SFA, SFB, ...)`：MXFP8 块缩放 GEMM，MX scale 每 32 个 K 一个因子。

归约（`reduce_op.py`，SimdVF 内直接对 UB 区域发射，不经过 fragment）：

```python
T.reduce(buffer, out, reduce_type, dim, clear, batch=1, annotations=None)
T.reduce_sum / reduce_max / reduce_min / reduce_abssum / reduce_absmax
T.reduce_bitand / reduce_bitor / reduce_bitxor     # 仅整型/bool
```

输出 shape 必须是 `buf_shape[:dim] + buf_shape[dim+1:]` 或把该维保留为 1。

SIMT 内的延迟归约（reducer epoch，支持 `sum`/`max`/`min`）：

```python
with T.SimtVF(threads=128):
    acc = T.alloc_reducer((rows,), "float32", op="sum")
    T.reducer_init(acc)
    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
        T.reducer_update(acc[r], x_frag[r, i] * x_frag[r, i])
    out = T.alloc_fragment((rows,), "float32")
    T.finalize_reducer(acc, out)
```

reducer 句柄本身不能读、不能 `T.clear`/`T.fill`、不能原地 finalize；init 与 finalize
必须在同一层循环/分支作用域内。其他常用：`T.reshape(buf, shape)`、`T.if_then_else(cond, a, b)`、
`T.Parallel(..., coalesced_width=...)`。

原子与存储模式：

- `T.atomic_add/max/min(dst, value, return_prev=...)`：GM 标量原子。
- `T.set_atomic("add", dtype)` / `T.set_atomic_none()`：让 DMA/FixPipe 存储以累加方式写入；
  累加前必须初始化目标。
- `T.set_mmad_direction(...)`、`T.set_hf32_mode(...)`：MAD 方向与 FP32 吞吐取舍。

其他：`T.rsqrt`、`T.unroll(...)`（映射 `#pragma unroll N`）。

## 6. 调度

| API | 用途 |
|---|---|
| `T.Pipelined(extent, num_stages=k)` | 流水循环，编译器做多缓冲与重叠 |
| `T.Persistent([tiles], num_cores, bx)` | 持久化遍历，固定网格跑完所有 tile |
| `T.Stage(k)` | 前端阶段约束，保留自动依赖分析与同步 |
| `T.Task(latency=..., ii=...)` | 把串行语句归为一个调度任务；`latency >= ii > 0` |
| `T.PerCoreTask()` | 同一逻辑操作按 block/core 选择唯一执行点，候选分支必须互斥且完备 |
| `T.assume_no_conflict(a, b, level=..., cross=..., group=...)` / `assume_conflict` | 冲突提示，影响调度保守度 |

持久 GEMM 调度器（`tile_schedule.py`，GROUP_M swizzle + N-first snake）：

```python
sched = T.AscendTileScheduler(block_m=BM, block_n=BN, num_cores=NUM, shape_m=M, shape_n=N)
sched.init(bx)
while sched.valid():
    m_tile, n_tile = sched.m_idx[0], sched.n_idx[0]
    ...
    sched.next_block()
```

变体：`AscendBatchedTileScheduler`、`AscendMGroupedTileScheduler`、`AscendKGroupedTileScheduler`。
`T.Persistent` 是通用写法；需要 swizzle/分组时用上述调度器。

## 7. 同步

**默认自动**：自动调度开（`PassConfigKey.TL_ENABLE_AUTO_SCHEDULE=True`）时，编译器负责
core 分配、flag 配对与 buffer 版本，用户按顺序写操作即可。

手动接口（完全手动路径，或补足编译器无法推断的边）：

```text
T.ascend_pipe_barrier(pipe_t)              # PIPE_ALL/V/M/MTE1/MTE2/MTE3/S
T.ascend_set_flag(hard_event, event_id)    # 如 "MTE2_MTE1"
T.ascend_wait_flag(hard_event, event_id)
T.ascend_sync_inter_arrive / ascend_sync_inter_wait
T.ascend_cross_core_set_flag / ascend_cross_core_wait_flag
T.ascend_threadfence()
T.ascend_get_buf(buf) / T.ascend_rls_buf(buf)
```

手动协议必须整体照抄一个完整示例，禁止混搭半自动/半手动片段。

## 8. SIMD 指令（`T.simd.*`，仅 `T.SimdVF` 内）

| 类别 | 代表接口 |
|---|---|
| 谓词/掩码 | `pset`, `pge`, `update_mask`, `pand`, `por`, `pxor`, `pnot`, `psel`, `ppack`, `punpack`, `pintlv`, `pdintlv` |
| 谓词访存 | `pld(addr, dist=...)`, `pst(addr, src, dist=...)` |
| 向量访存 | `vld(addr, post_inc=...)`, `vld2(addr, off=...)`, `vsts(addr, src, mask, extent=...)`, `vsstb(src, base, stride, ...)` |
| 访存扩展 | `vgather2(base, index, mask=None)`、`vgatherb(base, index, mask=None)`（32B 块索引）、`vscatter(src, base, index, mask=None)` |
| 算术 | `vadd`, `vaddc`, `vsub`, `vsubc`, `vaddcs`, `vsubcs`, `vmul`, `vmull`, `vmula`, `vmadd`, `vaxpy`, `vdiv` |
| 标量广播算术 | `vadds`, `vmuls`, `vmaxs`, `vmins`（第二操作数为标量，`mode="MODE_ZEROING"`） |
| 位/移位 | `vand`, `vor`, `vxor`, `vnot`, `vshl`, `vshr`, `vshls`, `vshrs`（`*s` 为标量移位） |
| 数学 | `vexp`, `vln`, `vsqrt`, `vabs`, `vneg`, `vrelu`, `vlrelu`, `vprelu`, `vabsdif`（`|a-b|`）、`vexpdif`（`exp(a-b)`） |
| 比较/选择 | `vcmp(op=...)`, `vcmps(op=...)`, `vsel(src0, src1, mask)`, `vselr(src, index)` |
| 归约/广播 | `vcpadd`, `vcadd`, `vcmax`, `vcmin`, `vcgadd`, `vcgmax`, `vcgmin`, `vdup`, `vdupv`, `vci` |
| 打包/解包 | `vpack(src, part=0)`（u32→u16、u16→u8）、`vunpack(src, part=0)`（u8→u16、u16→u32） |
| 直方图 | `dhistv2`, `chistv2` |
| 其他 | `vsqz`, `vusqz`, `vmin`, `alloc_var`, `alloc_local` |

`vld`/`vsts`/`pld`/`pst` 没有独立 `off` 参数，位移写进地址；`vld2` 仍保留 `off`。
完整清单见 `tilelang/ascend/language/simd.py`。

**A3 残留注意**：A3 的 `T.tile.*` 命名空间在 A5 已移除，对应 `T.simd.*`；
`T.printf`、`alloc_L1/L0A/L0B/L0C`、`alloc_ub`、`T.ascend.*` 同样不存在。
逐项对照与「A5架构暂无」清单见
[12-op-compatibility.md](12-op-compatibility.md)。

## 9. 调试与随机数

```python
T.print(...)
T.device_assert(cond, ...)       # 降低为工具链 assert 宏
with T.SimtVF(threads=128):
    T.rng_init(seed, seq=..., off=...)
    T.rng_rand()                 # 整型
    T.rng_rand_float()           # 均匀
    T.rng_rand_float(dist="normal")
```

## 10. 写错就会失败的五条

1. `T.Kernel(n, threads=128)` → `TypeError`；线程域只能写在 `T.SimtVF(threads=...)`。
2. 多维 grid → 不支持；自己算 tile 索引。
3. L1 输入 GEMM 改动 `transpose_A/B` → 走错 lowering；先确认所选路径。
4. 在 `T.SimtVF` 里手动加 `__syncthreads` 等价物 → 多余；编译器已按数据流插入。
5. 写了 `import tilelang.ascend.language as T` 后，`T.<op>` 必须是该目录声明面成员，
   `T.simd.<op>` 必须在 `simd.py` 中定义；否则静态检查按 `API001`（未知）、
   `API003`（A3 别名）、`API004`（越界命名空间）拦截。逐项签名查
   [13-ascend-op-reference.md](13-ascend-op-reference.md)。
   reducer 只能 `alloc_reducer → reducer_init → reducer_update → finalize_reducer(acc, out)`，
   把 reducer 当作 `T.reduce_*` 的 out 或直接读 `acc[i]` 会被 `SEM018` 拒绝。
