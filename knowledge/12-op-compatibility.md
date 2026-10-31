# OP 兼容性与 A3→A5 迁移清单

本文是 aiop 的 OP 标准化基线：记录 A3（`tilelang_ascend`）专属语法 OP 与
A5 主仓（`<tilelang-repo>`，已合入 `tilelang/ascend`）的逐项对照、
「A5架构暂无」清单，以及本次新收录的主仓独有标准 OP。

> 事实来源：运行时 `dir(tilelang.ascend.language)`、`dir(tilelang.ascend.language.simd)`
> 与主仓源码；不以函数名猜测能力。复核命令见第 6 节。

## 1. 导入面

```python
import tilelang.ascend.language as T
```

| 面 | 定义位置 | 运行时数量 | 说明 |
|---|---|---|---|
| `T.*` | `tilelang/language/*.py`（通用）+ `tilelang/ascend/language/*.py` | 620 | A5 kernel 唯一合法入口 |
| `T.simd.*` | `tilelang/ascend/language/simd.py` | 94 | 仅 `T.SimdVF()` 内使用 |
| `tilelang.language` | 通用 + `tilelang/cuda/language` | 728 | 默认 CUDA 外观；其中 CUDA-only OP 不经 `T` 导出 |

`T` 上**不存在** `tile` / `ascend` 子命名空间，也不存在 `printf`、
`alloc_L1/L0A/L0B/L0C`、`alloc_ub`、`gemm_v0`、`dump_tensor` 等 A3 名字。

## 2. A3→A5 对照（摘要）

完整逐项表见 skill：
`.agent/skills/tilelang-ascend/references/op-compatibility.md`。

| A3 | A5 | 说明 |
|---|---|---|
| `T.Kernel(n, threads=...)` / `is_npu=True` | `T.Kernel(n)` + `T.SimtVF(threads=...)` | A5 只接受 1-D core grid |
| `T.Scope("C"/"V")` | `T.Cube()` / `T.Vector()` | 配合 `T.MixedKernel` |
| `T.alloc_L1` / `alloc_L0A` / `alloc_L0B` / `alloc_L0C` | `T.alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` | 仅大小写变化 |
| `T.alloc_ub` | `T.alloc_shared` | UB scope 为 `shared.dyn` |
| `T.copy(src, dst, width)`（第 3 位置参数） | `T.copy(src, dst, coalesced_width=width)` | A5 所有 hint 均为 keyword-only |
| `T.gemm(..., k_pack=, wg_wait=)` | `T.gemm(..., unit_flag_ctrl=, annotations=)` | CUDA 旋钮改走 `annotations` |
| `T.mma` / `T.gemm_v0` | `T.gemm` | — |
| `T.set_cross_flag(pipe, flag, mode)` | `T.ascend_cross_core_set_flag(mode, pipe, flag)` | 参数顺序变化 |
| `T.wait_cross_flag(flag, pipe)` | `T.ascend_sync_inter_wait(pipe, flag)` | 或 `ascend_cross_core_wait_flag(0, pipe, flag)` |
| `T.set_flag(src, dst, id)` / `T.wait_flag(src, dst, id)` | `T.ascend_set_flag(f"{src}_{dst}", id)` / `T.ascend_wait_flag(...)` | — |
| `T.barrier_all()` | `T.ascend_pipe_barrier("PIPE_ALL")` | — |
| `T.pipe_barrier(pipe)` | `T.ascend_pipe_barrier("PIPE_" + pipe.upper())` | — |
| `T.printf(fmt, *args)` | `T.print(obj, msg="")` | A5 无格式化变参 |
| `T.tile.add/sub/mul/div/...` | `T.simd.vadd/vsub/vmul/vdiv/...` | 见完整表 |

## 3. 「A5架构暂无」清单

以下 OP 仅在 A3 仓存在，合并后的 A5 主仓无等价实现。生成器不得发射；
如算子确实依赖，按第 5 节登记待适配，不要静默近似。

### 3.1 `tilelang/language/ascend_tile.py`（经 `T.tile` 暴露）

| 原始名称 | 原有功能 | A5 替代建议 |
|---|---|---|
| `arith_progression` | 在 buffer 中生成等差数列 | 用 `T.Parallel` + 索引表达式构造 |
| `bilinear_interpolation` | 双线性插值（A3 已标弃用） | 无；改用显式 gather/乘加分解 |
| `block_reduce_sum` | 块级求和归约 | 无块级 OP；整 buffer 用 `T.reduce_sum`，块内用 `T.alloc_reducer` |
| `block_reduce_max` | 块级最大值归约 | 同上，用 `T.reduce_max` / `T.alloc_reducer(op="max")` |
| `block_reduce_min` | 块级最小值归约 | 同上，用 `T.reduce_min` / `T.alloc_reducer(op="min")` |
| `brcb` | 广播重复拷贝块（BRCB） | 无；用 `T.simd.vdup/vdupv` 或显式循环展开 |
| `brcb_experiment` | 双后端 BRCB 实验版 | 无 |
| `datacachecleanandinvalid_experiment` | 数据缓存清理/失效 | 无；依赖编译器与运行时一致性 |
| `init_sort_buf` | 排序 buffer 初始化 | 无（排序族整体未迁移） |
| `merge_sort` | 2/3/4 路归并排序 | 无 |
| `reciprocal` | 逐元素倒数 `1/x` | 无单指令；用 `T.simd.vdiv(1, x)` 组合 |
| `reduce_sum_mask_experiment` | 带掩码的全量求和 | 无；用 `T.reduce_sum` + 预置 0 的掩码 buffer |
| `row_expand_mul` | 行广播乘 `dst[i,j]=src0[i,j]*src1[i]` | 无；用 `T.Parallel` 索引广播 |
| `row_expand_mul_experiment` | 行广播乘实验版 | 无 |
| `row_expand_sub_experiment` | 行广播减实验版 | 无 |
| `row_expand_div_experiment` | 行广播除实验版 | 无 |
| `silu` | SiLU 激活 `x*sigmoid(x)` | 无单指令；由 `T.simd.vexp` 组合 |
| `sort` | 降序排序，输出 (value,index) 交错 | 无 |
| `sort32` | 32 元素块排序 | 无 |
| `topk` | 排序式 Top-K 抽取 | 无 |

### 3.2 `tilelang/language/ascend.py`

| 原始名称 | 原有功能 | A5 替代建议 |
|---|---|---|
| `sync_all` | 全计算单元同步 intrinsic | `T.ascend_threadfence()` 为最接近的栅栏，但语义更弱 |
| `dump_tensor(tensor, desc, dump_size, shape_info)` | 把指定 tensor 区域 dump 到 host | `T.print(buffer)`；无 dump 尺寸/描述控制 |
| `reinterpretcast(dst, src, casttype)` | 按 `casttype` 字符串按位重解释 buffer | `T.view(buf, dtype=...)` 同存储视图；任意 casttype 字符串不支持 |
| `set_deq_scale(scale)` | 设置反量化 scale 寄存器 | 无寄存器控制；在 kernel 内显式乘 scale |
| `shmem_put_nbi(dst, src, nelems, newPe)` | 非阻塞写入 peer GM | 无；经 GM 中转 + `T.copy` |
| `shmem_ub_put_nbi(ub, dst, nelems, newPe, strelem)` | UB → peer GM 非阻塞写 | 无；经 GM 中转 |
| `shmem_get_nbi(dst, src, nelems, newPe)` | 从 peer GM 非阻塞读 | 无；经 GM 中转 |
| `shmem_ub_get_nbi(dst, ub, nelems, newPe, strelem)` | peer GM → UB 非阻塞读 | 无；经 GM 中转 |

### 3.3 部分等价（需改写，不可逐字替换）

| 原始名称 | A5 近似 | 限制 |
|---|---|---|
| `gather_mask` / `gathermask_experiment` | `T.simd.vgatherb` | A3 支持自定义 mask 模式；A5 只有 32B 块索引 gather |
| `transpose` | `T.transpose` | A3 是向量转置；A5 `T.transpose` 是矩阵/DMA 转置 |
| `pow` | `T.pow`（标量） | 无向量 `vpow`；向量场景需 `vexp(vln(x)*y)` |
| `sin` / `cos` / `round` | `T.sin` / `T.cos` / `T.round`（标量） | 无对应向量 intrinsic |
| `sigmoid` | `T.sigmoid`（标量） | 向量场景需 `1/(1+vexp(-x))` |

## 4. 本次新增收录的主仓独有标准 OP

以下 OP 在合并主仓中存在、但此前未被 aiop 的 skill/知识库收录。签名以运行时
`inspect.signature` 为准；示例均为 kernel 内片段。

### 4.1 A5 SIMD 补充（`T.simd.*`，仅 `T.SimdVF()`）

#### `vadds / vmuls / vmaxs / vmins`

- **定义**：向量与标量的逐 lane 加/乘/取大/取小。
- **签名**：`vadds(src, scalar, mask=None, mode="MODE_ZEROING")`，其余同名同签名。
- **参数**：`src` 向量寄存器；`scalar` 标量 PrimExpr；`mask` 谓词，缺省按 dtype 全 1；
  `mode` 越界 lane 处理策略。
- **示例**：

  ```python
  with T.SimdVF():
      mask = T.simd.pset(32)
      r = T.simd.vld(x[off])
      y = T.simd.vmuls(r, T.float32(0.5), mask)
      T.simd.vsts(out[off], y, mask)
  ```

- **适用场景**：逐元素 affine 变换、ReLU 的 scalar 形式、缩放/偏置。
  `vmaxs(x, 0)` 等价 ReLU；`vmins/vmaxs` 组合等价 clamp。

#### `vshls / vshrs`

- **定义**：向量按标量左/右移。
- **签名**：`vshls(src, scalar, mask=None, mode="MODE_ZEROING")`、`vshrs(...)`。
- **示例**：`T.simd.vshls(r, 2, mask)`（整型向量）。
- **适用场景**：2 的幂缩放、定点量化/反量化、位域打包。

#### `vabsdif`

- **定义**：单指令 `dst = |src0 - src1|`。
- **签名**：`vabsdif(src0, src1, mask=None, mode="MODE_ZEROING")`。
- **适用场景**：L1/L2 距离、误差统计、SAD 类算子。
- **dtype**：float32 / float16。

#### `vexpdif`

- **定义**：单指令 `dst = exp(src0 - src1)`。
- **签名**：`vexpdif(src0, src1, mask=None)`。
- **约束**：两侧 dtype 必须一致且为 float32；fp16 需拆 even/odd，不能用本接口。
- **适用场景**：softmax 的 `exp(x - max)` 稳定化路径。

#### `vsel / vselr`

- **定义**：`vsel` 按谓词位选 `mask ? src0 : src1`；`vselr` 按每 lane 索引从 `src`
  选 lane。
- **签名**：`vsel(src0, src1, mask)`、`vselr(src, index)`。
- **示例**：

  ```python
  r = T.simd.vsel(a, b, T.simd.vcmp(a, b, mask, "lt"))
  ```

- **适用场景**：无分支 select、shuffle/查表、masked 数据重排。
  `vselr` 的 index 为无符号整型向量；float32 会自动按位重解释。

#### `vpack / vunpack`

- **定义**：`vpack` 把宽 lane 压到窄 lane（u32→u16、u16→u8，也支持 float32→float16）；
  `vunpack` 反向扩宽（u8→u16、s8→s16、u16→u32、s16→s32）。
- **签名**：`vpack(src, part=0)`、`vunpack(src, part=0)`；`part` 为 `0/"LOWER"` 或
  `1/"HIGHER"`。
- **适用场景**：MX scale（UE8M0）密集打包、int8/fp8 量化、带宽压缩。

#### `vgather2 / vgatherb / vscatter`

- **定义**：`vgather2` 按每 lane 元素偏移 gather；`vgatherb` 按 32B 块偏移 gather；
  `vscatter` 按每 lane 偏移散射写。
- **签名**：`vgather2(base, index, mask=None)`、`vgatherb(base, index, mask=None)`、
  `vscatter(src, base, index, mask=None)`。
- **参数**：`base` 为 buffer 元素或 `access_ptr`；`index` 为 uint32（32-bit 元素）
  或 uint16（8/16-bit 元素）向量。
- **适用场景**：embedding 查表、MoE token 重排、稀疏索引写回、top-k 结果搬移。
- **注意**：gather 的 mask 按源 dtype 位宽对齐；scatter 的 `index` 是元素偏移，
  不是字节偏移。

#### `vmin`

- **定义**：双向量逐 lane 最小值（`vmax` 的对偶）。
- **签名**：`vmin(src0, src1, mask=None, mode="MODE_ZEROING")`。
- **适用场景**：clamp 上界、min-plus 组合、逐元素上截断。

### 4.2 主仓通用面补充（`T.*`，A5 前端已导出）

#### `all_of / any_of`

- **定义**：对整 buffer 做布尔全称/存在判断，返回标量 PrimExpr。
- **签名**：`all_of(buffer)`、`any_of(buffer)`。
- **示例**：`if T.all_of(mask_ub): ...`
- **适用场景**：全局条件检查、早退判断、校验分支。A5 lowering 以实测为准；
  性能敏感路径优先用 reducer。

#### `cumsum / cummax`

- **定义**：沿 `dim` 的累积和 / 累积最大值，可反向。
- **签名**：`cumsum(src, dst=None, dim=0, reverse=False, annotations=None)`，
  `cummax(...)` 同签名。
- **示例**：`T.cumsum(x_frag, y_frag, dim=-1)`
- **适用场景**：前缀和、采样概率累积、单调约束、Top-P 截断。

#### `alloc_global / alloc_barrier`

- **定义**：`alloc_global` 申请 GM workspace buffer；`alloc_barrier` 申请 mbarrier
  计数 buffer。
- **签名**：`alloc_global(shape, dtype, scope="global")`、
  `alloc_barrier(arrive_count)`。
- **适用场景**：跨核 workspace、持久化中间结果；barrier 仅用于支持 mbarrier 的
  后端（A5 上以 lowering 支持为准）。

#### `atomic_addx2 / atomic_addx4`

- **定义**：双/四宽原子加，按向量宽度合并原子请求。
- **签名**：`atomic_addx2(dst, value, return_prev=False)`、`atomic_addx4(...)`。
- **适用场景**：高竞争累加、批量 scatter-add；比逐元素 `T.atomic_add` 少事务。

#### `atomic_load / atomic_store / atomic_or`

- **定义**：带内存序的原子读/写/按位或。
- **签名**：`atomic_load(src, memory_order="seq_cst")`、
  `atomic_store(dst, src, memory_order="seq_cst")`、
  `atomic_or(dst, value, memory_order=None)`。
- **适用场景**：跨核标志位、计数器、无锁队列；A5 支持度以实测为准。

#### `add2 / sub2 / mul2 / max2 / min2 / abs2 / fma2`

- **定义**：打包标量/向量成对运算（`fma2` = `x*y+z`），一条指令处理两路。
- **签名**：`add2(x, y)`、`sub2(x, y)`、`mul2(x, y)`、`max2(x, y)`、`min2(x, y)`、
  `abs2(x)`、`fma2(x, y, z)`。
- **适用场景**：fp32x2 成对计算、提升 Vector 吞吐；与 `T.alloc_fragment` 的
  偶数 lane 布局配合。

#### `im2col`

- **定义**：2-D 卷积的 im2col 变换，把滑窗展开成矩阵列。
- **签名**：`im2col(img, col, nhw_step, c_step, kernel, stride, dilation, pad, annotations=None)`。
- **适用场景**：卷积走 GEMM 的 lowering；A5 后端支持度以实测为准。
  `c2d_im2col` 为已弃用别名。

#### `dp4a`

- **定义**：四元素 int8 点积累加到 int32。
- **签名**：`dp4a(A, B, C)`。
- **适用场景**：int8 量化 GEMM/卷积的 SIMT 侧累加；A5 lowering 以实测为准。

#### `annotate_layout / annotate_safe_value / annotate_restrict_buffers`

- **定义**：`annotate_layout(layout_map)` 显式绑定 buffer layout；
  `annotate_safe_value(safe_value_map)` 声明越界安全填充值；
  `annotate_restrict_buffers(*buffers)` 关闭对应 buffer 的 restrict 假设。
- **适用场景**：手工 layout（NZ/compact）、guarded tail 的填充语义、别名 buffer
  的正确性约束。
- **示例**：`T.annotate_safe_value({x_ub: 0.0})`

#### `make_tensor / make_tensor_from_addr / ptr`

- **定义**：从指针/地址构造 TIR buffer；`ptr` 创建指针变量。
- **签名**：`make_tensor(ptr, shape, dtype="float32", strides=None)`、
  `make_tensor_from_addr(addr, shape, dtype="float32", strides=None, storage_scope="global")`、
  `ptr(dtype=None, storage_scope="global", *, is_size_var=False)`。
- **适用场景**：自定义 C 入口、外部 workspace 绑定、`import_source` 手写 kernel
  与 TIR 混编。

#### `get_block_binding / get_block_bindings / get_block_extent / get_block_extents`

- **定义**：读取当前 launch 的 block（core）索引/范围。
- **签名**：`get_block_binding(dim=0)`、`get_block_bindings()`、
  `get_block_extent(dim=0)`、`get_block_extents()`。
- **适用场景**：手动 index 计算、`T.MixedKernel` 内的 AIC/AIV 分工、
  自定义 swizzle。A5 的 `bx` 也可直接从 `T.Kernel(...) as bx` 获取。

#### `index_to_coordinates`

- **定义**：把扁平索引拆成给定 shape 的多维坐标。
- **签名**：`index_to_coordinates(index, shape)`。
- **适用场景**：持久化 tile 循环里由 `tile` 反解 `(m, n)`，替代手写
  `tile // N_TILES, tile % N_TILES`。

#### `AscendBaseTileScheduler`

- **定义**：A5 持久 GEMM 调度器基类；具体实现为 `AscendTileScheduler` /
  `AscendBatchedTileScheduler` / `AscendMGroupedTileScheduler` /
  `AscendKGroupedTileScheduler`。
- **用法**：继承后实现 tile 序（GROUP_M swizzle、N-first snake），再用
  `init(bx) / valid() / next_block()` 驱动。
- **适用场景**：需要自定义 tile 遍历顺序或分组策略的 GEMM/attention kernel；
  默认优先 `T.Persistent`。

## 5. 使用规则

1. 只 import `tilelang` 与 `tilelang.ascend.language as T`。
2. 禁止 A3 拼写：`T.printf`、`T.tile.*`、`T.ascend.*`、`T.alloc_L1/L0A/L0B/L0C`、
   `T.alloc_ub`、`T.gemm_v0`、`T.mma`。静态检查会报 `API003`。
3. `API003` 消息会给出 A5 替代名；若提示「A5架构暂无」，登记到本文件的
   第 3 节并选择受支持的分解方案，不要静默替换成语义不同的 OP。
4. 新增 OP 时先查本文件与 skill 的
   `references/op-compatibility.md`，再查源码 `tilelang/ascend/language/`。

## 6. 复核命令

```bash
cd <aiop-repo-root>
python -m pytest -q                                   # 回归
python -m aiop check --kernel <file>                  # API001/002/003
python -m aiop search "T.simd vgatherb vscatter"      # 知识库召回
```

```python
import tilelang.ascend.language as T
assert not hasattr(T, "tile") and not hasattr(T, "ascend") and not hasattr(T, "printf")
assert {"vgatherb", "vscatter", "vpack", "vselr"} <= set(dir(T.simd))
```
