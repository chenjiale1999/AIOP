# Ascend 专属 OP 参考（A5 / Ascend 后端）

> 本文件由 `scripts/gen_ascend_op_kb.py` 从源码自动生成，请勿手工编辑。
> 解析目录：`/home/cjl/workspace/tilelang/tilelang/ascend/language`
> 覆盖：目录内 158 个公开定义（`T.*` 72 个，`T.simd.*` 86 个）
> 合法调用面：`tilelang.ascend.language.__all__` 606 项（含再导出的通用面）

## 0. 使用规则

1. 只允许 `import tilelang.ascend.language as T`（或 `from tilelang.ascend import language as T`）。
2. `T.<op>` 必须是 `tilelang.ascend.language` 的声明面成员；`T.simd.<op>` 必须是 `tilelang/ascend/language/simd.py` 中定义的算子。
3. 禁止 `T.tile.*`、`T.ascend.*`、`T.printf`、`alloc_L1/L0A/L0B/L0C/ub` 等 A3 拼写。
4. 非法调用由静态检查拦截：`API001`（未知）、`API003`（A3 别名）、`API004`（越界命名空间）；reducer 误用由 `SEM018` 拦截。
5. 本文件是生成算子代码时的第一手 OP 参考；签名与行为以源码为准，本文随源码重新生成。

## 1. 覆盖总览

| 模块 | 标题 | 算子数 |
|---|---|---:|
| `allocate.py` | 内存分配（UB / L1 / L0A / L0B / L0C / SS） | 8 |
| `annotations.py` | buffer 版本与容量注解 | 3 |
| `copy_op.py` | 数据搬运（DMA / MTE） | 2 |
| `dma.py` | DMA 辅助（pad 寄存器 / ND→NZ） | 2 |
| `frame.py` | AIC / AIV 作用域 | 9 |
| `gemm_op.py` | Cube GEMM | 2 |
| `kernel.py` | 核启动与线程域 | 9 |
| `loop.py` | 循环构造 | 1 |
| `math.py` | 标量数学 | 1 |
| `mode.py` | Cube / 存储模式控制 | 4 |
| `print.py` | 调试打印 | 1 |
| `random.py` | 随机数 | 1 |
| `reduce_op.py` | 归约 | 9 |
| `schedule_hint.py` | 调度提示 | 5 |
| `sync.py` | PIPE / 跨核同步 | 10 |
| `tile_schedule.py` | 持久化 tile 调度 | 5 |
| `simd.py` | SIMD MicroAPI | 86 |
| **合计** | — | **158** |

## 2. `T.*` 顶层方言算子

### 2.1 `allocate.py` — 内存分配（UB / L1 / L0A / L0B / L0C / SS）

#### `T.alloc_shared`

- **签名**：`T.alloc_shared(shape: ShapeType, dtype: DType, scope='shared.dyn')`
- **核心功能**：Allocate a UB buffer. Same surface as the common ``T.alloc_shared`` minus its CUDA-only bool workaround (bool buffers there downgrade to the static "shared" scope because the smem-merge pass cannot handle bool): Ascend UB handles bool buffers in the requested scope directly.
- **入参**：
  - `shape` (ShapeType)：buffer 形状（静态或符号维度）；需满足对齐与容量约束
  - `dtype` (DType)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
  - `scope` (默认 `'shared.dyn'`)：内存 scope；默认值见签名，必须与目标存储层级一致
- **返回值**：返回 `Buffer`
- **A5 硬件适配**：UB。片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用
- **性能建议**：
  - 按最小生命周期分配；复用交给 AutoSchedule 多缓冲，避免手工放大版本数
- **示例**：

```python
T.alloc_shared(<shape>, <dtype>)
```

- **注意事项**：
  - Same surface as the common ``T.alloc_shared`` minus its CUDA-only bool workaround (bool buffers there downgrade to the static "shared" scope because the smem-merge pass cannot handle bool): Ascend UB handles bool buffers in the requested scope directly.

#### `T.alloc_l1`

- **签名**：`T.alloc_l1(shape: ShapeType, dtype: DType, scope='shared.l1')`
- **核心功能**：Allocate an L1 buffer (__cbuf__) on Ascend NPU.
- **入参**：
  - `shape` (ShapeType)：Buffer shape
  - `dtype` (DType)：Data type
  - `scope` (默认 `'shared.l1'`)：Memory scope. Defaults to "shared.l1"
- **返回值**：返回 `Buffer`
- **A5 硬件适配**：L1/CBuf。片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用
- **性能建议**：
  - 按最小生命周期分配；复用交给 AutoSchedule 多缓冲，避免手工放大版本数
- **示例**：

```python
T.alloc_l1(<shape>, <dtype>)
```

- **注意事项**：
  - shape 与 scope 必须满足片上容量与对齐约束

#### `T.alloc_ss`

- **签名**：`T.alloc_ss(shape: ShapeType, dtype: DType='uint32')`
- **核心功能**：Allocate scalar storage shared by one AIC and its two AIVs. SS buffers have static shapes, uint32/uint64 elements, and 32-byte aligned allocation bases within the MIX 1:2 group's 3 KiB SS pool. Allocation does not initialize the contents. ``T.view`` may reinterpret the same storage. SS follows ordinary buffer optimi...
- **入参**：
  - `shape` (ShapeType)：buffer 形状（静态或符号维度）；需满足对齐与容量约束
  - `dtype` (DType, 默认 `'uint32'`)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
- **返回值**：返回 `Buffer`
- **A5 硬件适配**：AIC/Cube、AIV/Vector、UB、L1/CBuf、跨核同步、原子。片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用；AIV/AIC dependencies use ordinary automatic cross-core synchronization.
- **性能建议**：
  - SS buffers have static shapes, uint32/uint64 elements, and 32-byte aligned allocation bases within the MIX 1:2 group's 3 KiB SS pool.
  - Synchronization establishes publication and slot-reuse order.
- **示例**：

```python
T.alloc_ss(<shape>)
```

- **注意事项**：
  - f'alloc_ss requires uint32 or uint64, got {dtype}'
  - alloc_ss requires a nonempty static shape with positive extents
  - alloc_ss exceeds the MIX 1:2 SS capacity of 3072 bytes
  - The two AIVs must write disjoint byte ranges.

#### `T.alloc_l0a`

- **签名**：`T.alloc_l0a(shape: ShapeType, dtype: DType, scope='shared.l0a')`
- **核心功能**：Allocate an L0A buffer (__ca__) on Ascend NPU. Layout is deferred to the consuming gemm operation, which infers K-major or MN-major from its transpose flags.
- **入参**：
  - `shape` (ShapeType)：buffer 形状（静态或符号维度）；需满足对齐与容量约束
  - `dtype` (DType)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
  - `scope` (默认 `'shared.l0a'`)：内存 scope；默认值见签名，必须与目标存储层级一致
- **返回值**：返回 `Buffer`
- **A5 硬件适配**：L0A。片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用
- **性能建议**：
  - 按最小生命周期分配；复用交给 AutoSchedule 多缓冲，避免手工放大版本数
- **示例**：

```python
T.alloc_l0a(<shape>, <dtype>)
```

- **注意事项**：
  - shape 与 scope 必须满足片上容量与对齐约束

#### `T.alloc_l0b`

- **签名**：`T.alloc_l0b(shape: ShapeType, dtype: DType, scope='shared.l0b')`
- **核心功能**：Allocate an L0B buffer (__cb__) on Ascend NPU. Layout is deferred to the consuming gemm operation, which infers K-major or MN-major from its transpose flags.
- **入参**：
  - `shape` (ShapeType)：buffer 形状（静态或符号维度）；需满足对齐与容量约束
  - `dtype` (DType)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
  - `scope` (默认 `'shared.l0b'`)：内存 scope；默认值见签名，必须与目标存储层级一致
- **返回值**：返回 `Buffer`
- **A5 硬件适配**：L0B。片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用
- **性能建议**：
  - 按最小生命周期分配；复用交给 AutoSchedule 多缓冲，避免手工放大版本数
- **示例**：

```python
T.alloc_l0b(<shape>, <dtype>)
```

- **注意事项**：
  - shape 与 scope 必须满足片上容量与对齐约束

#### `T.alloc_l0c`

- **签名**：`T.alloc_l0c(shape: ShapeType, dtype: DType, scope='shared.l0c', layout: bool=True)`
- **核心功能**：Allocate an L0C buffer (__cc__) on Ascend NPU.
- **入参**：
  - `shape` (ShapeType)：Buffer shape
  - `dtype` (DType)：Data type
  - `scope` (默认 `'shared.l0c'`)：Memory scope. Defaults to "shared.l0c"
  - `layout` (bool, 默认 `True`)：Whether to annotate the fixed Ascend L0C accumulator layout.
- **返回值**：返回 `Buffer`
- **A5 硬件适配**：L0C。片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用
- **性能建议**：
  - 按最小生命周期分配；复用交给 AutoSchedule 多缓冲，避免手工放大版本数
- **示例**：

```python
T.alloc_l0c(<shape>, <dtype>)
```

- **注意事项**：
  - shape 与 scope 必须满足片上容量与对齐约束

#### `T.alloc_l0a_sf`

- **签名**：`T.alloc_l0a_sf(buf: Buffer, sf_dtype: DType='uint16', sf_shape: ShapeType | None=None)`
- **核心功能**：Return the MX scale-factor handle of an L0A data tile. The handle (scope ``shared.l0a.sf``) materializes no storage: the hardware keys a tile's scale slots to the tile's own address. Load scales with ``T.copy(sf_l1, handle)`` and pass the handle to ``T.gemm_blockscaled`` as ``SFA``. Allocation permanently binds this...
- **入参**：
  - `buf` (Buffer)：见签名与源码约束
  - `sf_dtype` (DType, 默认 `'uint16'`)：见签名与源码约束
  - `sf_shape` (ShapeType | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `Buffer`
- **A5 硬件适配**：L1/CBuf、L0A。片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用
- **性能建议**：
  - Return the MX scale-factor handle of an L0A data tile.
  - The handle (scope ``shared.l0a.sf``) materializes no storage: the hardware keys a tile's scale slots to the tile's own address.
- **示例**：

```python
T.alloc_l0a_sf(<buf>)
```

- **注意事项**：
  - shape 与 scope 必须满足片上容量与对齐约束

#### `T.alloc_l0b_sf`

- **签名**：`T.alloc_l0b_sf(buf: Buffer, sf_dtype: DType='uint16', sf_shape: ShapeType | None=None)`
- **核心功能**：Return the MX scale-factor handle of an L0B data tile. See :func:`alloc_l0a_sf`; identical semantics for the L0B slot shadow.
- **入参**：
  - `buf` (Buffer)：见签名与源码约束
  - `sf_dtype` (DType, 默认 `'uint16'`)：见签名与源码约束
  - `sf_shape` (ShapeType | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `Buffer`
- **A5 硬件适配**：L0A、L0B。片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用
- **性能建议**：
  - Return the MX scale-factor handle of an L0B data tile.
- **示例**：

```python
T.alloc_l0b_sf(<buf>)
```

- **注意事项**：
  - shape 与 scope 必须满足片上容量与对齐约束

### 2.2 `annotations.py` — buffer 版本与容量注解

#### `T.annotate_buffer_versions`

- **签名**：`T.annotate_buffer_versions(buffer_versions_map: dict)`
- **核心功能**：Control automatic multi-buffer version counts and indexing modes. When using T.Persistent with num_stages > 1, the Z3 auto-scheduler computes optimal buffer version counts for each on-chip buffer. Each mapping value may be one of: - ``1``: opt out of multi-buffer eligibility, including explicit owner claims; - ``num...
- **入参**：
  - `buffer_versions_map` (dict)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。编译期注解：多缓冲版本、手动多缓冲、容量约束解除；不直接产生硬件指令
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
@T.prim_func
```

- **注意事项**：
  - f'buffer version tuple must be (num_versions, mode), got {specification!r} for buffer {buffer}'
  - f'num_versions must be a positive integer, got {num_versions} for buffer {buffer}'
  - f'aliases of one buffer storage must use the same version count, got {versions[data]} and {num_versions}'
  - Overriding explicit ``multi_buffer_eligible`` claims emits a warning once per storage.

#### `T.annotate_manual_multi_buffer`

- **签名**：`T.annotate_manual_multi_buffer(*items)`
- **核心功能**：Declare buffers that the user has *manually* multi-buffered. Unlike :func:`annotate_buffer_versions` (which asks the Z3 auto-scheduler to version a buffer for you), this annotation tells AutoSchedule that the buffer is *already* multi-buffered by hand: the user writes the version index into every access themselves (...
- **入参**：
  - `*items`：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。编译期注解：多缓冲版本、手动多缓冲、容量约束解除；不直接产生硬件指令
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
@T.prim_func
```

- **注意事项**：
  - f'manual multi-buffer version count must be >= 1, got {n} for buffer {buffer}'
  - f'annotate_manual_multi_buffer needs a constant leading (version) dimension to infer N for buffer {buffer}; pass {{buffer: N}} to specify it explicitly'
  - ``buf[l % 2]`` / ``buf[(l + 1) % 2]`` ping-pong), and the pass only needs to analyze dependencies and emit the ``set_flag`` / ``wait_flag`` synchronization.
  - Each positional argument is either: - a bare buffer: ``N`` defaults to its leading dimension ``buffer.shape[0]`` (which must then be a compile-time constant) — the common ``T.alloc_shared((N, ...))`` ping-pong case;

#### `T.annotate_unlimit_memory`

- **签名**：`T.annotate_unlimit_memory(*scopes: str)`
- **核心功能**：Remove the memory capacity limit for the given scopes during auto-scheduling. When unlimit'd, a scope's capacity is effectively unbounded so Z3 will not constrain multi-buffer versions based on that memory pool. Use this when you know a particular scope has spare headroom. Valid scopes: ``"shared"``, ``"shared.l1"``...
- **入参**：
  - `*scopes`：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：L1/CBuf、L0A、L0B、L0C。编译期注解：多缓冲版本、手动多缓冲、容量约束解除；不直接产生硬件指令
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
@T.prim_func
```

- **注意事项**：
  - f'Invalid memory scope {scope!r}, must be one of {valid}'

### 2.3 `copy_op.py` — 数据搬运（DMA / MTE）

#### `T.dual_copy`

- **签名**：`T.dual_copy(src: BufferLikeType, dst: BufferLikeType, *, unit_flag_ctrl: int | tirx.PrimExpr | None=None, l2_cache_ctrl: int | str=0)`
- **核心功能**：Copy a region using an M- or N-split across the two AIVs. ``dual_copy`` is a mixed-kernel operation: the user must also provide the Cube-side work in the enclosing kernel. It is not supported as a way to turn an otherwise pure Vector kernel into a two-AIV launch; the compiler assumes this contract and does not synth...
- **入参**：
  - `src` (BufferLikeType)：Source region in GM, L0C, or UB, according to the supported paths.
  - `dst` (BufferLikeType)：Destination region in UB, GM, or L1, according to the supported paths.
  - `unit_flag_ctrl` (int | tirx.PrimExpr | None, 默认 `None`)：Unit flag control (0=manual sync, 3=pipelined with mad). ``None`` omits the annotation and lowers as 0.
  - `l2_cache_ctrl` (int | str, 默认 `0`)：Ascend L2 cache control policy for the UB→GM store path. Accepts an integer or case-insensitive string name (e.g. ``"notalloc_clean"``). Defaults to 0 (``"normal_fv"``). Ascend only; ignored on other backends.
- **返回值**：返回 `tirx.PrimExpr | tirx.Stmt`
- **A5 硬件适配**：Cube、AIV/Vector、UB、L1/CBuf、L0C、PIPE。DMA/MTE 搬移路径（GM↔UB、GM↔L1、L1↔L0、L0C↔UB/GM），可带 L2 cache 策略与 NZ 转换
- **性能建议**：
  - The rewrite pass reuses that block's ``cthread`` sid and rejects unscoped software dual copies.
  - The larger region is the full logical tile.
- **示例**：

```python
T.dual_copy(<src>, <dst>)
```

- **注意事项**：
  - ValueError: If the split direction cannot be inferred from shapes.
  - f'Cannot infer dual_copy split direction: src shape {src_shape} vs dst shape {dst_shape}. ' + ratio_requirement
  - f'dual_copy: src dtype ({src_dtype}) != dst dtype ({dst_dtype}). Type conversion during cc->ub dual-destination copy is NOT supported on dav-950. Use a separate copy or cast inside a VF block instead.'
  - dual_copy requires source and destination regions that are both one-dimensional or both have rank at least two

#### `T.copy`

- **签名**：`T.copy(src: BufferLikeType, dst: BufferLikeType, *, coalesced_width: int | None=None, transpose: bool=False, l0_pitch: int | tirx.PrimExpr | None=None, l2_cache_ctrl: int | str | None=None, unit_flag_ctrl: int | tirx.PrimExpr | None=None, sub_blockid: int | tirx.PrimExpr | None=None, pad_value: int | float | tirx.PrimExpr | None=None, data_select: bool=False, annotations: dict | None=None, loop_layout: Any | None=None)`
- **核心功能**：Copy data between memory regions, with Ascend DMA lowering hints. Uses the common region handling with Ascend-specific whole-buffer checks. L1-to-L0A/L0B copies default to a compact tile whose pitch comes from the destination region. When assembling a tile with multiple copies, pass ``l0_pitch`` on each copy to spec...
- **入参**：
  - `src` (BufferLikeType)：Source memory region (Buffer, BufferLoad or BufferRegion).
  - `dst` (BufferLikeType)：Destination memory region.
  - `coalesced_width` (int | None, 默认 `None`)：Width for coalesced memory access. Defaults to None.
  - `transpose` (bool, 默认 `False`)：Ascend GM→L1 only. Emit dn2nz (which transposes the N/D mapping) instead of nd2nz. Defaults to False.
  - `l0_pitch` (int | tirx.PrimExpr | None, 默认 `None`)：L0 matrix pitch in 16-element M/N fractal blocks, not bytes or an inter-block gap. For L1→L0A/L0B, overrides the destination pitch; for L0C→UB/GM, overrides the source pitch. Use ``ceildiv(M, 16)`` for A/C and ``ceildiv(N, 16)`` for B, including MajorMN transposed loads. May vary at runtime. ``None`` keeps the compact region pitch.
  - `l2_cache_ctrl` (int | str | None, 默认 `None`)：L2 cache control for the GM↔L1 and UB↔GM DMA paths. Accepts the raw integer or a cache-policy name (case-insensitive, suffix-matched), e.g. ``"NOTALLOC_KEEP"``.
  - `unit_flag_ctrl` (int | tirx.PrimExpr | None, 默认 `None`)：Unit flag control for the Cube instruction; ``None`` omits the annotation.
  - `sub_blockid` (int | tirx.PrimExpr | None, 默认 `None`)：Sub-block id that routes the copy to one of the AIV sub-blocks.
  - `pad_value` (int | float | tirx.PrimExpr | None, 默认 `None`)：Ascend GM→UB only. Round the row width up to the next 32B boundary and fill the pad lanes with this value. Emits a leading ``T.ascend_set_copy_pad_value(value)`` so AutoSchedule syncs the pad-register write before the copy. Mutually exclusive with ``data_select``.
  - `data_select` (bool, 默认 `False`)：Ascend GM→UB only. Same right-pad behavior as ``pad_value``, but reuses whatever the hardware pad register currently holds, which the caller must have set via ``T.ascend_set_copy_pad_value(...)`` beforehand.
  - `annotations` (dict | None, 默认 `None`)：Additional annotations dict; values in it take precedence over the individual keywords.
  - `loop_layout` (Any | None, 默认 `None`)：Parallel loop layout hint for the SIMT copy path.
- **返回值**：tirx.PrimExpr | tirx.Stmt: A handle to the copy operation.
- **A5 硬件适配**：Cube、AIV/Vector、UB、L1/CBuf、L0A、L0B、L0C、MTE/DMA、fractal 布局。DMA/MTE 搬移路径（GM↔UB、GM↔L1、L1↔L0、L0C↔UB/GM），可带 L2 cache 策略与 NZ 转换；When assembling a tile with multiple copies, pass ``l0_pitch`` on each copy to specify the full tile's fractal pitch.
- **性能建议**：
  - L1-to-L0A/L0B copies default to a compact tile whose pitch comes from the destination region.
  - When assembling a tile with multiple copies, pass ``l0_pitch`` on each copy to specify the full tile's fractal pitch.
- **示例**：

```python
T.copy(<src>, <dst>)
```

- **注意事项**：
  - T.copy: pad_value and data_select are mutually exclusive. Pass pad_value to set the fill value on this copy, or data_select to reuse a pad value already set via T.ascend_set_copy_pad_value().
  - f'Ascend T.copy src/dst element count mismatch: {src.name} shape={src.shape} ({src_elems}) vs {dst.name} shape={dst.shape} ({dst_elems}). Use explicit regions for a partial copy.'
  - Ascend T.copy l0_pitch must be an integer or integer PrimExpr
  - the caller must match the consuming GEMM's effective M/N.

### 2.4 `dma.py` — DMA 辅助（pad 寄存器 / ND→NZ）

#### `T.ascend_set_copy_pad_value`

- **签名**：`T.ascend_set_copy_pad_value(value, dtype=None)`
- **核心功能**：Set the padding fill value for subsequent Ascend padded MTE copies. Generates: asc_set_copy_pad_val(value_bits); This call moves no data; it configures the stateful hardware pad value consumed by a following padded GM -> UB/L1 copy. Set it immediately before each padded copy rather than relying on a single set persi...
- **入参**：
  - `value` (int, float, or PrimExpr)：Padding fill value.
  - `dtype` (str, optional, 默认 `None`)：Copy element dtype (one of int8/uint8/int16/uint16/float16/bfloat16/ int32/uint32/float32). Required when ``value`` is a Python literal; for a typed PrimExpr it defaults to ``value.dtype``.
- **返回值**：返回 `None`
- **A5 硬件适配**：UB、L1/CBuf、MTE/DMA。GM→UB 行填充与 ND→NZ 后处理；涉及 PIPE_S pad 寄存器
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.ascend_set_copy_pad_value(0.0, dtype="float32")
```

- **注意事项**：
  - ascend_set_copy_pad_value requires dtype for Python literals; use dtype="float32" or pass a typed PrimExpr.
  - f'ascend_set_copy_pad_value does not support dtype {dtype!r}; supported dtypes are {sorted(_PAD_VALUE_SUPPORTED_DTYPES)}.'

#### `T.ascend_nd2nz_post_copy`

- **签名**：`T.ascend_nd2nz_post_copy(dst, src, rows, cols, full_rows, dst_dtype: str)`
- **核心功能**：Ascend ND->NZ post-copy from UB scratch to L1 with NZ pointer correction.
- **入参**：
  - `dst`：目的 buffer 或区域
  - `src`：源 buffer 或区域
  - `rows`：见签名与源码约束
  - `cols`：见签名与源码约束
  - `full_rows`：见签名与源码约束
  - `dst_dtype` (str)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：UB、L1/CBuf。GM→UB 行填充与 ND→NZ 后处理；涉及 PIPE_S pad 寄存器
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.ascend_nd2nz_post_copy(<dst>, <src>, <rows>, <cols>, <full_rows>, <dst_dtype>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

### 2.5 `frame.py` — AIC / AIV 作用域

#### `T.CubeFrame`

- **签名**：`T.CubeFrame()`
- **核心功能**：Frame for Cube (AIC) execution region.
- **入参**：无显式入参
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：AIC/Cube、Cube、UB。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.CubeFrame()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.Cube`

- **签名**：`T.Cube()`
- **核心功能**：Construct a Cube (AIC) execution region frame.
- **入参**：无显式入参
- **返回值**：返回 `CubeFrame`
- **A5 硬件适配**：AIC/Cube、Cube、UB。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.Cube()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.VectorFrame`

- **签名**：`T.VectorFrame()`
- **核心功能**：Frame for Vector (AIV) execution region. In Mix mode, provides a subblock ID variable via ``as`` binding:: with T.Vector(vector=2) as sid: # sid = asc_get_sub_block_id() ...
- **入参**：无显式入参
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：AIV/Vector、UB。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.VectorFrame()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.Vector`

- **签名**：`T.Vector(vector: int=2)`
- **核心功能**：Construct a Vector (AIV) execution region frame.
- **入参**：
  - `vector` (int, 默认 `2`)：Number of AIV cores (1 or 2). Default 2.
- **返回值**：res : VectorFrame
    The VectorFrame used to denote an AIV execution region.
- **A5 硬件适配**：AIV/Vector。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.Vector()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.SimtVFFrame`

- **签名**：`T.SimtVFFrame()`
- **核心功能**：Frame for SimtVF (SIMT Vector Fixed) region. The C++ frame generates the ``SIMT_VF`` block and its thread-extent bindings. The Python frame additionally maintains the active SIMT context.
- **入参**：无显式入参
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：SimtVF。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.SimtVFFrame()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.SimtVF`

- **签名**：`T.SimtVF(threads: int | list[int] | tuple=128, latency: int=0)`
- **核心功能**：Construct a dedicated SIMT vector-fragment region frame.
- **入参**：
  - `threads` (int | list[int] | tuple, 默认 `128`)：Thread count for the SimtVF frame. A sequence may contain up to three dimensions; omitted dimensions default to one.
  - `latency` (int, 默认 `0`)：Measured latency in cycles. Zero uses the scheduler's latency model.
- **返回值**：res : SimtVFFrame
    The SimtVFFrame used to denote a SIMT vector-fragment region.
- **A5 硬件适配**：SimtVF。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - latency : int Measured latency in cycles.
  - Zero uses the scheduler's latency model.
- **示例**：

```python
T.SimtVF()
```

- **注意事项**：
  - SimtVF requires all thread dimensions > 0
  - SimtVF supports at most 3 thread dimensions
  - f'threads must be int, list, or tuple, got {type(threads)}'

#### `T.SimdVFFrame`

- **签名**：`T.SimdVFFrame()`
- **核心功能**：Frame for SimdVF (SIMD Vector Fragment) region. This is a register-level SIMD region without thread dimensions. The C++ frame generates the ``SIMD_VF`` block and its scope marker.
- **入参**：无显式入参
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：SimdVF。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.SimdVFFrame()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.inside_simdvf`

- **签名**：`T.inside_simdvf()`（内部接口）
- **核心功能**：Return whether the current builder is inside a SimdVF region.
- **入参**：无显式入参
- **返回值**：返回 `bool`
- **A5 硬件适配**：SimdVF。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.inside_simdvf()
```

- **注意事项**：
  - 内部接口：不出现在 T.__all__，不应从生成代码直接调用

#### `T.SimdVF`

- **签名**：`T.SimdVF(latency: int=0)`
- **核心功能**：Construct a SimdVF region for Ascend NPU MicroAPI operations.
- **入参**：
  - `latency` (int, 默认 `0`)：Measured latency in cycles. Zero uses the scheduler's latency model.
- **返回值**：res : SimdVFFrame
    The SimdVFFrame used to denote a register-level SIMD region.
- **A5 硬件适配**：SimdVF。AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域
- **性能建议**：
  - Parameters ---------- latency : int Measured latency in cycles.
  - Zero uses the scheduler's latency model.
- **示例**：

```python
T.SimdVF()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

### 2.6 `gemm_op.py` — Cube GEMM

#### `T.gemm_blockscaled`

- **签名**：`T.gemm_blockscaled(A: BufferLikeType, B: BufferLikeType, C: BufferLikeType, SFA: BufferLikeType, SFB: BufferLikeType, transpose_A: bool=False, transpose_B: bool=False, clear_accum: bool=False, unit_flag_ctrl: int | tirx.PrimExpr | None=None)`
- **核心功能**：Ascend block-scaled MXFP8 GEMM, shadowing the common ``T.gemm_blockscaled``. ``SFA``/``SFB`` are required. For L1 A/B inputs they are the L1 scale-factor buffers, consumed directly by the fused L1 lowering. For L0 A/B inputs they are the MX scale-factor handles of the operand tiles (:func:`tilelang.ascend.language.a...
- **入参**：
  - `A` (BufferLikeType)：见签名与源码约束
  - `B` (BufferLikeType)：见签名与源码约束
  - `C` (BufferLikeType)：见签名与源码约束
  - `SFA` (BufferLikeType)：见签名与源码约束
  - `SFB` (BufferLikeType)：见签名与源码约束
  - `transpose_A` (bool, 默认 `False`)：是否转置 A
  - `transpose_B` (bool, 默认 `False`)：是否转置 B
  - `clear_accum` (bool, 默认 `False`)：是否在累加前清零
  - `unit_flag_ctrl` (int | tirx.PrimExpr | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `tirx.PrimExpr`
- **A5 硬件适配**：L1/CBuf、L0A、L0B。Cube/AIC MAD 单元；L1 输入或 L0A/L0B→L0C 直接 MAD；支持 MX block-scaled
- **性能建议**：
  - For L0 A/B inputs they are the MX scale-factor handles of the operand tiles (:func:`tilelang.ascend.language.alloc_l0a_sf` / :func:`~tilelang.ascend.language.alloc_l0b_sf`), loaded by a preceding ``T.copy(sf_l1, handle)``.
  - Each handle must be allocated for the corresponding data buffer, select the same leading indices, and describe its compact trailing tile (zero origin and matching K).
- **示例**：

```python
T.gemm_blockscaled(<A>, <B>, <C>, <SFA>, <SFB>)
```

- **注意事项**：
  - Each handle must be allocated for the corresponding data buffer, select the same leading indices, and describe its compact trailing tile (zero origin and matching K).

#### `T.gemm`

- **签名**：`T.gemm(A: BufferLikeType, B: BufferLikeType, C: BufferLikeType, transpose_A: bool=False, transpose_B: bool=False, policy: GemmWarpPolicy=GemmWarpPolicy.Square, clear_accum: bool=False, unit_flag_ctrl: int | tirx.PrimExpr | None=None, annotations: dict | None=None)`
- **核心功能**：TileLang GEMM operator, with the Ascend unit-flag hint. Same semantics as the common :func:`tilelang.language.gemm_op.gemm`. ``unit_flag_ctrl`` records the Cube unit-flag control on the tile op, which the Ascend lowering pairs with the following accumulator drain; ``None`` omits the annotation and lowers as 0. On As...
- **入参**：
  - `A` (BufferLikeType)：Input A, input B and output C.
  - `B` (BufferLikeType)：Input A, input B and output C.
  - `C` (BufferLikeType)：Input A, input B and output C.
  - `transpose_A` (bool, 默认 `False`)：Whether to transpose A. Defaults to False.
  - `transpose_B` (bool, 默认 `False`)：Whether to transpose B. Defaults to False.
  - `policy` (GemmWarpPolicy, 默认 `GemmWarpPolicy.Square`)：GEMM warp partition policy.
  - `clear_accum` (bool, 默认 `False`)：Whether to clear the accumulator.
  - `unit_flag_ctrl` (int | tirx.PrimExpr | None, 默认 `None`)：Unit flag control for the instruction. ``None`` omits the annotation and lowers as 0.
  - `annotations` (dict | None, 默认 `None`)：Additional annotations; values in it take precedence over the individual keywords.
- **返回值**：tirx.PrimExpr: A handle to the GEMM operation.
- **A5 硬件适配**：Cube、UB。Cube/AIC MAD 单元；L1 输入或 L0A/L0B→L0C 直接 MAD；支持 MX block-scaled
- **性能建议**：
  - TileLang GEMM operator, with the Ascend unit-flag hint.
  - Same semantics as the common :func:`tilelang.language.gemm_op.gemm`.
- **示例**：

```python
T.gemm(<A>, <B>, <C>)
```

- **注意事项**：
  - Their trailing matrix dimensions must start at zero and describe a compact tile.
  - Copy regions must cover the physical transfer.

### 2.7 `kernel.py` — 核启动与线程域

#### `T.SimtVFContext`

- **签名**：`T.SimtVFContext(thread_vars, thread_extents)`（内部接口）
- **核心功能**：Thread vars and extents of the innermost nested thread scope.
- **入参**：
  - `thread_vars`：见签名与源码约束
  - `thread_extents`：见签名与源码约束
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：1-D AI core 网格启动；线程域由 SimtVF 在核内声明
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.SimtVFContext(<thread_vars>, <thread_extents>)
```

- **注意事项**：
  - 内部接口：不出现在 T.__all__，不应从生成代码直接调用

#### `T.push_simtvf_context`

- **签名**：`T.push_simtvf_context(ctx: SimtVFContext)`（内部接口）
- **核心功能**：Enter a nested thread scope, making its thread vars the current ones.
- **入参**：
  - `ctx` (SimtVFContext)：见签名与源码约束
- **返回值**：无显式返回值（intrinsic / 副作用）
- **A5 硬件适配**：1-D AI core 网格启动；线程域由 SimtVF 在核内声明
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.push_simtvf_context(<ctx>)
```

- **注意事项**：
  - 内部接口：不出现在 T.__all__，不应从生成代码直接调用

#### `T.pop_simtvf_context`

- **签名**：`T.pop_simtvf_context()`（内部接口）
- **核心功能**：Leave the innermost nested thread scope.
- **入参**：无显式入参
- **返回值**：无显式返回值（intrinsic / 副作用）
- **A5 硬件适配**：1-D AI core 网格启动；线程域由 SimtVF 在核内声明
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.pop_simtvf_context()
```

- **注意事项**：
  - 内部接口：不出现在 T.__all__，不应从生成代码直接调用

#### `T.get_thread_binding`

- **签名**：`T.get_thread_binding(dim: int=0)`
- **核心功能**：Returns the thread binding for the given dimension.
- **入参**：
  - `dim` (int, 默认 `0`)：归约/累积维度
- **返回值**：返回 `Var`
- **A5 硬件适配**：1-D AI core 网格启动；线程域由 SimtVF 在核内声明
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.get_thread_binding()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.get_thread_bindings`

- **签名**：`T.get_thread_bindings()`
- **核心功能**：Returns all three thread bindings.
- **入参**：无显式入参
- **返回值**：返回 `list[Var]`
- **A5 硬件适配**：1-D AI core 网格启动；线程域由 SimtVF 在核内声明
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.get_thread_bindings()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.get_thread_extent`

- **签名**：`T.get_thread_extent(dim: int=0)`
- **核心功能**：Returns the thread extent for the given dimension.
- **入参**：
  - `dim` (int, 默认 `0`)：归约/累积维度
- **返回值**：返回 `int`
- **A5 硬件适配**：1-D AI core 网格启动；线程域由 SimtVF 在核内声明
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.get_thread_extent()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.get_thread_extents`

- **签名**：`T.get_thread_extents()`
- **核心功能**：Returns all three thread extents.
- **入参**：无显式入参
- **返回值**：返回 `list[int]`
- **A5 硬件适配**：1-D AI core 网格启动；线程域由 SimtVF 在核内声明
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.get_thread_extents()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.Kernel`

- **签名**：`T.Kernel(*blocks: int | tirx.PrimExpr, prelude: str | None=None)`
- **核心功能**：Construct a kernel launch frame for Ascend: a 1-D grid of AI cores. The grid becomes the NPU core index (``blockIdx.x``). There is no SIMT thread domain at this scope, so this dialect has no ``threads`` parameter: ``with T.Kernel(N) as bx`` yields one program index, and ``bx`` is iterable as ``(bx,)``. Use ``T.SimtV...
- **入参**：
  - `prelude` (str | None, 默认 `None`)：AscendC source injected before the generated kernel, e.g. ``#include`` lines or helper functions.
  - `*blocks` (int | PrimExpr)：Extent of the 1-D core grid. Exactly one dimension is allowed; a multi-dimensional launch is rejected here rather than silently flattened downstream.
- **返回值**：返回 `KernelLaunchFrame`
- **A5 硬件适配**：AIC/Cube、AIV/Vector、SimtVF。1-D AI core 网格启动；线程域由 SimtVF 在核内声明
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.Kernel()
```

- **注意事项**：
  - f'Ascend targets a 1-D core grid: T.Kernel(N) takes exactly one grid extent. Got {len(blocks)}-D: {blocks}.'
  - a multi-dimensional launch is rejected here rather than silently flattened downstream.

#### `T.MixedKernel`

- **签名**：`T.MixedKernel(*blocks: int | tirx.PrimExpr, sids: int=2, prelude: str | None=None)`
- **核心功能**：Construct an Ascend NPU Mixed-kernel launch frame with sub-kernel ID binding.
- **入参**：
  - `sids` (int, 默认 `2`)：Number of active AIV sub-cores (1 or 2). Default is 2. On dav-3510, mixed kernels always launch the physical ``__mix__(1, 2)`` group; ``sids=1`` restricts the vector body to sub-core 0. Binds as ``sid`` via ``asc_get_sub_block_id()`` in generated code.
  - `prelude` (str | None, 默认 `None`)：Import C code injected before the generated kernel.
  - `*blocks`：见签名与源码约束
  - `blocks` (int)：Number of blocks in the 1-D grid (blockIdx.x extent).
- **返回值**：res : KernelLaunchFrame
    The resulting frame providing ``(bx, sid)`` variable bindings.
- **A5 硬件适配**：AIC/Cube、Cube、AIV/Vector、UB、dav-3510。1-D AI core 网格启动；线程域由 SimtVF 在核内声明；On dav-3510, mixed kernels always launch the physical ``__mix__(1, 2)`` group;
- **性能建议**：
  - grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内
- **示例**：

```python
T.MixedKernel(<blocks>)
```

- **注意事项**：
  - T.MixedKernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available.
  - f'T.MixedKernel() only supports 1-D block grid. Got {len(blocks)}-D: {blocks}'
  - f'T.MixedKernel() sids must be 1 or 2. Got {sids}'

### 2.8 `loop.py` — 循环构造

#### `T.unroll`

- **签名**：`T.unroll(start: tirx.PrimExpr, stop: tirx.PrimExpr | None=None, step: tirx.PrimExpr | None=None, *, explicit: bool=False, unroll_factor: int | None=None, annotations: dict[str, Any] | None=None)`
- **核心功能**：The unrolled For statement, with the Ascend unroll-factor pragma. Same semantics as the common :func:`tilelang.language.loop.unroll`. ``unroll_factor`` is lowered as the ``"pragma_unroll_factor"`` annotation, which the Ascend codegen emits as ``#pragma unroll N``.
- **入参**：
  - `start` (tirx.PrimExpr)：Iteration range.
  - `stop` (tirx.PrimExpr | None, 默认 `None`)：Iteration range.
  - `step` (tirx.PrimExpr | None, 默认 `None`)：Iteration range.
  - `explicit` (bool, 默认 `False`)：Whether to explicitly unroll the loop at compile time.
  - `unroll_factor` (int | None, 默认 `None`)：Partial unroll factor, lowered as the ``"pragma_unroll_factor"`` annotation.
  - `annotations` (dict[str, Any] | None, 默认 `None`)：Additional loop annotations; values in it take precedence.
- **返回值**：返回 `frame.ForFrame`
- **A5 硬件适配**：通用循环构造，unroll_factor 由 Ascend codegen 转成 #pragma unroll
- **性能建议**：
  - Same semantics as the common :func:`tilelang.language.loop.unroll`.
- **示例**：

```python
T.unroll(<start>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

### 2.9 `math.py` — 标量数学

#### `T.rsqrt`

- **签名**：`T.rsqrt(x, *, dtype=None)`
- **核心功能**：Compose scalar float32 reciprocal sqrt, preserving existing VF intrinsics. As with the common math wrappers, the input determines the result dtype. The optional ``dtype`` keyword is accepted for the same frontend contract.
- **入参**：
  - `x`：见签名与源码约束
  - `dtype` (默认 `None`)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
- **返回值**：返回 `tirx.PrimExpr`
- **A5 硬件适配**：标量数学；在 VF 内优先复用已有向量 intrinsic
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.rsqrt(<x>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

### 2.10 `mode.py` — Cube / 存储模式控制

#### `T.set_mmad_direction`

- **签名**：`T.set_mmad_direction(direction: str)`
- **核心功能**：Select whether Cube generates results along M or N first. ``direction`` is ``"m"`` or ``"n"``. The setting applies to subsequent GEMMs on the AIC, including block-scaled GEMMs. It changes the traversal within a MAD, independently of the kernel's outer tile-loop order.
- **入参**：
  - `direction` (str)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：AIC/Cube、Cube、UB。Cube MAD 方向、HF32 精度模式、GM 存储模式原子
- **性能建议**：
  - It changes the traversal within a MAD, independently of the kernel's outer tile-loop order.
- **示例**：

```python
T.set_mmad_direction(<direction>)
```

- **注意事项**：
  - f"mmad direction must be 'm' or 'n', got {direction!r}"

#### `T.set_hf32_mode`

- **签名**：`T.set_hf32_mode(mode=None)`
- **核心功能**：Set HF32 mode for Ascend Cube fp32 matmul. Controls whether fp32 inputs to the Cube unit are truncated to HF32 (19-bit mantissa) before multiplication, trading precision for throughput (~2x). Ignored for non-fp32 input types. Generates: AscendC::SetHF32Mode(HF32Mode::DISABLE/ENABLE) and AscendC::SetHF32TransMode(HF3...
- **入参**：
  - `mode` (None | "nearest_zero" | "nearest_even", 默认 `None`)：- None (default): disable HF32, full fp32 precision. - "nearest_zero": enable HF32, round toward zero. - "nearest_even": enable HF32, round to nearest even.
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：Cube、UB。Cube MAD 方向、HF32 精度模式、GM 存储模式原子
- **性能建议**：
  - Controls whether fp32 inputs to the Cube unit are truncated to HF32 (19-bit mantissa) before multiplication, trading precision for throughput (~2x).
- **示例**：

```python
T.set_hf32_mode("nearest_even")   # enable HF32
T.set_hf32_mode(None)             # restore full fp32
```

- **注意事项**：
  - f"hf32 mode must be None, 'nearest_zero', or 'nearest_even', got {mode!r}"

#### `T.set_atomic`

- **签名**：`T.set_atomic(op='add', dtype='float32')`
- **核心功能**：Arm an Ascend hardware store-mode atomic op. Call once before the stores it should affect. With the flag armed, an ordinary L0C->GM (``T.copy``) or UB->GM (``T.dual_copy``) store reduces into GM (``D op= tile``) in hardware instead of overwriting — no vector loop. Clear with :func:`set_atomic_none` to restore plain...
- **入参**：
  - `op` ("add" | "max" | "min", 默认 `'add'`)：The store-mode reduction.
  - `dtype` (str, 默认 `'float32'`)：Accumulate type of the GM destination. One of float32/float16/bfloat16/ int8/int16/int32 (the dav_3510-supported set).
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB、L0C、原子、dav-3510。Cube MAD 方向、HF32 精度模式、GM 存储模式原子；Arm an Ascend hardware store-mode atomic op.
- **性能建议**：
  - With the flag armed, an ordinary L0C->GM (``T.copy``) or UB->GM (``T.dual_copy``) store reduces into GM (``D op= tile``) in hardware instead of overwriting — no vector loop.
- **示例**：

```python
T.set_atomic("add", "float32")   # arm
# ... stores that should accumulate into GM ...
T.set_atomic_none()              # restore
```

- **注意事项**：
  - f'set_atomic op must be one of {_ATOMIC_OPS}, got {op!r}'
  - f'set_atomic dtype must be one of {_ATOMIC_DTYPES}, got {dtype!r}'

#### `T.set_atomic_none`

- **签名**：`T.set_atomic_none()`
- **核心功能**：Clear the Ascend store-mode atomic flag, restoring plain (overwrite) stores. Call once to undo :func:`set_atomic`.
- **入参**：无显式入参
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：原子。Cube MAD 方向、HF32 精度模式、GM 存储模式原子；Clear the Ascend store-mode atomic flag, restoring plain (overwrite) stores.
- **性能建议**：
  - set_atomic 只在需要 GM 累加时开启，存储结束后立即 set_atomic_none
- **示例**：

```python
T.set_atomic_none()
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

### 2.11 `print.py` — 调试打印

#### `T.print`

- **签名**：`T.print(obj: Any=None, msg: str='')`
- **核心功能**：Print a buffer, a primitive expression, or a plain message.
- **入参**：
  - `obj` (Any, 默认 `None`)：A ``tirx.Buffer`` (``local``, ``shared``, ``shared.dyn`` or ``global`` scope), a ``tirx.PrimExpr``, or ``None`` for a message-only print.
  - `msg` (str, 默认 `''`)：Optional message prefix.
- **返回值**：返回 `None`
- **A5 硬件适配**：NPU 执行模型下的调试打印：kernel 每核执行一次，UB/L1 为核内私有
- **性能建议**：
  - 以实测为准；先保证正确性再调优
- **示例**：

```python
T.print()
```

- **注意事项**：
  - f'Unexpected type: {type(obj)}. Supported types are tirx.Buffer, tirx.PrimExpr, and None.'
  - f'T.print: unsupported buffer scope on Ascend: {buffer.scope()}. Copy the buffer to UB (shared) or GM (global) first.'
  - Args: obj: A ``tirx.Buffer`` (``local``, ``shared``, ``shared.dyn`` or ``global`` scope), a ``tirx.PrimExpr``, or ``None`` for a message-only print.

### 2.12 `random.py` — 随机数

#### `T.rng_init`

- **签名**：`T.rng_init(seed, seq=None, off=0, generator='curandStatePhilox4_32_10_t')`
- **核心功能**：Initialize the random number generator state. See :func:`tilelang.cuda.language.random.rng_init`; the only difference is that the default ``seq`` (one stream per lane) is derived from the Ascend thread scope: the SimtVF lane id plus the AI-core index times the lane count.
- **入参**：
  - `seed`：见签名与源码约束
  - `seq` (默认 `None`)：见签名与源码约束
  - `off` (默认 `0`)：见签名与源码约束
  - `generator` (默认 `'curandStatePhilox4_32_10_t'`)：见签名与源码约束
- **返回值**：返回 `tirx.PrimExpr`
- **A5 硬件适配**：SimtVF。随机数状态初始化；默认 sequence id 由 SimtVF lane 与 core 索引推导
- **性能建议**：
  - See :func:`tilelang.cuda.language.random.rng_init`;
- **示例**：

```python
T.rng_init(<seed>)
```

- **注意事项**：
  - the only difference is that the default ``seq`` (one stream per lane) is derived from the Ascend thread scope: the SimtVF lane id plus the AI-core index times the lane count.

### 2.13 `reduce_op.py` — 归约

#### `T.reduce`

- **签名**：`T.reduce(buffer: tirx.Buffer, out: tirx.Buffer, reduce_type: str, dim: int, clear: bool, batch: int=1, annotations: dict | None=None)`
- **核心功能**：Common :func:`tilelang.language.reduce_op.reduce` plus the SimdVF path.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `reduce_type` (str)：见签名与源码约束
  - `dim` (int)：归约/累积维度
  - `clear` (bool)：是否先清零输出
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：SimdVF。Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - Common :func:`tilelang.language.reduce_op.reduce` plus the SimdVF path.
- **示例**：

```python
T.reduce(<buffer>, <out>, <reduce_type>, <dim>, <clear>)
```

- **注意事项**：
  - f'batch must be >= 1, got {batch}'
  - f'Invalid reduce output shape, buffer shape is {buf_shape}, dim is {dim}, output shape is {out_shape}, expected shapes are {expected_shapes_str}'
  - f'reduce_{reduce_type} requires an integer/bool input buffer, got dtype {buf_dtype}'

#### `T.reduce_max`

- **签名**：`T.reduce_max(buffer: tirx.Buffer, out: tirx.Buffer, dim: int=-1, clear: bool=True, batch: int=1, annotations: dict | None=None)`
- **核心功能**：See :func:`tilelang.language.reduce_op.reduce_max`.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `dim` (int, 默认 `-1`)：归约/累积维度
  - `clear` (bool, 默认 `True`)：是否先清零输出
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - See :func:`tilelang.language.reduce_op.reduce_max`.
- **示例**：

```python
T.reduce_max(<buffer>, <out>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.reduce_min`

- **签名**：`T.reduce_min(buffer: tirx.Buffer, out: tirx.Buffer, dim: int=-1, clear: bool=True, batch: int=1, annotations: dict | None=None)`
- **核心功能**：See :func:`tilelang.language.reduce_op.reduce_min`.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `dim` (int, 默认 `-1`)：归约/累积维度
  - `clear` (bool, 默认 `True`)：是否先清零输出
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - See :func:`tilelang.language.reduce_op.reduce_min`.
- **示例**：

```python
T.reduce_min(<buffer>, <out>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.reduce_sum`

- **签名**：`T.reduce_sum(buffer: tirx.Buffer, out: tirx.Buffer, dim: int=-1, clear: bool=True, batch: int=1, annotations: dict | None=None)`
- **核心功能**：See :func:`tilelang.language.reduce_op.reduce_sum`.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `dim` (int, 默认 `-1`)：归约/累积维度
  - `clear` (bool, 默认 `True`)：是否先清零输出
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - See :func:`tilelang.language.reduce_op.reduce_sum`.
- **示例**：

```python
T.reduce_sum(<buffer>, <out>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.reduce_abssum`

- **签名**：`T.reduce_abssum(buffer: tirx.Buffer, out: tirx.Buffer, dim: int=-1, batch: int=1, annotations: dict | None=None)`
- **核心功能**：See :func:`tilelang.language.reduce_op.reduce_abssum`.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `dim` (int, 默认 `-1`)：归约/累积维度
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - See :func:`tilelang.language.reduce_op.reduce_abssum`.
- **示例**：

```python
T.reduce_abssum(<buffer>, <out>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.reduce_absmax`

- **签名**：`T.reduce_absmax(buffer: tirx.Buffer, out: tirx.Buffer, dim: int=-1, clear: bool=True, batch: int=1, annotations: dict | None=None)`
- **核心功能**：See :func:`tilelang.language.reduce_op.reduce_absmax`.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `dim` (int, 默认 `-1`)：归约/累积维度
  - `clear` (bool, 默认 `True`)：是否先清零输出
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - See :func:`tilelang.language.reduce_op.reduce_absmax`.
- **示例**：

```python
T.reduce_absmax(<buffer>, <out>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.reduce_bitand`

- **签名**：`T.reduce_bitand(buffer: tirx.Buffer, out: tirx.Buffer, dim: int=-1, clear: bool=True, batch: int=1, annotations: dict | None=None)`
- **核心功能**：See :func:`tilelang.language.reduce_op.reduce_bitand`.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `dim` (int, 默认 `-1`)：归约/累积维度
  - `clear` (bool, 默认 `True`)：是否先清零输出
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - See :func:`tilelang.language.reduce_op.reduce_bitand`.
- **示例**：

```python
T.reduce_bitand(<buffer>, <out>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.reduce_bitor`

- **签名**：`T.reduce_bitor(buffer: tirx.Buffer, out: tirx.Buffer, dim: int=-1, clear: bool=True, batch: int=1, annotations: dict | None=None)`
- **核心功能**：See :func:`tilelang.language.reduce_op.reduce_bitor`.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `dim` (int, 默认 `-1`)：归约/累积维度
  - `clear` (bool, 默认 `True`)：是否先清零输出
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - See :func:`tilelang.language.reduce_op.reduce_bitor`.
- **示例**：

```python
T.reduce_bitor(<buffer>, <out>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.reduce_bitxor`

- **签名**：`T.reduce_bitxor(buffer: tirx.Buffer, out: tirx.Buffer, dim: int=-1, clear: bool=True, batch: int=1, annotations: dict | None=None)`
- **核心功能**：See :func:`tilelang.language.reduce_op.reduce_bitxor`.
- **入参**：
  - `buffer` (tirx.Buffer)：输入 buffer（UB/共享内存）
  - `out` (tirx.Buffer)：归约输出 buffer
  - `dim` (int, 默认 `-1`)：归约/累积维度
  - `clear` (bool, 默认 `True`)：是否先清零输出
  - `batch` (int, 默认 `1`)：批量归约宽度
  - `annotations` (dict | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 `None`
- **A5 硬件适配**：Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返
- **性能建议**：
  - See :func:`tilelang.language.reduce_op.reduce_bitxor`.
- **示例**：

```python
T.reduce_bitxor(<buffer>, <out>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

### 2.14 `schedule_hint.py` — 调度提示

#### `T.PerCoreTask`

- **签名**：`T.PerCoreTask()`
- **核心功能**：Represent one logical per-core task with guarded candidate sites. AutoSchedule treats the enclosed control flow as one logical task when it analyzes dependencies, estimates cost, and allocates flags. When synchronization is emitted, the same logical wait/set pair is placed next to every candidate access under its or...
- **入参**：无显式入参
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：PIPE。自动调度提示：Stage / Task / PerCoreTask / 冲突假设
- **性能建议**：
  - Explicit candidates also define the latency model.
  - The compiler wraps each unmarked inferred dependency-bearing leaf in an equivalent internal marker before applying the same latency and synchronization model.
- **示例**：

```python
with T.PerCoreTask():
```

- **注意事项**：
  - Every candidate must use the same single hardware pipe and represent the same logical read/write role.
  - For every dynamic invocation and every participating block index, the candidate guards must be mutually exclusive and collectively exhaustive: exactly one candidate must execute on each core.

#### `T.Task`

- **签名**：`T.Task(latency: int | None=None, ii: int | None=None)`
- **核心功能**：Group one or more serial statements into one scheduler task. AutoSchedule creates exactly one TaskNode for the complete marker body. Statements issue in source order without internal synchronization, so each preceding statement contributes its II and only the final statement exposes its latency. All statements must...
- **入参**：
  - `latency` (int | None, 默认 `None`)：见签名与源码约束
  - `ii` (int | None, 默认 `None`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：AIC/Cube、AIV/Vector、UB、L1/CBuf、PIPE、跨核同步。自动调度提示：Stage / Task / PerCoreTask / 冲突假设；split cross-pipe or cross-core sequences into separate tasks.
- **性能建议**：
  - Statements issue in source order without internal synchronization, so each preceding statement contributes its II and only the final statement exposes its latency.
  - ``latency`` is the number of cycles from task issue until its outputs are ready.
- **示例**：

```python
with T.Task():
with T.PerCoreTask():
```

- **注意事项**：
  - f'Task latency must be a non-negative integer, got {latency!r}'
  - f'Task ii must be a positive integer, got {ii!r}'
  - f'Task latency must be greater than or equal to ii, got latency={latency} and ii={ii}'
  - Statements issue in source order without internal synchronization, so each preceding statement contributes its II and only the final statement exposes its latency.

#### `T.Stage`

- **签名**：`T.Stage(stage: int)`
- **核心功能**：Assign a manual software-pipeline stage to enclosed scheduler tasks. If any direct child task or loop in a scheduled child list has a ``T.Stage`` annotation, AutoSchedule treats that child list as manual. One scope may contain several consecutive statements; every scheduler task materialized from those statements re...
- **入参**：
  - `stage` (int)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：SimdVF、SimtVF、PIPE。自动调度提示：Stage / Task / PerCoreTask / 冲突假设
- **性能建议**：
  - Assign a manual software-pipeline stage to enclosed scheduler tasks.
  - ``T.Pipelined(..., num_stages=...)`` controls the automatic buffer-version search bound;
- **示例**：

```python
for i in T.Pipelined(8, annotations={"enable_offset": True}):
```

- **注意事项**：
  - f'Stage must be a non-negative integer, got {stage!r}'
  - ``T.Stage`` must wrap complete scheduler tasks.
  - It must not appear inside ``T.Task``, ``T.SimtVF``, ``T.SimdVF``, an SBlock, or a parallel loop.

#### `T.assume_no_conflict`

- **签名**：`T.assume_no_conflict(a, b=None, *, level=None, cross=None, group=None)`
- **核心功能**：Assert that two buffer regions do not conflict, suppressing a false dependency the auto-scheduler would otherwise conservatively insert. The auto-scheduler's dependency analysis cannot prove that dynamically indexed accesses (e.g. a permutation ``state_cache[perm[idx], ...]``) are disjoint across loop iterations, so...
- **入参**：
  - `a`：first region -- ``buffer[indices...]`` or a bare ``buffer`` (the whole buffer). A region must exactly match the footprint of the access it refers to: use slices for the accessed extents (e.g. ``buf[i, 0:M, 0:N]``), since a fully scalar-indexed ``buf[i, 0, 0]`` denotes only a 1-element region and will not match a wider copy.
  - `b` (默认 `None`)：second region; defaults to ``a`` (a self-dependency).
  - `level` (默认 `None`)：where the non-overlap is asserted. ``None`` applies it to the outermost sequence represented by ``tilelang_root`` and independently to every loop enclosing both operands. ``-1`` applies it only to the outermost sequence. Non-negative values select one common enclosing loop, counted from the outermost (``0`` = outermost, ``1`` = next inner, ...).
  - `cross` (默认 `None`)：``None`` = both same- and cross-iteration are conflict-free (default); ``True`` = only cross-iteration; ``False`` = only same-iteration.
  - `group` (默认 `None`)：string tag pairing two separately-written half-declarations whose regions live in different scopes (so they cannot be written in one call). Must appear exactly twice; the combined hint is attached to the loops enclosing both. Rarely needed.
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：自动调度提示：Stage / Task / PerCoreTask / 冲突假设
- **性能建议**：
  - a permutation ``state_cache[perm[idx], ...]``) are disjoint across loop iterations, so it inserts redundant synchronization.
  - This hint lets the user declare that the regions do not overlap.
- **示例**：

```python
# self-dependency across iterations: A[perm[i]] written each
# iteration lands in a disjoint slot (perm is a permutation), so
# there is no cross-iteration conflict
T.assume_no_conflict(A[perm[i], 0:N], level=0, cross=True)
# two distinct regions that never overlap -- both within the same
# iteration and across iterations (cross=None, the default)
T.assume_no_conflict(A[lo, 0:N], A[hi, 0:N], level=0)
# whole buffer: drop every dependency involving B at every loop
# enclosing this statement (level=None broadcasts to all of them)
T.assume_no_conflict(B)
# group: pair two regions written in separate (sibling) scopes
T.assume_no_conflict(A[wperm[i], 0:N], group="perm")   # in one loop
T.assume_no_conflict(A[rperm[j], 0:N], group="perm")   # in another
```

- **注意事项**：
  - The auto-scheduler's dependency analysis cannot prove that dynamically indexed accesses (e.g.
  - A region must exactly match the footprint of the access it refers to: use slices for the accessed extents (e.g.

#### `T.assume_conflict`

- **签名**：`T.assume_conflict(a, b=None, *, level=None, cross=None, group=None)`
- **核心功能**：Assert that two buffer regions conflict for dependency analysis. This is the conservative counterpart of :func:`assume_no_conflict`. It makes the auto-scheduler treat matching accesses as if they used the same storage, even when the buffers are distinct or their regions are provably disjoint. The usual RAW, WAR, and...
- **入参**：
  - `a`：first region -- ``buffer[indices...]`` or a bare ``buffer`` (the whole buffer). A concrete region must exactly match the footprint of the access it refers to.
  - `b` (默认 `None`)：second region; defaults to ``a`` (a forced self-conflict).
  - `level` (默认 `None`)：where the conflict is asserted. ``None`` applies it to the outermost sequence represented by ``tilelang_root`` and independently to every loop enclosing both operands. ``-1`` applies it only to the outermost sequence. Non-negative values select one common enclosing loop, counted from the outermost.
  - `cross` (默认 `None`)：``None`` = both same- and cross-iteration conflicts (default); ``True`` = only cross-iteration; ``False`` = only same-iteration.
  - `group` (默认 `None`)：string tag pairing two declarations in different scopes. It must appear exactly twice. With ``level=None``, declarations with no common serial loop are still related at the root sequence.
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。自动调度提示：Stage / Task / PerCoreTask / 冲突假设
- **性能建议**：
  - ``None`` applies it to the outermost sequence represented by ``tilelang_root`` and independently to every loop enclosing both operands.
  - Example: >>> # Force ordering between accesses to two independent staging >>> # buffers, including the loop-carried WAR edge needed for reuse.
- **示例**：

```python
# Force ordering between accesses to two independent staging
# buffers, including the loop-carried WAR edge needed for reuse.
T.assume_conflict(a_ub, b_ub, level=0)
# Pair regions declared in sibling loops at their common parent.
T.assume_conflict(a_ub, group="reuse", level=-1)
T.assume_conflict(b_ub, group="reuse", level=-1)
```

- **注意事项**：
  - The usual RAW, WAR, and WAW rules still apply: two read-only accesses do not form a dependency.
  - A concrete region must exactly match the footprint of the access it refers to.

### 2.15 `sync.py` — PIPE / 跨核同步

#### `T.ascend_pipe_barrier`

- **签名**：`T.ascend_pipe_barrier(pipe_t: str)`
- **核心功能**：Insert an AscendC pipeline barrier. Generates: AscendC::PipeBarrier<pipe_t::XXX>();
- **入参**：
  - `pipe_t` (str)：Pipeline type string, e.g., "PIPE_ALL", "PIPE_V", "PIPE_M", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_S".
- **返回值**：返回 `None`
- **A5 硬件适配**：PIPE、MTE/DMA。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁
- **性能建议**：
  - Insert an AscendC pipeline barrier.
  - Parameters ---------- pipe_t : str Pipeline type string, e.g., "PIPE_ALL", "PIPE_V", "PIPE_M", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_S".
- **示例**：

```python
T.ascend_pipe_barrier("PIPE_ALL")
```

- **注意事项**：
  - f'pipe_t must be a string, got {type(pipe_t)}'

#### `T.ascend_set_flag`

- **签名**：`T.ascend_set_flag(hard_event: str, event_id)`
- **核心功能**：Insert an AscendC SetFlag for pipeline synchronization. Generates: AscendC::SetFlag<AscendC::HardEvent::XXX>(eventID);
- **入参**：
  - `hard_event` (str)：HardEvent enum name, e.g., "MTE2_MTE1", "S_MTE3", "V_MTE2".
  - `event_id` (int or PrimExpr)：Event ID for synchronization.
- **返回值**：返回 `None`
- **A5 硬件适配**：PIPE、MTE/DMA。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁
- **性能建议**：
  - Insert an AscendC SetFlag for pipeline synchronization.
- **示例**：

```python
T.ascend_set_flag("S_MTE3", event_id)
```

- **注意事项**：
  - f'hard_event must be a string, got {type(hard_event)}'

#### `T.ascend_wait_flag`

- **签名**：`T.ascend_wait_flag(hard_event: str, event_id)`
- **核心功能**：Insert an AscendC WaitFlag for pipeline synchronization. Generates: AscendC::WaitFlag<AscendC::HardEvent::XXX>(eventID);
- **入参**：
  - `hard_event` (str)：HardEvent enum name, e.g., "MTE2_MTE1", "S_MTE3", "V_MTE2".
  - `event_id` (int or PrimExpr)：Event ID for synchronization.
- **返回值**：返回 `None`
- **A5 硬件适配**：PIPE、MTE/DMA。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁
- **性能建议**：
  - Insert an AscendC WaitFlag for pipeline synchronization.
- **示例**：

```python
T.ascend_wait_flag("S_MTE3", event_id)
```

- **注意事项**：
  - f'hard_event must be a string, got {type(hard_event)}'

#### `T.ascend_sync_inter_arrive`

- **签名**：`T.ascend_sync_inter_arrive(pipe: str, flag_id)`
- **核心功能**：Signal an Ascend inter-core synchronization event. Generates: AscendC::CrossCoreSetFlag<0, pipe>(flag_id);
- **入参**：
  - `pipe` (str)：Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
  - `flag_id` (int or PrimExpr)：Inter-core synchronization flag ID.
- **返回值**：返回 `None`
- **A5 硬件适配**：PIPE、MTE/DMA。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁
- **性能建议**：
  - Parameters ---------- pipe : str Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
- **示例**：

```python
T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
```

- **注意事项**：
  - 手写同步必须整体照抄一个完整示例，禁止混搭半自动/半手动片段

#### `T.ascend_sync_inter_wait`

- **签名**：`T.ascend_sync_inter_wait(pipe: str, flag_id)`
- **核心功能**：Wait for an Ascend inter-core synchronization event. Generates: AscendC::CrossCoreWaitFlag<0, pipe>(flag_id);
- **入参**：
  - `pipe` (str)：Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
  - `flag_id` (int or PrimExpr)：Inter-core synchronization flag ID.
- **返回值**：返回 `None`
- **A5 硬件适配**：PIPE、MTE/DMA。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁
- **性能建议**：
  - Parameters ---------- pipe : str Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
- **示例**：

```python
T.ascend_sync_inter_wait("PIPE_MTE3", flag_id)
```

- **注意事项**：
  - 手写同步必须整体照抄一个完整示例，禁止混搭半自动/半手动片段

#### `T.ascend_threadfence`

- **签名**：`T.ascend_threadfence()`
- **核心功能**：Insert an AscendC thread fence. Generates: asc_threadfence();
- **入参**：无显式入参
- **返回值**：返回 `None`
- **A5 硬件适配**：AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁
- **性能建议**：
  - 优先自动调度；手写 flag 必须成对且覆盖所有分支
- **示例**：

```python
T.ascend_threadfence()
```

- **注意事项**：
  - 手写同步必须整体照抄一个完整示例，禁止混搭半自动/半手动片段

#### `T.ascend_cross_core_set_flag`

- **签名**：`T.ascend_cross_core_set_flag(mode_id: int, pipe: str, flag_id)`
- **核心功能**：Insert an AscendC CrossCoreSetFlag for cross-core synchronization. Generates: AscendC::CrossCoreSetFlag<modeId, pipe>(flagId);
- **入参**：
  - `mode_id` (int)：Cross-core sync mode: 0, 1, 2, or 4.
  - `pipe` (str)：Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
  - `flag_id` (int or PrimExpr)：Cross-core synchronization flag ID.
- **返回值**：返回 `None`
- **A5 硬件适配**：PIPE、MTE/DMA、跨核同步。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁；Insert an AscendC CrossCoreSetFlag for cross-core synchronization.
- **性能建议**：
  - pipe : str Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
- **示例**：

```python
T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 0x8)
```

- **注意事项**：
  - f'pipe must be a string, got {type(pipe)}'

#### `T.ascend_cross_core_wait_flag`

- **签名**：`T.ascend_cross_core_wait_flag(mode_id: int, pipe: str, flag_id)`
- **核心功能**：Insert an AscendC CrossCoreWaitFlag for cross-core synchronization. Generates: AscendC::CrossCoreWaitFlag<modeId, pipe>(flagId);
- **入参**：
  - `mode_id` (int)：Cross-core sync mode: 0, 1, 2, or 4.
  - `pipe` (str)：Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
  - `flag_id` (int or PrimExpr)：Cross-core synchronization flag ID.
- **返回值**：返回 `None`
- **A5 硬件适配**：PIPE、MTE/DMA、跨核同步。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁；Insert an AscendC CrossCoreWaitFlag for cross-core synchronization.
- **性能建议**：
  - pipe : str Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
- **示例**：

```python
T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 0x8)
```

- **注意事项**：
  - f'pipe must be a string, got {type(pipe)}'

#### `T.ascend_get_buf`

- **签名**：`T.ascend_get_buf(pipe, buf_id, mode=False)`
- **核心功能**：Acquire a pipe buffer for double-buffer management. Generates: asc_lock(pipe, buf_id, mode);
- **入参**：
  - `pipe` (str)：Pipeline type, e.g., "PIPE_MTE1", "PIPE_M".
  - `buf_id` (int or PrimExpr)：Buffer slot index, typically (sk & 1) for double-buffering.
  - `mode` (bool, 默认 `False`)：Optional mode flag (default False).
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB、PIPE、MTE/DMA。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁
- **性能建议**：
  - Parameters ---------- pipe : str Pipeline type, e.g., "PIPE_MTE1", "PIPE_M".
- **示例**：

```python
T.ascend_get_buf("PIPE_MTE1", sk & 1)
```

- **注意事项**：
  - 手写同步必须整体照抄一个完整示例，禁止混搭半自动/半手动片段

#### `T.ascend_rls_buf`

- **签名**：`T.ascend_rls_buf(pipe, buf_id, mode=False)`
- **核心功能**：Release a pipe buffer for double-buffer management. Generates: asc_unlock(pipe, buf_id, mode);
- **入参**：
  - `pipe` (str)：Pipeline type, e.g., "PIPE_MTE1", "PIPE_M".
  - `buf_id` (int or PrimExpr)：Buffer slot index, typically (sk & 1) for double-buffering.
  - `mode` (bool, 默认 `False`)：Optional mode flag (default False).
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB、PIPE、MTE/DMA。AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁
- **性能建议**：
  - Parameters ---------- pipe : str Pipeline type, e.g., "PIPE_MTE1", "PIPE_M".
- **示例**：

```python
T.ascend_rls_buf("PIPE_M", sk & 1)
```

- **注意事项**：
  - 手写同步必须整体照抄一个完整示例，禁止混搭半自动/半手动片段

### 2.16 `tile_schedule.py` — 持久化 tile 调度

#### `T.AscendBaseTileScheduler`

- **签名**：`T.AscendBaseTileScheduler(*, block_m, block_n, num_cores, shape_m, shape_n, group_m: int=4, xor_n: bool=True, snake: bool=True, stateful: bool=True, name: str | None=None)`
- **核心功能**：Base for persistent GEMM block schedulers (GROUP_M swizzle + N-first snake). Holds everything shared across GEMM variants: the grid geometry, the stateless ``_swizzle`` / ``coord`` decode, the persistent-block-index advance, and the ``init`` / ``valid`` / ``next_block`` loop protocol. Concrete variants subclass this...
- **入参**：
  - `block_m` (int)：Tile sizes along M and N.
  - `block_n` (int)：Tile sizes along M and N.
  - `num_cores` (int | PrimExpr)：Persistent stride = number of resident cores (grid dim).
  - `shape_m` (int | PrimExpr)：Problem M and N.
  - `shape_n` (int | PrimExpr)：Problem M and N.
  - `group_m` (int, 默认 `4`)：Swizzle panel height along M (4 or 8).
  - `xor_n` (bool, 默认 `True`)：XOR the in-panel n-index with the m-index (default True).
  - `snake` (bool, 默认 `True`)：Reverse the N-group order on odd superrows (default True).
  - `stateful` (bool, 默认 `True`)：If True (default) allocate state and enable ``init`` / ``valid`` / ``next_block``. If False expose only the store-free ``coord`` decode.
  - `name` (str | None, 默认 `None`)：Optional state-buffer name prefix. The default ``None`` uses generic auto-generated names.
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：UB。持久化 GEMM tile 遍历（GROUP_M swizzle / N-first snake）
- **性能建议**：
  - The block index handed out on iteration ``current_iter`` is ``current_iter * num_cores + core_id``, mapped to a tile by ``_swizzle``.
  - Usage:: sched = T.AscendTileScheduler(block_m=BM, block_n=BN, num_cores=NUM_CORES, shape_m=M, shape_n=N) sched.init(bx) while sched.valid(): m_tile, n_tile = sched.m_idx[0], sched.n_idx[0] # ...
- **示例**：

```python
T.AscendBaseTileScheduler(<block_m>, <block_n>, <num_cores>, <shape_m>, <shape_n>)
```

- **注意事项**：
  - If False expose only the store-free ``coord`` decode.

#### `T.AscendTileScheduler`

- **签名**：`T.AscendTileScheduler(*, block_m, block_n, num_cores, shape_m, shape_n, group_m: int=4, xor_n: bool=True, snake: bool=True, stateful: bool=True, name: str | None=None)`
- **核心功能**：Single flat M x N grid, GROUP_M swizzle + N-first snake; the last ``< group_m`` M-rows fall back to row-major. This is the plain (non-grouped, non-batched) GEMM scheduler.
- **入参**：
  - `block_m` (int)：Tile sizes along M and N.
  - `block_n` (int)：Tile sizes along M and N.
  - `num_cores` (int | PrimExpr)：Persistent stride = number of resident cores (grid dim).
  - `shape_m` (int | PrimExpr)：Problem M and N.
  - `shape_n` (int | PrimExpr)：Problem M and N.
  - `group_m` (int, 默认 `4`)：Swizzle panel height along M (4 or 8).
  - `xor_n` (bool, 默认 `True`)：XOR the in-panel n-index with the m-index (default True).
  - `snake` (bool, 默认 `True`)：Reverse the N-group order on odd superrows (default True).
  - `stateful` (bool, 默认 `True`)：If True (default) allocate state and enable ``init`` / ``valid`` / ``next_block``. If False expose only the store-free ``coord`` decode.
  - `name` (str | None, 默认 `None`)：Optional state-buffer name prefix. The default ``None`` uses generic auto-generated names.
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：持久化 GEMM tile 遍历（GROUP_M swizzle / N-first snake）
- **性能建议**：
  - 按 core 数选择 GROUP_M 分组；用实测 device time 选择 swizzle 顺序
- **示例**：

```python
T.AscendTileScheduler(<block_m>, <block_n>, <num_cores>, <shape_m>, <shape_n>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.AscendBatchedTileScheduler`

- **签名**：`T.AscendBatchedTileScheduler(*, block_m, block_n, num_cores, shape_m, shape_n, num_groups: int, group_m: int=4, xor_n: bool=True, snake: bool=True, stateful: bool=True, name: str | None=None)`
- **核心功能**：Standard batched matmul: the Normal grid replicated ``num_groups`` (= batch) times. ``get_batch_idx()`` exposes the batch of the current block. Extra parameter ``num_groups`` (= batch count) beyond the base scheduler.
- **入参**：
  - `block_m` (int)：Tile sizes along M and N.
  - `block_n` (int)：Tile sizes along M and N.
  - `num_cores` (int | PrimExpr)：Persistent stride = number of resident cores (grid dim).
  - `shape_m` (int | PrimExpr)：Problem M and N.
  - `shape_n` (int | PrimExpr)：Problem M and N.
  - `num_groups` (int)：见签名与源码约束
  - `group_m` (int, 默认 `4`)：Swizzle panel height along M (4 or 8).
  - `xor_n` (bool, 默认 `True`)：XOR the in-panel n-index with the m-index (default True).
  - `snake` (bool, 默认 `True`)：Reverse the N-group order on odd superrows (default True).
  - `stateful` (bool, 默认 `True`)：If True (default) allocate state and enable ``init`` / ``valid`` / ``next_block``. If False expose only the store-free ``coord`` decode.
  - `name` (str | None, 默认 `None`)：Optional state-buffer name prefix. The default ``None`` uses generic auto-generated names.
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：持久化 GEMM tile 遍历（GROUP_M swizzle / N-first snake）
- **性能建议**：
  - 按 core 数选择 GROUP_M 分组；用实测 device time 选择 swizzle 顺序
- **示例**：

```python
T.AscendBatchedTileScheduler(<block_m>, <block_n>, <num_cores>, <shape_m>, <shape_n>, <num_groups>)
```

- **注意事项**：
  - 无额外约束；以源码与静态检查为准

#### `T.AscendMGroupedTileScheduler`

- **签名**：`T.AscendMGroupedTileScheduler(*, block_m, block_n, num_cores, shape_m, shape_n, grouped_layout, num_groups: int, alignment: int=256, group_m: int=4, xor_n: bool=True, snake: bool=True, stateful: bool=True, name: str | None=None)`
- **核心功能**：MoE m-grouped "contiguous" layout. ``grouped_layout`` is the prefix-sum-of-rows array (length ``num_groups``); group ``g``'s valid rows are ``[align(psum[g-1], alignment), psum[g])`` in the global M space (group 0 starts at 0). ``get_group_idx()`` exposes the current group for per-group B/SF offsets. A/D must alloca...
- **入参**：
  - `block_m` (int)：Tile sizes along M and N.
  - `block_n` (int)：Tile sizes along M and N.
  - `num_cores` (int | PrimExpr)：Persistent stride = number of resident cores (grid dim).
  - `shape_m` (int | PrimExpr)：Problem M and N.
  - `shape_n` (int | PrimExpr)：Problem M and N.
  - `grouped_layout`：见签名与源码约束
  - `num_groups` (int)：见签名与源码约束
  - `alignment` (int, 默认 `256`)：见签名与源码约束
  - `group_m` (int, 默认 `4`)：Swizzle panel height along M (4 or 8).
  - `xor_n` (bool, 默认 `True`)：XOR the in-panel n-index with the m-index (default True).
  - `snake` (bool, 默认 `True`)：Reverse the N-group order on odd superrows (default True).
  - `stateful` (bool, 默认 `True`)：If True (default) allocate state and enable ``init`` / ``valid`` / ``next_block``. If False expose only the store-free ``coord`` decode.
  - `name` (str | None, 默认 `None`)：Optional state-buffer name prefix. The default ``None`` uses generic auto-generated names.
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：持久化 GEMM tile 遍历（GROUP_M swizzle / N-first snake）
- **性能建议**：
  - group ``g``'s valid rows are ``[align(psum[g-1], alignment), psum[g])`` in the global M space (group 0 starts at 0).
  - A/D must allocate all aligned rows, and physical M must be a multiple of alignment.
- **示例**：

```python
T.AscendMGroupedTileScheduler(<block_m>, <block_n>, <num_cores>, <shape_m>, <shape_n>, <grouped_layout>, <num_groups>)
```

- **注意事项**：
  - A/D must allocate all aligned rows, and physical M must be a multiple of alignment.

#### `T.AscendKGroupedTileScheduler`

- **签名**：`T.AscendKGroupedTileScheduler(*, block_m, block_n, num_cores, shape_m, shape_n, shape_k, grouped_layout, num_groups: int, alignment: int=256, group_m: int=4, xor_n: bool=True, snake: bool=True, stateful: bool=True, name: str | None=None)`
- **核心功能**：MoE k-grouped. All groups share one M x N grid but differ in K length; ``grouped_layout[i]`` is the cumulative end-K of group ``i``. ``get_group_idx()``, ``get_k_idx_base()`` / ``get_sf_idx_base()`` give the group and its A/B and SF K offsets; ``get_shape_k()`` is the current group's K length. Extra parameters beyon...
- **入参**：
  - `block_m` (int)：Tile sizes along M and N.
  - `block_n` (int)：Tile sizes along M and N.
  - `num_cores` (int | PrimExpr)：Persistent stride = number of resident cores (grid dim).
  - `shape_m` (int | PrimExpr)：Problem M and N.
  - `shape_n` (int | PrimExpr)：Problem M and N.
  - `shape_k`：见签名与源码约束
  - `grouped_layout`：见签名与源码约束
  - `num_groups` (int)：见签名与源码约束
  - `alignment` (int, 默认 `256`)：见签名与源码约束
  - `group_m` (int, 默认 `4`)：Swizzle panel height along M (4 or 8).
  - `xor_n` (bool, 默认 `True`)：XOR the in-panel n-index with the m-index (default True).
  - `snake` (bool, 默认 `True`)：Reverse the N-group order on odd superrows (default True).
  - `stateful` (bool, 默认 `True`)：If True (default) allocate state and enable ``init`` / ``valid`` / ``next_block``. If False expose only the store-free ``coord`` decode.
  - `name` (str | None, 默认 `None`)：Optional state-buffer name prefix. The default ``None`` uses generic auto-generated names.
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：持久化 GEMM tile 遍历（GROUP_M swizzle / N-first snake）
- **性能建议**：
  - only the K-range contract uses it), ``alignment`` (group-start K alignment, default 256).
- **示例**：

```python
T.AscendKGroupedTileScheduler(<block_m>, <block_n>, <num_cores>, <shape_m>, <shape_n>, <shape_k>, <grouped_layout>, <num_groups>)
```

- **注意事项**：
  - only the K-range contract uses it), ``alignment`` (group-start K alignment, default 256).

## 3. `T.simd.*` SIMD MicroAPI

### 3.1 分配与辅助

#### `T.simd.SimdPair`

- **签名**：`T.simd.SimdPair(pair, dtype=None)`
- **核心功能**：Wraps a two-result SIMD intrinsic and preserves each result dtype. ``a, b = ...`` emits two ``pair_get`` calls against the pair. The pair- producing op (e.g. ``vintlv``, ``vld2``, or post-update ``vld``) is bound once at its call site by the frontend, so both ``pair_get`` calls reference a single bound variable rath...
- **入参**：
  - `pair`：见签名与源码约束
  - `dtype` (默认 `None`)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
- **返回值**：构造并返回该类实例
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.SimdPair(<pair>)
```

- **注意事项**：
  - f'SimdPair `dtype` must be a tuple or list, got {type(dtype).__name__}'
  - f'SimdPair requires exactly two result dtypes, got {len(dtype)}'

#### `T.simd.alloc_var`

- **签名**：`T.simd.alloc_var(dtype: DType)`
- **核心功能**：Allocate a single mutable SIMD register variable (return-value style). Uses ``local.var`` scope and behaves as one vector register value. For an addressable array of registers (``v[i]``), use :func:`alloc_local`.
- **入参**：
  - `dtype` (DType)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.alloc_var(<dtype>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.alloc_local`

- **签名**：`T.simd.alloc_local(shape: ShapeType, dtype: DType)`
- **核心功能**：Allocate an addressable array of mutable SIMD register variables. Uses ``local`` scope so each element is an individually addressable register, allowing indexed access like ``v[i]``:: v = T.simd.alloc_local(4, "float32") for i in T.Unroll(4, explicit=True): v[i] = vld(s_ub[i * VL])
- **入参**：
  - `shape` (ShapeType)：buffer 形状（静态或符号维度）；需满足对齐与容量约束
  - `dtype` (DType)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.alloc_local(<shape>, <dtype>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.make_ubuf_ptr`

- **签名**：`T.simd.make_ubuf_ptr(buf_access, dtype)`
- **核心功能**：Allocate a mutable UB pointer for post-update :func:`vld` / :func:`vsstb` calls. ``dtype`` declares the pointee element type used by :func:`vld` and checked against the source vector by :func:`vsstb`. It is stored in the enclosing block's IR annotations; the mutable carrier remains a ``handle`` buffer. The pointer i...
- **入参**：
  - `buf_access`：见签名与源码约束
  - `dtype`：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.make_ubuf_ptr(<buf_access>, <dtype>)
```

- **注意事项**：
  - make_ubuf_ptr requires a scalar numeric element dtype
  - make_ubuf_ptr expects a buffer access or handle expression

#### `T.simd.pair_get`

- **签名**：`T.simd.pair_get(pair, index)`
- **核心功能**：Extract element from a pair: reg = pair_get(pair, 0) or pair_get(pair, 1).
- **入参**：
  - `pair`：见签名与源码约束
  - `index`：逐 lane 索引向量（无符号整型）
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pair_get(<pair>, <index>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.mem_bar`

- **签名**：`T.simd.mem_bar(mem_type)`
- **核心功能**：Memory barrier: mem_bar(mem_type), e.g. mem_bar(VST_VLD).
- **入参**：
  - `mem_type`：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.mem_bar(<mem_type>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

### 3.2 谓词与掩码

#### `T.simd.pset`

- **签名**：`T.simd.pset(elem_width: int, dist: str='PAT_ALL')`
- **核心功能**：Create a predicate mask: pset_bXX(dist). Returns a vector_bool (boolx256). dist: "PAT_ALL", "PAT_VL1".."PAT_VL128", "PAT_M3", "PAT_M4", "PAT_H", "PAT_Q", etc.
- **入参**：
  - `elem_width` (int)：见签名与源码约束
  - `dist` (str, 默认 `'PAT_ALL'`)：见签名与源码约束
- **返回值**：Returns a vector_bool (boolx256).
- **A5 硬件适配**：L1/CBuf。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pset(<elem_width>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.pge`

- **签名**：`T.simd.pge(elem_width: int, dist: str='PAT_ALL')`
- **核心功能**：Create a predicate mask from pge_bXX(dist).
- **入参**：
  - `elem_width` (int)：见签名与源码约束
  - `dist` (str, 默认 `'PAT_ALL'`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pge(<elem_width>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.pand`

- **签名**：`T.simd.pand(src0, src1, mask)`
- **核心功能**：谓词按位与（谓词寄存器逐 lane 合并）
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask`：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pand(<src0>, <src1>, <mask>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.por`

- **签名**：`T.simd.por(src0, src1, mask)`
- **核心功能**：谓词按位或
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask`：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.por(<src0>, <src1>, <mask>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.pxor`

- **签名**：`T.simd.pxor(src0, src1, mask)`
- **核心功能**：谓词按位异或
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask`：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pxor(<src0>, <src1>, <mask>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.pnot`

- **签名**：`T.simd.pnot(src, mask)`
- **核心功能**：谓词按位取反
- **入参**：
  - `src`：源 buffer 或区域
  - `mask`：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pnot(<src>, <mask>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.psel`

- **签名**：`T.simd.psel(src0, src1, mask)`
- **核心功能**：按谓词选择两个谓词源
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask`：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.psel(<src0>, <src1>, <mask>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.ppack`

- **签名**：`T.simd.ppack(src, part=0)`
- **核心功能**：Predicate pack 2:1 (zeroing): dst = ppack(src, LOWER/HIGHER).
- **入参**：
  - `src`：源 buffer 或区域
  - `part` (默认 `0`)：LOWER/HIGHER 半区选择
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.ppack(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.punpack`

- **签名**：`T.simd.punpack(src, part=0)`
- **核心功能**：Predicate unpack 1:2 (zeroing): dst = punpack(src, LOWER/HIGHER).
- **入参**：
  - `src`：源 buffer 或区域
  - `part` (默认 `0`)：LOWER/HIGHER 半区选择
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.punpack(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.pintlv`

- **签名**：`T.simd.pintlv(src0, src1, width=32)`
- **核心功能**：Predicate interleave -> pair of predicates (b8/b16/b32).
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `width` (默认 `32`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pintlv(<src0>, <src1>)
```

- **注意事项**：
  - pintlv requires an active IRBuilder to bind the pair

#### `T.simd.pdintlv`

- **签名**：`T.simd.pdintlv(src0, src1, width=32)`
- **核心功能**：Predicate deinterleave -> pair of predicates (b8/b16/b32).
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `width` (默认 `32`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pdintlv(<src0>, <src1>)
```

- **注意事项**：
  - pdintlv requires an active IRBuilder to bind the pair

#### `T.simd.pld`

- **签名**：`T.simd.pld(addr, dist='NORM')`
- **核心功能**：Load a predicate from UB. Express address offsets in ``addr``.
- **入参**：
  - `addr`：见签名与源码约束
  - `dist` (默认 `'NORM'`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pld(<addr>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.pst`

- **签名**：`T.simd.pst(addr, src, dist='NORM')`
- **核心功能**：Store a predicate to UB. Express address offsets in ``addr``.
- **入参**：
  - `addr`：见签名与源码约束
  - `src`：源 buffer 或区域
  - `dist` (默认 `'NORM'`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.pst(<addr>, <src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

### 3.3 访存与索引

#### `T.simd.vld`

- **签名**：`T.simd.vld(addr, dist='NORM', *, post_inc=None)`
- **核心功能**：Vector load. Returns a typed vector register. addr can be a BufferLoad auto-wrapped as tl.access_ptr. Express address offsets in ``addr``. With ``post_inc=step``, load through a mutable :func:`make_ubuf_ptr` handle and return ``(vector, advanced_pointer)``. Assign the second result back to the handle. ``step`` is a...
- **入参**：
  - `addr`：见签名与源码约束
  - `dist` (默认 `'NORM'`)：见签名与源码约束
  - `post_inc` (默认 `None`)：见签名与源码约束
- **返回值**：Returns a typed vector register.
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vld(<addr>)
```

- **注意事项**：
  - vld post_inc requires a mutable pointer from make_ubuf_ptr
  - vld post_inc requires an 8/16/32-bit pointer element dtype
  - f'Unsupported vld post-update distribution: {dist!r}'
  - The dtype must be 8/16/32-bit and match the distribution.

#### `T.simd.vld2`

- **签名**：`T.simd.vld2(addr, dist='DINTLV_B16', off=None)`
- **核心功能**：Dual-dest vector load: ``a, b = vld2(x_ub[i, col], dist="DINTLV_B8")``. Supported dists: - ``DINTLV_B8``: load 512xu8/fp8 -> two 256-lane regs (even/odd bytes) - ``DINTLV_B16``: load 256xbf16/u16 -> two 128-lane regs - ``DINTLV_B32``: load 128xf32/u32 -> two 64-lane regs The access_ptr footprint is always 2x the sin...
- **入参**：
  - `addr`：见签名与源码约束
  - `dist` (默认 `'DINTLV_B16'`)：见签名与源码约束
  - `off` (默认 `None`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vld2(<addr>)
```

- **注意事项**：
  - f'vld2 only supports dist in {_VLD2_DISTS}, got {dist!r}'
  - vld2 requires an active IRBuilder to bind the opaque load

#### `T.simd.vsts`

- **签名**：`T.simd.vsts(addr, src, mask=None, dist='NORM_B32', extent=None)`
- **核心功能**：Vector store to ``addr``. addr can be a BufferLoad auto-wrapped as tl.access_ptr. Express address offsets in ``addr``. Optional ``extent`` overrides the default access_ptr footprint (e.g. 8 for a dense PAT_VL8 NORM_B16 recip pack).
- **入参**：
  - `addr`：见签名与源码约束
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `dist` (默认 `'NORM_B32'`)：见签名与源码约束
  - `extent` (默认 `None`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vsts(<addr>, <src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vsstb`

- **签名**：`T.simd.vsstb(src, base, stride, mask=None, update=False)`
- **核心功能**：Scatter-store 32B blocks with an optional POST_UPDATE pointer. Passing a regular buffer access performs a store and returns ``void``. Set ``update=True`` with the mutable handle returned by :func:`make_ubuf_ptr` to enable POST_UPDATE and return the advanced pointer, which should be assigned back to the same handle::...
- **入参**：
  - `src`：源 buffer 或区域
  - `base`：基地址 buffer/指针
  - `stride`：见签名与源码约束
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `update` (默认 `False`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vsstb(<src>, <base>, <stride>)
```

- **注意事项**：
  - vsstb update=True base must be a pointer from make_ubuf_ptr
  - vsstb source element dtype must match the pointer dtype

#### `T.simd.vgatherb`

- **签名**：`T.simd.vgatherb(base, index, mask=None)`
- **核心功能**：Gather 32B blocks from base using vector_u32 block offsets. Returns a vector.
- **入参**：
  - `base`：基地址 buffer/指针
  - `index`：逐 lane 索引向量（无符号整型）
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：Returns a vector.
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vgatherb(<base>, <index>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vgather2`

- **签名**：`T.simd.vgather2(base, index, mask=None)`
- **核心功能**：Gather elements from base using per-lane offsets.
- **入参**：
  - `base`：基地址 buffer/指针
  - `index`：逐 lane 索引向量（无符号整型）
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vgather2(<base>, <index>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vscatter`

- **签名**：`T.simd.vscatter(src, base, index, mask=None)`
- **核心功能**：Scatter-store: base[index[lane]] = src[lane]. Write-side counterpart to vgatherb/vgather2. ``index`` is a vector register of per-lane element offsets (uint32 for 32-bit elements, uint16 for 8/16-bit).
- **入参**：
  - `src`：源 buffer 或区域
  - `base`：基地址 buffer/指针
  - `index`：逐 lane 索引向量（无符号整型）
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vscatter(<src>, <base>, <index>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

### 3.4 向量计算

#### `T.simd.vadd`

- **签名**：`T.simd.vadd(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 加法
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vadd(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vaddc`

- **签名**：`T.simd.vaddc(src0, src1, mask=None)`
- **核心功能**：Add int32/uint32 vectors without carry-in and return ``(carry, result)``. ``carry`` is a ``boolx256`` predicate register and ``result`` has the same dtype as the inputs. The underlying Ascend ``vaddc`` instruction only supports full-register ``int32x64`` and ``uint32x64`` operands.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vaddc(<src0>, <src1>)
```

- **注意事项**：
  - vaddc requires an active IRBuilder to bind the pair
  - The underlying Ascend ``vaddc`` instruction only supports full-register ``int32x64`` and ``uint32x64`` operands.

#### `T.simd.vsubc`

- **签名**：`T.simd.vsubc(src0, src1, mask=None)`
- **核心功能**：Subtract int32/uint32 without carry-in and return ``(carry, result)``. ``carry`` is 1 where the subtraction completes without borrow.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vsubc(<src0>, <src1>)
```

- **注意事项**：
  - vsubc requires an active IRBuilder to bind the pair

#### `T.simd.vaddcs`

- **签名**：`T.simd.vaddcs(src0, src1, carrysrcp, mask=None)`
- **核心功能**：Add int32/uint32 with carry-in predicate and return ``(carry, result)``.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `carrysrcp`：见签名与源码约束
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vaddcs(<src0>, <src1>, <carrysrcp>)
```

- **注意事项**：
  - vaddcs requires an active IRBuilder to bind the pair

#### `T.simd.vsubcs`

- **签名**：`T.simd.vsubcs(src0, src1, carrysrcp, mask=None)`
- **核心功能**：Subtract int32/uint32 with carry-in predicate and return ``(carry, result)``.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `carrysrcp`：见签名与源码约束
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vsubcs(<src0>, <src1>, <carrysrcp>)
```

- **注意事项**：
  - vsubcs requires an active IRBuilder to bind the pair

#### `T.simd.vmull`

- **签名**：`T.simd.vmull(src0, src1, mask=None)`
- **核心功能**：Widening 32x32->64 multiply returning ``(lo, hi)`` (int32/uint32).
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmull(<src0>, <src1>)
```

- **注意事项**：
  - vmull requires an active IRBuilder to bind the pair

#### `T.simd.vsub`

- **签名**：`T.simd.vsub(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 减法
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vsub(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vmul`

- **签名**：`T.simd.vmul(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 乘法
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmul(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vmula`

- **签名**：`T.simd.vmula(dst, src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Fused multiply-add: dst = dst + src0 * src1.
- **入参**：
  - `dst`：目的 buffer 或区域
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmula(<dst>, <src0>, <src1>)
```

- **注意事项**：
  - f'vmula: `dst` must be a buffer access (e.g. c[0]) allocated with T.simd.alloc_local()/alloc_var(), not an immutable register value. Got {type(dst).__name__} {dst}.'

#### `T.simd.vmadd`

- **签名**：`T.simd.vmadd(dst, src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Fused multiply-add: dst = dst * src0 + src1.
- **入参**：
  - `dst`：目的 buffer 或区域
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmadd(<dst>, <src0>, <src1>)
```

- **注意事项**：
  - f'vmadd: `dst` must be a buffer access (e.g. c[0]) allocated with T.simd.alloc_local()/alloc_var(), not an immutable register value. Got {type(dst).__name__} {dst}.'

#### `T.simd.vaxpy`

- **签名**：`T.simd.vaxpy(dst, src, scalar, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Fused scalar multiply-add: dst = src * scalar + dst.
- **入参**：
  - `dst`：目的 buffer 或区域
  - `src`：源 buffer 或区域
  - `scalar`：标量操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vaxpy(<dst>, <src>, <scalar>)
```

- **注意事项**：
  - f'vaxpy: `dst` must be a buffer access (e.g. c[0]) allocated with T.simd.alloc_local()/alloc_var(), not an immutable register value. Got {type(dst).__name__} {dst}.'

#### `T.simd.vdiv`

- **签名**：`T.simd.vdiv(src0, src1, mask=None, mode='MODE_ZEROING', precision=None)`
- **核心功能**：Divide vectors, optionally selecting the SFU implementation. ``precision=None`` follows ``tl.enable_fast_math`` (precise fp32 division when fast math is off, hardware instruction otherwise). Non-fp32 division always uses the hardware instruction. ``precision`` selects the implementation per op: - ``'ftz_true'``: bar...
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
  - `precision` (默认 `None`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vdiv(<src0>, <src1>)
```

- **注意事项**：
  - ``precision`` selects the implementation per op: - ``'ftz_true'``: bare hardware SFU (flush-to-zero semantics) - ``'exact'`` (alias ``'vdiv_0ulp_ftz_true'``): correctly-rounded fp32 division (CANN DivAlgo::PRECISION_0ULP_FTZ_TRUE / DivPrecisionImpl) Requires float32 for precise paths.

#### `T.simd.vmax`

- **签名**：`T.simd.vmax(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 最大值
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmax(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vmin`

- **签名**：`T.simd.vmin(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 最小值
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmin(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vand`

- **签名**：`T.simd.vand(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量按位与
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vand(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vor`

- **签名**：`T.simd.vor(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量按位或
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vor(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vxor`

- **签名**：`T.simd.vxor(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量按位异或
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vxor(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vshl`

- **签名**：`T.simd.vshl(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 左移
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vshl(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vshr`

- **签名**：`T.simd.vshr(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 右移
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vshr(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vexp`

- **签名**：`T.simd.vexp(src, mask=None, mode='MODE_ZEROING', precision=None)`
- **核心功能**：向量逐 lane 指数 exp
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
  - `precision` (默认 `None`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vexp(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vln`

- **签名**：`T.simd.vln(src, mask=None, mode='MODE_ZEROING', precision=None)`
- **核心功能**：向量逐 lane 自然对数 ln
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
  - `precision` (默认 `None`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vln(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vsqrt`

- **签名**：`T.simd.vsqrt(src, mask=None, mode='MODE_ZEROING', precision=None)`
- **核心功能**：向量逐 lane 平方根
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
  - `precision` (默认 `None`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vsqrt(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vabs`

- **签名**：`T.simd.vabs(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 绝对值
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vabs(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vneg`

- **签名**：`T.simd.vneg(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量逐 lane 取负
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vneg(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vrelu`

- **签名**：`T.simd.vrelu(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量 ReLU（负值截断为 0）
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vrelu(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vlrelu`

- **签名**：`T.simd.vlrelu(src, alpha, mask=None)`
- **核心功能**：Leaky ReLU with scalar slope (f16/f32): dst = src >= 0 ? src : alpha * src.
- **入参**：
  - `src`：源 buffer 或区域
  - `alpha`：见签名与源码约束
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vlrelu(<src>, <alpha>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vprelu`

- **签名**：`T.simd.vprelu(src0, src1, mask=None)`
- **核心功能**：Parametric ReLU with per-lane slope vector (f16/f32).
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vprelu(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vnot`

- **签名**：`T.simd.vnot(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量按位取反
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vnot(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vdup`

- **签名**：`T.simd.vdup(src, dtype_str: DType, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Broadcast scalar to all lanes: dst = vdup(scalar, dtype_str, mask). dtype_str specifies the target vector element type (e.g. "float32").
- **入参**：
  - `src`：源 buffer 或区域
  - `dtype_str` (DType)：见签名与源码约束
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vdup(<src>, <dtype_str>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vdupv`

- **签名**：`T.simd.vdupv(src, mask=None, pos='POS_LOWEST', mode='MODE_ZEROING')`
- **核心功能**：Broadcast lane N of src vector to all lanes: dst = vdupv(src, mask, pos). pos: "POS_LOWEST" (lane 0) or "POS_HIGHEST" (lane N).
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `pos` (默认 `'POS_LOWEST'`)：见签名与源码约束
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vdupv(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcpadd`

- **签名**：`T.simd.vcpadd(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Pairwise adjacent-lane add. The sums of adjacent source lanes are packed into the low half of the result vector. Supported types: float32, float16.
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcpadd(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcadd`

- **签名**：`T.simd.vcadd(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Pairwise add reduction: dst = vcadd(src, mask).
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcadd(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcmax`

- **签名**：`T.simd.vcmax(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Pairwise max reduction: dst = vcmax(src, mask).
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcmax(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcmin`

- **签名**：`T.simd.vcmin(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Pairwise min reduction: dst = vcmin(src, mask).
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcmin(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcgadd`

- **签名**：`T.simd.vcgadd(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Grouped add reduction: dst = vcgadd(src, mask).
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcgadd(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcgmax`

- **签名**：`T.simd.vcgmax(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Grouped max reduction: dst = vcgmax(src, mask).
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcgmax(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcgmin`

- **签名**：`T.simd.vcgmin(src, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Grouped min reduction: dst = vcgmin(src, mask).
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcgmin(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vsqz`

- **签名**：`T.simd.vsqz(src, mask=None, mode='MODE_STORED')`
- **核心功能**：Squeeze selected lanes toward the low lanes: dst = vsqz(src, mask).
- **入参**：
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_STORED'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vsqz(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vusqz`

- **签名**：`T.simd.vusqz(mask, dtype='int32')`
- **核心功能**：Per-lane exclusive prefix count of mask (s8/s16/s32).
- **入参**：
  - `mask`：谓词掩码；缺省按 dtype 全 lane 激活
  - `dtype` (默认 `'int32'`)：元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vusqz(<mask>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vci`

- **签名**：`T.simd.vci(index, dtype_str: DType, order='INC_ORDER')`
- **核心功能**：Index ramp: dst = vci(index, dtype_str, order). dst[lane] = index + lane (INC_ORDER) / index - lane (DEC_ORDER). dtype_str is the destination vector element type (e.g. "int32", "float32").
- **入参**：
  - `index`：逐 lane 索引向量（无符号整型）
  - `dtype_str` (DType)：见签名与源码约束
  - `order` (默认 `'INC_ORDER'`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vci(<index>, <dtype_str>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcmp`

- **签名**：`T.simd.vcmp(src0, src1, mask=None, op='eq')`
- **核心功能**：Elementwise compare -> vector_bool: dst = vcmp_<op>(src0, src1, mask). op: eq/ne/gt/ge/lt/le.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `op` (默认 `'eq'`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcmp(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcmps`

- **签名**：`T.simd.vcmps(src, scalar, mask=None, op='lt')`
- **核心功能**：Compare vector vs scalar -> vector_bool: dst = vcmps_<op>(src, scalar, mask). op: eq/ne/gt/ge/lt/le.
- **入参**：
  - `src`：源 buffer 或区域
  - `scalar`：标量操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `op` (默认 `'lt'`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vcmps(<src>, <scalar>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vintlv`

- **签名**：`T.simd.vintlv(src0, src1)`
- **核心功能**：Interleave two vector registers. Unpack: a, b = vintlv(x, y) The pair is bound once at the call site so the unpack's two ``pair_get`` calls share a single permutation instruction instead of inlining the pair expression twice.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vintlv(<src0>, <src1>)
```

- **注意事项**：
  - vintlv requires an active IRBuilder to bind the pair

#### `T.simd.vdintlv`

- **签名**：`T.simd.vdintlv(src0, src1)`
- **核心功能**：De-interleave two vector registers. Unpack: a, b = vdintlv(x, y) The pair is bound once at the call site so the unpack's two ``pair_get`` calls share a single permutation instruction instead of inlining the pair expression twice.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vdintlv(<src0>, <src1>)
```

- **注意事项**：
  - vdintlv requires an active IRBuilder to bind the pair

#### `T.simd.vpack`

- **签名**：`T.simd.vpack(src, part=0)`
- **核心功能**：Pack wider lanes to narrower lanes: dst = vpack(src, LOWER/HIGHER). Supports u32->u16 and u16->u8 (needed for dense UE8M0 scale packing).
- **入参**：
  - `src`：源 buffer 或区域
  - `part` (默认 `0`)：LOWER/HIGHER 半区选择
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vpack(<src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vunpack`

- **签名**：`T.simd.vunpack(src, part=0)`
- **核心功能**：Widen half of src: u8->u16, s8->s16, u16->u32, s16->s32.
- **入参**：
  - `src`：源 buffer 或区域
  - `part` (默认 `0`)：LOWER/HIGHER 半区选择
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vunpack(<src>)
```

- **注意事项**：
  - f'vunpack does not support {src_elem}'

#### `T.simd.vexpdif`

- **签名**：`T.simd.vexpdif(src0, src1, mask=None)`
- **核心功能**：Fused exp-sub: dst = exp(src0 - src1). The same-width form supports matching float32 vectors. Widening float16 inputs to float32 requires separate even/odd results and is not represented by this API.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vexpdif(<src0>, <src1>)
```

- **注意事项**：
  - T.simd.vexpdif requires matching float32 vectors
  - Widening float16 inputs to float32 requires separate even/odd results and is not represented by this API.

#### `T.simd.vabsdif`

- **签名**：`T.simd.vabsdif(src0, src1, mask=None, mode='MODE_ZEROING')`
- **核心功能**：Fused abs-sub: dst = vabsdif(src0, src1, mask, mode). Computes dst = abs(src0 - src1) in a single instruction. Supported types: float32, float16.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - Computes dst = abs(src0 - src1) in a single instruction.
- **示例**：

```python
with T.SimdVF():
    T.simd.vabsdif(<src0>, <src1>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vcvt`

- **签名**：`T.simd.vcvt(src, target_dtype: DType, mask=None, round='ROUND_R', sat=True, part=0, mode='MODE_ZEROING')`
- **核心功能**：Vector type conversion between float32, float16, bfloat16, float8, float4, and integers.
- **入参**：
  - `src` (VReg)：Source vector register holding the input element values.
  - `target_dtype` (DType)：Destination element type, e.g. "float32", "float16", "bfloat16", "float8_e4m3", "float8_e5m2", "float4_e2m1fn", "int32", "int16", "uint16", "int8", "uint8", "int64".
  - `mask` (VReg (bool), 默认 `None`)：Predicate mask; only lanes where mask[i] is true participate and are written. Defaults to an all-lanes mask matching the narrower (low-lane) element: the source width when widening, the target width when narrowing.
  - `round` (str, 默认 `'ROUND_R'`)：IEEE-754 rounding mode. Valid values: - ``"ROUND_R"`` (default): Round to nearest, ties to even (banker's rounding). - ``"ROUND_A"``: Round away from zero. - ``"ROUND_F"``: Round toward -inf (floor). - ``"ROUND_C"``: Round toward +inf (ceiling). - ``"ROUND_Z"``: Round toward zero (truncation). - ``"ROUND_O"``: Round to odd (only for f32->f16). - ``"ROUND_H"``: Round half away from zero (only for hif8 conversions). Not all modes are valid for every conversion pair; unused modes are ignored when absent.
  - `sat` (bool, 默认 `True`)：saturation mode for narrow/overflow-prone conversions. - ``True`` (default): Saturate to target range on overflow. - ``False``: No sat; out-of-range values wrap. Appears in: all float->fp8, f32->f16/bf16, all float->int, integer narrowing. Absent from: widening conversions, f16->bf16, s16->f16, s32->f32 (no overflow possible).
  - `part` (int, 默认 `0`)：Sub-register half/quarter selector for widening/narrowing conversions. For even/odd 2-way splits: ``"PART_EVEN"`` (0), ``"PART_ODD"`` (1). For fp8/fp4/int4 4-way splits: ``"PART_P0"`` (0), ``"PART_P1"`` (1), ``"PART_P2"`` (2), ``"PART_P3"`` (3). Not needed for same-width conversions (e.g. f16<->bf16, s32->f32, f32->s32, f16->s16).
  - `mode` (str, 默认 `'MODE_ZEROING'`)：Write mode for masked-off lanes. - ``"MODE_ZEROING"`` (default): Inactive lanes are set to zero. - ``"MODE_MERGING"``: Inactive lanes preserve their prior value (only on mask-less int->int paths; 920R1 only for mask paths).
- **返回值**：VReg
    Destination vector with the converted elements.

Supported conversion pairs (simplified)
----------------------------------------
* Float->Int: f32->s64/s32/s16, f16->s32/s16/s8/u8, bf16->s32.
* Float->Float: f32->f16/bf16/fp8, f16->fp8/bf16/f32, bf16->fp8/f16/f32/fp4.
* Int->Float: s16/s32/s64->f16/f32, s8/u8->f16.
* Int->Int: Most s/u{8,16,32,64} widening/narrowing pairs.
- **A5 硬件适配**：UB。CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - Valid values: - ``"ROUND_R"`` (default): Round to nearest, ties to even (banker's rounding).
- **示例**：

```python
with T.SimdVF():
    T.simd.vcvt(<src>, <target_dtype>)
```

- **注意事项**：
  - f'Unsupported vcvt conversion {src_dtype}->{target_dtype}'
  - f"Invalid round '{round}'. Must be one of {rounds}"
  - f"Invalid mode '{mode}'. Must be one of {modes}"
  - only lanes where mask[i] is true participate and are written.

#### `T.simd.vsel`

- **签名**：`T.simd.vsel(src0, src1, mask)`
- **核心功能**：Bitwise select: mask ? src0 : src1.
- **入参**：
  - `src0`：第一源操作数
  - `src1`：第二源操作数
  - `mask`：谓词掩码；缺省按 dtype 全 lane 激活
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vsel(<src0>, <src1>, <mask>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vselr`

- **签名**：`T.simd.vselr(src, index)`
- **核心功能**：Select lanes from src using per-lane indices.
- **入参**：
  - `src`：源 buffer 或区域
  - `index`：逐 lane 索引向量（无符号整型）
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vselr(<src>, <index>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vmaxs`

- **签名**：`T.simd.vmaxs(src, scalar, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量与标量逐 lane 取大
- **入参**：
  - `src`：源 buffer 或区域
  - `scalar`：标量操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmaxs(<src>, <scalar>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vmins`

- **签名**：`T.simd.vmins(src, scalar, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量与标量逐 lane 取小
- **入参**：
  - `src`：源 buffer 或区域
  - `scalar`：标量操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmins(<src>, <scalar>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vmuls`

- **签名**：`T.simd.vmuls(src, scalar, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量与标量逐 lane 乘法
- **入参**：
  - `src`：源 buffer 或区域
  - `scalar`：标量操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vmuls(<src>, <scalar>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vadds`

- **签名**：`T.simd.vadds(src, scalar, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量与标量逐 lane 加法
- **入参**：
  - `src`：源 buffer 或区域
  - `scalar`：标量操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vadds(<src>, <scalar>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vshls`

- **签名**：`T.simd.vshls(src, scalar, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量按标量左移
- **入参**：
  - `src`：源 buffer 或区域
  - `scalar`：标量操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vshls(<src>, <scalar>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.vshrs`

- **签名**：`T.simd.vshrs(src, scalar, mask=None, mode='MODE_ZEROING')`
- **核心功能**：向量按标量右移
- **入参**：
  - `src`：源 buffer 或区域
  - `scalar`：标量操作数
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `mode` (默认 `'MODE_ZEROING'`)：越界 lane 处理模式，缺省 MODE_ZEROING
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.vshrs(<src>, <scalar>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

### 3.5 其他

#### `T.simd.update_mask`

- **签名**：`T.simd.update_mask(value, width=32)`
- **核心功能**：Runtime tail predicate: lanes [0, value) active (b8/b16/b32).
- **入参**：
  - `value`：写入/累加值
  - `width` (默认 `32`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.update_mask(<value>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.dhistv2`

- **签名**：`T.simd.dhistv2(dst, src, mask=None, bin=0)`
- **核心功能**：Accumulate a frequency histogram of a uint8 vector into a uint16 vector. ``bin=0`` counts values in ``[0, 127]`` and ``bin=1`` counts values in ``[128, 255]``. The destination register is updated in place.
- **入参**：
  - `dst`：目的 buffer 或区域
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `bin` (默认 `0`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.dhistv2(<dst>, <src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

#### `T.simd.chistv2`

- **签名**：`T.simd.chistv2(dst, src, mask=None, bin=0)`
- **核心功能**：Accumulate a cumulative histogram of a uint8 vector into a uint16 vector. ``bin=0`` returns cumulative counts through values ``[0, 127]`` and ``bin=1`` returns cumulative counts through values ``[128, 255]``. The destination register is updated in place.
- **入参**：
  - `dst`：目的 buffer 或区域
  - `src`：源 buffer 或区域
  - `mask` (默认 `None`)：谓词掩码；缺省按 dtype 全 lane 激活
  - `bin` (默认 `0`)：见签名与源码约束
- **返回值**：返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）
- **A5 硬件适配**：CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用
- **性能建议**：
  - 优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt
- **示例**：

```python
with T.SimdVF():
    T.simd.chistv2(<dst>, <src>)
```

- **注意事项**：
  - 仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐

## 4. 覆盖校验

```bash
cd /home/cjl/workspace/aiop
python scripts/gen_ascend_op_kb.py --check
```

校验会逐个核对：目录内每个公开定义都有对应条目，且签名与源码一致。
