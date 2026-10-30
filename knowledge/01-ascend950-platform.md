# 昇腾 950（dav-3510）平台事实

只写生成算子必须知道的硬约束。事实来源：
`tilelang/ascend/README.md`、`.agents/skills/asc-npu-arch/references/npu-arch-facts.json`、
`docs/zh/api/context/内置数据类型.md`（经 asc-devkit）。

## 1. 目标标识

| 项 | 值 |
|---|---|
| 产品 | Ascend 950PR / Ascend 950DT（Atlas A5） |
| NpuArch | `3510` |
| SocVersion | `ASCEND950` |
| `__NPU_ARCH__` | `3510` |
| 编译器 `--npu-arch` | `dav-3510` |
| TileLang 默认 | 后端默认 `dav-3510`；`ASCEND_NPU_ARCH` 可选 |

其他代际对照（仅用于避免误用旧架构经验）：

| 统一调用名 | `__NPU_ARCH__` | SocVersion | 产品 |
|---|---|---|---|
| `ascend910b1` | 2201 | ASCEND910B | Atlas A2/A3 |
| `ascend310p` / `ascend610` | 2002 | ASCEND310P | Atlas 推理 |
| `ascend310b1` | 3002 | ASCEND310B | Atlas 200I/500 A2 |
| `ascend950pr_9599` | 3510 | ASCEND950 | Ascend 950PR |

## 2. 计算核心

| 核心 | 引擎 | 职责 | TileLang 入口 |
|---|---|---|---|
| AIC | Cube | 矩阵乘（GEMM/MAD） | `T.Cube()`，或含 `T.gemm` 的自动判定 |
| AIV | Vector | 逐元素、归约、搬运 | `T.Vector()`、`T.SimtVF(...)`、`T.SimdVF()` |

- 混合核在 3510 上始终以 `__mix__(1, 2)` 组启动；`T.MixedKernel(n, sids=1)` 只让
  sub-core 0 执行向量体，`sids=2` 保留双 AIV。
- 一个 AIC 与它的两个 AIV 共享一组标量存储 `SS`：`T.alloc_ss(shape, "uint32"/"uint64")`，
  **3072 字节池**，分配基址 32B 对齐，不复用存储；仅标量访问，VF/DMA/原子不支持。
- 两个 AIV 写同一 buffer 时必须写**不相交字节区间**。

## 3. 内存层级与分配 API

```text
GM (HBM)                        全局，容量大
  ↕ MTE2/MTE3 (DMA)
UB (Unified Buffer)             T.alloc_shared(...)      scope=shared.dyn
  ↕
L1 (Cube Buffer / CBuf)         T.alloc_l1(...)          scope=shared.l1
  ↕
L0A / L0B (Cube 输入)           T.alloc_l0a/l0b(...)     shared.l0a / shared.l0b
  → Cube MAD
L0C (累加器)                    T.alloc_l0c(...)         shared.l0c
寄存器级 fragment               T.alloc_fragment(...)
```

- L0 还提供 MX 缩放因子槽位：`T.alloc_l0a_sf` / `T.alloc_l0b_sf`。
- **容量判断**：分配 shape、copy 访问区域、物理布局是三个不同量；必须把 layout padding、
  buffer 版本数、编译器预留一起算进去。

容量硬数字（来源 `src/ascend/transform/auto_schedule.cc`）：

| 存储 | 容量 | API |
|---|---|---|
| UB | 物理 248 KB；含 SIMT VF 时调度上限 216 KB（32 KB 线程上下文）；按 208 KB 留余量 | `T.alloc_shared` |
| L1 | 512 KB | `T.alloc_l1` |
| L0C | 256 KB | `T.alloc_l0c` |
| L0A / L0B | 各 64 KB | `T.alloc_l0a/l0b` |
| SS | 3 KB 共享池 | `T.alloc_ss` |

并行度参考：Vector（AIV）官方示例用 **64** 个 block，Cube（AIC）GEMM 示例用 **32** 个 block
（`examples/ascend/example_{vecadd,rmsnorm,gemm}.py`）。峰值 TFLOPS/HBM 带宽不在本地事实
资料中，**不要编造**；性能对比以同环境实测 baseline 为准。
- `T.annotate_unlimit_memory("shared", "shared.l1")` 只解除调度器的容量约束，
  不会扩大物理内存，也不让超容量分配变得可执行。

## 4. 向量宽度

SIMD VF 的向量为 **2048 bit**，即 `64 × float32` 或 `128 × float16/bfloat16`。
SIMD 指令测试与 mask 设置按目标架构的向量宽度取值（`T.simd.pset(32)` 表示 32-bit 元素全 lane 有效）。

## 5. dtype

950 支持（`npu-arch-facts.json → dtype_product_support.ascend950`）：

```text
bool, int8_t, uint8_t, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t,
fp8_e5m2_t, fp8_e4m3fn_t, fp8_e8m0_t, int4x2_t,
int16_t, uint16_t, half, bfloat16_t,
int32_t, uint32_t, float, complex32,
int64_t, uint64_t, double, complex64
```

A2/A3（2201）**没有** fp8/fp4/hifloat8/int4 与 complex：这些 dtype 一旦出现在算子规格里，
目标必须锁 3510。

常用 dtype 尺寸与 32B 对齐元素数：

| dtype | size (B) | 32B 元素数 | 生成器可直接初始化 |
|---|---|---|---|
| `bool` / `int8_t` / `uint8_t` | 1 | 32 | 是 |
| `half` / `bfloat16_t` / `int16_t` / `uint16_t` | 2 | 16 | half/int16 是，bf16 否 |
| `float` / `int32_t` / `uint32_t` | 4 | 8 | 是 |
| `int64_t` / `uint64_t` / `double` | 8 | 4 | 整型是，double 否 |
| `fp8_*` / `hifloat8_t` / `int4x2_t` / `fp4x2_*` | 1（打包） | 32 | 否，需专用转换 |

低精度打包：`fp4x2_*` 与 `int4x2_t` 每字节 2 个 lane（4 bit/lane），
`S4` 简写对应 `int4b_t`/`int4x2_t` 存储。生成 golden 与初始化代码前必须确认打包顺序。

## 6. GEMM 硬约束

| 项 | 约束 |
|---|---|
| L1 输入路径 | `transpose_A=False`、`transpose_B=True`（`tilelang/ascend/op/gemm/gemm_mad.py`） |
| 权重布局 | 该 NT 路径下权重为 `[N, K]`，不是 `[K, N]` |
| 累加 dtype | 示例中 fp16/bf16 输入累加到 `float32`；其他组合以所选 GEMM 与 SDK 契约为准 |
| 清零 | 首个 k 步 `clear_accum=True`，其后 `clear_accum=False` |
| L0 输入 GEMM | 有自己的 layout/transpose 处理，不可把 L1 限制外推到所有形式 |
| 精度 | `T.set_hf32_mode("nearest_even"/"nearest_zero")` 仅在支持的 fp32 路径使用，用完恢复 `None` |

L1→L0A/L0B 分块拼接时用 `l0_pitch` 指定完整 tile 的 fractal pitch（单位是 16 元素 M/N 块），
`A/C` 用 `ceildiv(M,16)`，`B` 用 `ceildiv(N,16)`。

## 7. 对齐与尾部

- 32B 对齐是搬运与 DMA 的基本粒度；`DataCopy`/`T.copy` 的行宽非 32B 时按字节粒度处理，
  需要填充时报 `pad_value=` / `data_select=True`（仅 GM→UB）。
- 填充 lane **不会**被自动初始化，后续若被消费必须显式写值。
- 尾部（tail）不改变 GM 输出的逻辑范围；不要把向上圆整后的物理范围当作逻辑 shape。

## 8. 平台能力速查

| 能力 | 说明 | 生成时注意 |
|---|---|---|
| Regbase 编程 | 直接操作寄存器 | 仅在文档或本地代码明确支持时生成 |
| SIMT 编程 | 线程级并行 | `T.SimtVF(threads=...)`；3510 支持 |
| 低精度格式 | FP8/FP4/INT4 | 需确认转换路径与 golden 语义 |
| 标量共享 SS | AIC+2AIV 共享 | 仅标量；shapes 必须静态 |
