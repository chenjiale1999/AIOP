# 官方示例地图与算子骨架

上游示例是最可靠的 few-shot 语料。本文给出「需求 → 模板」的映射，以及每个模板的
最小骨架、关键 API 与硬约束。骨架是结构示意，完整可运行版本以列出的文件为准。

## 1. 示例地图

目录：`/home/cjl/workspace/tilelang/examples/ascend/`

| 文件 | 算子 | 教学重点 |
|---|---|---|
| `example_gemm.py` | 持久化 GEMM | swizzle、`T.Pipelined`、精度模式、混合核 epilogue |
| `example_gemm_splitk.py` | Split-K GEMM | 跨 K 分区的原子 / 确定性归约 |
| `example_vecadd.py` | 向量加 | SIMT 与显式 SIMD 两种写法、流水搬运 |
| `example_rmsnorm.py` | RMSNorm | fragment、reducer epoch、行级流水 |
| `example_per_token_cast_to_fp8.py` | per-token FP8 量化 | 同一算法的 SIMT / SIMD 两种实现 |
| `deepgemm/` | 进阶 GEMM | L0 staging、量化、epilogue、K/M/N 三种遍历（`kernels/kmn`、`kernels/mnk`） |
| `flash_attention/` | MHA / GQA 前向与 GQA 反向 | online softmax、`T.Stage`、多 GEMM 融合、原子累加 dQ |

`deepgemm` 内部结构：`api.py`（builder）、`config.py`（配置）、`_heuristics.pyx`
（启发式）、`kernels/{common,copy,epilogue,transform_sf}.py`。
`flash_attention` 内部结构：`core.py`（共享 fwd builder + `FwdTiling`）、`core_bwd.py`
（Delta + `T.Stage` 融合 builder）、`example_*.py`（包装 + 参考实现 + benchmark）。

每个顶层 `example_<name>.py` 都有对应 `test_<name>.py`。

## 2. 模式 A：逐元素 / 带宽受限

适用：add、mul、激活、cast、scale 等无跨元素依赖的算子。

```python
with T.Kernel(num_blocks) as bx:
    a = T.alloc_shared((tile,), "float32")
    b = T.alloc_shared((tile,), "float32")
    c = T.alloc_shared((tile,), "float32")
    T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})
    for step in T.Pipelined(iterations, num_stages=num_stages):
        begin = (step * num_blocks + bx) * tile
        T.copy(A[begin : begin + tile], a)
        T.copy(B[begin : begin + tile], b)
        with T.SimtVF(threads=num_threads):
            for i in T.Parallel(tile):
                c[i] = a[i] + b[i]
        T.copy(c, C[begin : begin + tile])
```

| 项 | 说明 |
|---|---|
| 关键 API | `T.alloc_shared`、`annotate_buffer_versions`、`T.Pipelined`、`T.SimtVF`+`T.Parallel` |
| SIMD 变体 | `with T.SimdVF(): mask = T.simd.pset(32)`，按 64 个 fp32 一段 `vld`→`vadd`→`vsts` |
| 约束 | 尾部单独处理；`T.Parallel` 的 extent 与线程数不必相等（编译器映射） |
| 参考 | `examples/ascend/example_vecadd.py` |

## 3. 模式 B：行级归约 / Normalization

适用：RMSNorm、LayerNorm、Softmax（单段）、sum/max 行归约。

```python
with T.Kernel(N_CORES) as core_id:
    x_ub = T.alloc_shared(rows * d, "float32")
    T.annotate_buffer_versions({x_ub: 2})
    for t in T.Pipelined(iterations, num_stages=2):
        base = ...
        T.copy(X[base : base + rows * d], x_ub)
        with T.SimtVF(threads=threads):
            x_frag = T.alloc_fragment((rows, TILE), "float32")
            for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
            acc = T.alloc_reducer((rows,), "float32", op="sum")
            T.reducer_init(acc)
            for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                T.reducer_update(acc[r], x_frag[r, i] * x_frag[r, i])
            out = T.alloc_fragment((rows,), "float32")
            T.finalize_reducer(acc, out)
        T.copy(out, RSTD[...])
```

| 项 | 说明 |
|---|---|
| 关键 API | `T.alloc_fragment`、`T.alloc_reducer`/`reducer_init`/`reducer_update`/`finalize_reducer`、`T.rsqrt`、`T.if_then_else` |
| 归约语义 | reducer 只支持 `sum`/`max`/`min`；句柄不可读、不可 fill、不可原地 finalize |
| 尾部 | 用 `if_then_else` + 零填充把非整 tile 补齐，避免单独分支 |
| 参考 | `examples/ascend/example_rmsnorm.py` |

## 4. 模式 C：持久化 GEMM

适用：大矩阵乘、带 epilogue 的融合 GEMM。

```python
with T.Kernel(num_blocks) as bx:
    A_l1 = T.alloc_l1((block_M, block_K), T.bfloat16)
    B_l1 = T.alloc_l1((block_N, block_K), T.bfloat16)
    C_l0c = T.alloc_l0c((block_M, block_N), T.float32)
    C_ub = T.alloc_shared((block_M // 2, block_N), T.float32)

    for tile in T.Persistent([M // block_M * n_tiles], num_blocks, bx):
        m, n = tile // n_tiles * block_M, tile % n_tiles * block_N
        for k in T.Pipelined(K // block_K, num_stages=2):
            T.copy(A[m, k * block_K], A_l1)
            T.copy(B[n, k * block_K], B_l1, l2_cache_ctrl="NOTALLOC_KEEP")
            T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))
        T.dual_copy(C_l0c, C_ub)
        with T.SimtVF(threads=128):
            for i, j in T.Parallel(block_M // 2, block_N):
                C_ub[i, j] = T.max(C_ub[i, j], 0)
        T.dual_copy(C_ub, C[m : m + block_M, n : n + block_N],
                    l2_cache_ctrl="NOTALLOC_PW")
```

| 项 | 说明 |
|---|---|
| 关键 API | `alloc_l1`/`alloc_l0c`、`T.Persistent`、`T.Pipelined`、`T.gemm`、`T.dual_copy` |
| 布局 | 该路径 `transpose_A=False`、`transpose_B=True`，权重按 `[N, K]` 存 |
| 累加 | 首 k 步 `clear_accum=True`；fp16/bf16 → fp32 累加 |
| L0 staging | 需要更细的分块时走 `alloc_l0a/l0b` + `T.copy(..., l0_pitch=...)`；参考 `deepgemm/kernels/mnk` |
| swizzle | 需要 GROUP_M 面板/N-first snake 时用 `T.AscendTileScheduler` 系列 |
| 参考 | `examples/ascend/example_gemm.py`、`examples/ascend/deepgemm/` |

## 5. 模式 D：Split-K

适用：K 方向过大、单核无法一次累加，或需要提高并行度。

| 决策 | 做法 |
|---|---|
| 确定性优先 | 每个 K 分区写独立缓冲，最后一次有序归约 |
| 性能优先 | 直接 `T.atomic_add` 累加到 GM；用 `return_prev=True` 可读旧值 |
| DMA 路径 | `T.set_atomic("add", dtype)` 让 FixPipe 存储以累加方式写，结束 `T.set_atomic_none()` |

**必须先初始化输出**（原子累加不会清零），并明确哪个 task 负责初始化与最终合并。
参考 `examples/ascend/example_gemm_splitk.py`（`--deterministic` 切换两种模式）。

## 6. 模式 E：Attention（online softmax）

适用：MHA / GQA / MLA 类融合注意力。

| 项 | 说明 |
|---|---|
| 入口 | `flash_attention/core.py` 的 `flash_attention_fwd` builder + `FwdTiling` |
| 前向 | 在线 softmax，Q tile 在核内循环 KV；MHA 是 `q_len == kv_len` 的特例 |
| 反向 | 两次启动：先 `bwd_preprocess` 算 `Delta = sum(O*dO)`，再用 KV 为中心的混合核一次跑 5 个 GEMM，dQ 用 bf16 原子加回 GM |
| 关键 API | `T.gemm`、`T.SimtVF`（softmax）、`T.SimdVF`（NZ 打包）、`T.Stage`（反向调度）、`T.atomic_add` |
| 硬约束 | `head_dim` 固定 128；`block_q` 为偶数；`block_kv == 2*VL == 128`；`q_len % block_q == 0`、`kv_len % block_kv == 0`；输入 bf16、累加 fp32 |
| 参考 | `examples/ascend/flash_attention/{core.py,core_bwd.py,example_gqa_bwd.py}` |

改动 attention 时必须保持 head 映射、mask、scale 位置、归一化与累加顺序；反向要校验全部梯度输出。

## 7. 模式 F：低精度量化

适用：per-token / per-group FP8 量化、scale 生成与打包。

| 项 | 说明 |
|---|---|
| 两种实现 | SIMT（`T.Parallel` + 标量逻辑）与 SIMD（`T.simd.*` 显式指令），同一算法两份 |
| 关键 API | `T.reduce_absmax`、`T.simd.vcvt`/`vld`/`vsts`/`vsstb`、`T.alloc_reducer` |
| 选择依据 | 需要特定指令/mask/舍入时用显式 SIMD；否则先用高层 `T.Parallel`，确认 lowering 支持该 pattern |
| 参考 | `examples/ascend/example_per_token_cast_to_fp8.py` |

## 8. 需求 → 模板选型

| 需求特征 | 首选模板 | 次选 |
|---|---|---|
| 纯逐元素、单趟读写 | A（vecadd） | SIMD 变体 |
| 行内归约、归一化 | B（rmsnorm） | 模式 A + `T.reduce` |
| 矩阵乘、可融合 epilogue | C（gemm） | `deepgemm` |
| K 维超预算、需切 K | D（splitk） | C + 多次启动 |
| QK^T / softmax·V 融合 | E（attention） | C + B 手工拼装 |
| 量化/反量化、scale 生成 | F（quant） | A + `T.reduce_absmax` |
| 规整形状 + 极致性能 | `deepgemm` | C |

## 9. 生成前的骨架选择检查

1. 算子是否能在单个 `T.Kernel` 内表达？跨核归约是否需要 split-K 或两段式？
2. 数据流是否会跨 GM→UB→L1→L0 四级？只走 GM↔UB 的场景不要引入 Cube buffer。
3. epilogue 能否放进 `T.SimtVF` 与 GEMM 同核完成？能则不要拆成两次 launch。
4. 尾部是否可整除？不可整除时先确认模板的 tail 处理方式，再决定是否补零。
5. dtype 组合是否在 950 支持范围内（见 `01-ascend950-platform.md` §5）。
