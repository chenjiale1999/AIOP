# A3 → A5 OP compatibility

This reference is the authoritative map from the legacy `tilelang_ascend`
(A3) front-end to the merged A5 main repo. It exists because aiop kernels,
skills and knowledge snippets must use the A5 spelling only.

## 1. Authoritative surfaces

```python
import tilelang.ascend.language as T   # A5 entry point: common + Ascend + T.simd
```

| Surface | Where it is defined | Count (runtime) | Notes |
|---|---|---|---|
| `T.*` | `tilelang/language/*.py` (common) + `tilelang/ascend/language/*.py` | 620 | Everything an A5 kernel may call |
| `T.simd.*` | `tilelang/ascend/language/simd.py` | 94 | SIMD MicroAPI, only inside `T.SimdVF()` |
| `tilelang.language` | `tilelang/language/*.py` + `tilelang/cuda/language/*.py` | 728 | Default CUDA facade; CUDA-only names are **not** exported by `T` |

`T.tile.*` and `T.ascend.*` do **not** exist in A5. `dir(T)` has no `tile`
and no `ascend` attribute.

A3 definitions live in the legacy repo at
`tilelang/language/ascend.py`, `tilelang/language/ascend_tile.py`
(exported as `T.tile`) and `tilelang/language/reduce_ascend.py`.

## 2. A3 `T.ascend.*` / top-level migration

| A3 (legacy) | A5 replacement | Notes |
|---|---|---|
| `T.Kernel(*blocks, threads=..., is_npu=True, pipeline=...)` | `T.Kernel(n)` | A5 is a 1-D core grid; `threads=` is rejected. Open `T.SimtVF(threads=...)` inside the body |
| `T.Scope("C")` / `T.Scope("V")` | `T.Cube()` / `T.Vector()` | Use inside `T.MixedKernel(n, sids=...)` |
| `T.alloc_L1(shape, dtype)` | `T.alloc_l1(shape, dtype)` | A3 scope `shared.l1` → A5 `shared.l1` |
| `T.alloc_L0A/ L0B/ L0C` | `T.alloc_l0a/ l0b/ l0c` | A3 used `wmma.matrix_a/b/accumulator` |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` | A5 UB scope is `shared.dyn` |
| `T.copy(src, dst, coalesced_width)` (3rd positional) | `T.copy(src, dst, coalesced_width=...)` | A5 makes every hint keyword-only and adds `transpose`, `l0_pitch`, `l2_cache_ctrl`, `unit_flag_ctrl`, `sub_blockid`, `pad_value`, `data_select` |
| `T.gemm(..., k_pack=, wg_wait=)` | `T.gemm(..., unit_flag_ctrl=, annotations=)` | `k_pack` / `wg_wait` are CUDA knobs; pass backend hints through `annotations` |
| `T.mma(...)` (alias of `npu_gemm`) | `T.gemm(...)` | — |
| `T.gemm_v0(...)` | `T.gemm(...)` | — |
| `T.set_cross_flag(pipe, flag, mode=2)` | `T.ascend_cross_core_set_flag(mode, pipe, flag)` | Argument order changed |
| `T.wait_cross_flag(flag, pipe="")` | `T.ascend_cross_core_wait_flag(mode, pipe, flag)` or `T.ascend_sync_inter_wait(pipe, flag)` | mode defaults to 0 in `ascend_sync_inter_wait` |
| `T.set_flag(src, dst, event_id)` | `T.ascend_set_flag(f"{src}_{dst}", event_id)` | A5 takes one `"MTE2_MTE1"`-style event name |
| `T.wait_flag(src, dst, event_id)` | `T.ascend_wait_flag(f"{src}_{dst}", event_id)` | — |
| `T.barrier_all()` | `T.ascend_pipe_barrier("PIPE_ALL")` | — |
| `T.pipe_barrier(pipe)` | `T.ascend_pipe_barrier("PIPE_" + pipe.upper())` | — |
| `T.printf(fmt, *args)` | `T.print(obj, msg="")` | No format-string varargs on A5; print one buffer/expr/message per call |
| `T.reduce/reduce_sum/reduce_max/reduce_min` | Same names | A5 adds `reduce_abssum/absmax/bitand/bitor/bitxor` and the SimdVF direct path |

## 3. A3 `T.tile.*` namespace

Full list of the 77 public `tilelang/language/ascend_tile.py` names and their
A5 mapping. "A5暂无" means the merged repo has no equivalent operation; do not
emit it in an A5 kernel.

| A3 `T.tile.*` | A5 replacement | Status |
|---|---|---|
| `add` | `T.simd.vadd` / `a + b` | replaced |
| `sub` | `T.simd.vsub` / `a - b` | replaced |
| `mul` | `T.simd.vmul` / `a * b` | replaced |
| `div` | `T.simd.vdiv` / `a / b` | replaced |
| `max`, `min` | `T.simd.vmax`, `T.simd.vmin` | replaced |
| `abs`, `abs_experiment` | `T.simd.vabs` | replaced |
| `exp`, `exp_experiment` | `T.simd.vexp` | replaced |
| `ln` | `T.simd.vln` | replaced |
| `sqrt` | `T.simd.vsqrt` | replaced |
| `rsqrt` | `T.rsqrt` (scalar) or `1 / T.simd.vsqrt(x)` | replaced |
| `relu` | `T.simd.vrelu` | replaced |
| `leaky_relu` | `T.simd.vlrelu` | replaced |
| `axpy` | `T.simd.vaxpy` | replaced |
| `mul_add_dst` | `T.simd.vmadd` | replaced |
| `cast` | `T.simd.vcvt` | replaced |
| `fill`, `fill_experiment` | `T.fill` | replaced |
| `clear` | `T.clear` | replaced |
| `atomic_add` | `T.atomic_add` | replaced |
| `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not` | `T.simd.vand`, `vor`, `vxor`, `vnot` | replaced |
| `bitwise_lshift`, `bitwise_rshift` | `T.simd.vshl`, `T.simd.vshr` | replaced |
| `clamp_min` | `T.simd.vmaxs` | replaced |
| `clamp_max` | `T.simd.vmins` | replaced |
| `clamp` | `T.simd.vmaxs` + `T.simd.vmins` | replaced by composition |
| `compare` | `T.simd.vcmp` / `T.simd.vcmps` | replaced |
| `select` | `T.simd.vsel` | replaced |
| `broadcast` | `T.simd.vdup` / `T.simd.vdupv` | replaced |
| `createvecindex`, `create_vec_index` | `T.simd.vci` | replaced |
| `gather` | `T.simd.vgather2` | replaced |
| `gatherb` | `T.simd.vgatherb` | replaced |
| `gather_mask`, `gathermask_experiment` | `T.simd.vgatherb` | partial: A3 masked gather by pattern is not reproduced; only 32-B block gather exists |
| `mins_experiment` | `T.simd.vmins` | replaced |
| `sub_experiment` | `T.simd.vsub` | replaced |
| `sum_experiment`, `reduce_sum_experiment` | `T.reduce_sum` | replaced |
| `wholereducesum`, `wholereducemax`, `wholereducemin` | `T.reduce_sum`, `T.reduce_max`, `T.reduce_min` | replaced |
| `reduce_sum_mask_experiment` | — | A5暂无 |
| `transpose` | — | A5暂无 (vector transpose). `T.transpose` is a matrix/DMA transpose, not the same op |
| `pow` | — | A5暂无 (compose `exp(ln(x) * y)`) |
| `round` | — | A5暂无 (use `vcvt` rounding modes where applicable) |
| `reciprocal` | — | A5暂无 (compose `1 / x` via `vdiv`) |
| `sigmoid` | — | A5暂无 (compose from `vexp`) |
| `silu` | — | A5暂无 (compose `x * sigmoid(x)`) |
| `sin`, `cos` | — | A5暂无 (scalar `T.sin`/`T.cos` exist; no vector intrinsic) |
| `sort`, `sort32`, `merge_sort`, `topk`, `init_sort_buf` | — | A5暂无 (sort/topk family removed) |
| `brcb`, `brcb_experiment` | — | A5暂无 (block repeat broadcast copy) |
| `arith_progression` | — | A5暂无 |
| `bilinear_interpolation` | — | A5暂无 |
| `datacachecleanandinvalid_experiment` | — | A5暂无 |
| `row_expand_mul`, `row_expand_mul_experiment`, `row_expand_sub_experiment`, `row_expand_div_experiment` | — | A5暂无 (express with `T.Parallel` indexing) |
| `block_reduce_sum`, `block_reduce_max`, `block_reduce_min` | — | A5暂无 (whole-buffer `T.reduce_*` exists, block-level does not) |
| `binary_op`, `unary_op`, `scalar_op` | — | internal dispatch helpers, not user ops |
| `deprecated`, `not_tl` | — | decorators/helpers, not ops |

## 4. A3 `tilelang/language/ascend.py` names with A5暂无

| A3 name | Original file | Original behaviour | A5 workaround |
|---|---|---|---|
| `sync_all` | `tilelang/language/ascend.py` | Global `tl.ascend_sync_all` intrinsic | `T.ascend_threadfence()` is the closest fence; not a full substitute |
| `dump_tensor(tensor, desc, dump_size, shape_info)` | `tilelang/language/ascend.py` | Dump a tensor region to host for debugging | `T.print(buffer)` prints a UB/GM buffer; no dump size/description control |
| `reinterpretcast(dst, src, casttype)` | `tilelang/language/ascend.py` | Bit-reinterpret a buffer with a dtype string | `T.view(buf, dtype=...)` gives a same-storage view; arbitrary `casttype` strings are gone |
| `set_deq_scale(scale)` | `tilelang/language/ascend.py` | Program the dequantization scale register | No register control; apply an explicit multiply in the kernel |
| `shmem_put_nbi` | `tilelang/language/ascend.py` | Non-blocking put into peer GM | A5暂无; stage through GM with `T.copy` |
| `shmem_ub_put_nbi` | `tilelang/language/ascend.py` | Non-blocking UB → peer GM put | A5暂无; stage through GM with `T.copy` |
| `shmem_get_nbi` | `tilelang/language/ascend.py` | Non-blocking get from peer GM | A5暂无; stage through GM with `T.copy` |
| `shmem_ub_get_nbi` | `tilelang/language/ascend.py` | Non-blocking peer GM → UB get | A5暂无; stage through GM with `T.copy` |

## 5. Rules for generated code

1. Import only `tilelang` and `tilelang.ascend.language as T`.
2. Use the A5 spelling: `T.print`, `T.simd.*`, `T.ascend_*`,
   `T.alloc_l1/l0a/l0b/l0c/alloc_shared`. Never emit `T.printf`, `T.tile.*`,
   `T.ascend.*`, `T.alloc_L1/L0A/L0B/L0C`, or `T.alloc_ub`.
3. `aiop`'s static checker rejects legacy names with `API003`; the message
   either names the A5 replacement or marks the op as 「A5架构暂无」.
4. If a required operation only exists as A3暂无, do not silently approximate it:
   record it in the pending-adaptation list and choose a supported decomposition.

## 6. Re-verify

```bash
cd /home/cjl/workspace/aiop
python -m aiop check --kernel <file>     # API001/API002/API003
python - <<'PY'
import tilelang.ascend.language as T
assert not hasattr(T, "tile") and not hasattr(T, "ascend") and not hasattr(T, "printf")
PY
```
