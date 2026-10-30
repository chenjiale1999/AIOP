# 已验证 A5 算子样例库

样例由同一条 Agent 流水线生成（`scripts/build_samples.py`），每个目录包含：

```text
samples/<sample>/
├── DESIGN.md              # 设计：tiling / 容量 / 精度 / 风险
├── spec.yaml              # TaskSpec
├── <op>.py                # 纯 kernel（build + @T.prim_func）
├── test_<op>.py           # 自包含 golden + --json + 精度标记
├── kernel.cce.c           # tilelang.lower 产出的 CCE 源码
├── result.json            # 门禁结果
├── report.md              # 可读报告
└── manifest.json          # revision / flags / plan
```

## 样例索引

| 样例 | 家族 | shape / dtype | 关键结构 | 验证状态 |
|---|---|---|---|---|
| [elementwise_add](elementwise_add/) | elementwise_binary | [1048576] fp32 | UB 双缓冲 + `T.Pipelined` + SIMT | L1/L2 ✅ |
| [relu_bf16](relu_bf16/) | elementwise_unary | [1024, 4096] bf16 | 整 tile 分解 + `T.max`（fp32 计算后 cast） | L1/L2 ✅ |
| [reduce_sum](reduce_sum/) | row_reduce | [128, 1000] fp32 | 行组 + `alloc_reducer(sum)` | L1/L2 ✅ |
| [softmax](softmax/) | softmax | [128, 1000] fp32 | max → exp → sum 三段 reducer | L1/L2 ✅ |
| [rmsnorm](rmsnorm/) | rmsnorm | [128, 4096] fp32 + W | 平方和 reducer + `rsqrt` + 权重 | L1/L2 ✅ |
| [layernorm](layernorm/) | layernorm | [128, 1000] fp32 + W/B | 双 reducer（sum/sumsq）+ clamp | L1/L2 ✅ |
| [gemm_bf16](gemm_bf16/) | gemm | 256×448×256 bf16 | L1→L0C、`T.Persistent`、K 流水 | L1/L2 ✅ |
| [gemm_relu](gemm_relu/) | gemm | 256×224×128 bf16 | `dual_copy` L0C→UB + SIMT ReLU | L1/L2 ✅ |

“L1/L2 ✅” = `tilelang.lower` 产出 CCE 源码，且 `bisheng --npu-arch=dav-3510` 编译成功。
数值验证（L3）在具备匹配 `torch_npu` 与 NPU 的机器上执行
`python -m aiop simulate workspace/<task>` 后更新；本机无该环境，未标记为已验证。

## 复用方式

1. **直接改 shape/dtype**：复制 `<sample>/<op>.py` 与 `test_<op>.py`，修改 `build()` 中的
   常量；确保 tiling 仍整除，或改用 guarded 路径。
2. **作为骨架**：`knowledge/04` 的六类模式与样例一一对应；新算子优先改样例而不是从零写。
3. **通过 Agent 重新生成**：

   ```bash
   python -m aiop generate --requirement "rmsnorm on [256, 2048] bfloat16"
   ```

4. **上板验证**：

   ```bash
   python -m aiop board samples/rmsnorm --host user@a5-host
   ```

## 样例共用的 A5 约束

- `T.Kernel` 一维网格；线程域只在 `T.SimtVF(threads=...)`。
- Vector 算子 64 block 上限；GEMM 32 block 上限。
- UB 预算 208 KB；L1 512 KB；L0C 256 KB；L0A/B 64 KB。
- GEMM 走 `transpose_B=True`，首 K 步 `clear_accum=True`。
- bf16/fp16 的 relu/abs/max/min 用 fp32 计算再 cast，避免 CCE 重载歧义。
- 非整 tile shape 使用 guarded 标量路径（正确性优先，带宽会下降）。
- 交付代码不含 `T.print`/`assert`/`annotate_unlimit_memory`。

## 维护

```bash
python scripts/build_samples.py   # 重新生成全部样例并刷新 index.json
```

上游 TileLang/CANN 升级后必须重新跑该脚本；`index.json` 中的 `status` 是最后一次生成的
真实结果，不得手工修改。
