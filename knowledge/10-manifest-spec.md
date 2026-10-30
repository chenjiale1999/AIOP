# 10 · TileOPs manifest 规格接入

本文件说明 aiop 如何把 TileOPs 的算子契约（`src/tileops/manifest/spec/*.yaml`）
变成可执行的 `TaskSpec`，以及哪些契约当前能生成、哪些会被如实挡住。

上游权威：`/home/cjl/workspace/TileOPs/docs/design/manifest.md`（schema 定义）。

## 为什么要接

manifest 是算子对外契约的唯一事实来源：`forall` 索引、`params`、`inputs/outputs`
（含 `optional`/`values`/`requires`/`mutated`/`contiguous`）、`let`、`shape_rules`、
`dtype_combos`、`workloads`。手写 TaskSpec 容易与上游漂移；直接从 manifest 绑定
可以保证"生成的就是上游要的形状、dtype 和副作用"。

## 命令

```bash
# 列出入表条目与 aiop 的 family 映射
python -m aiop manifest list
python -m aiop manifest list --manifest /path/to/spec/attention.yaml --json

# 看某条目的签名与所有 workload 行
python -m aiop manifest show MLAPagedFwdOp

# 把一个 workload 行（+ dtype_cases 下标）绑成 TaskSpec
python -m aiop manifest convert RMSNormFwdOp --workload 0 --dtype-case 1 --out /tmp/spec.json

# 直接跑全流程（design → generate → static → compile → simulate）
python -m aiop generate --op GemmFwdOp --workload 1 --dtype-case 0
python -m aiop generate --manifest /path/to/spec --op SoftmaxFwdOp --workload 3
```

`--manifest` 可指向单文件或整个 `spec/` 目录；省略时用
`paths.tileops_manifest`（默认 `/home/cjl/workspace/TileOPs/src/tileops/manifest/spec`）。
`--workload` 接受行下标或 `label`。

## 绑定规则（`aiop/manifest.py`）

| manifest 概念 | 绑定方式 |
|---|---|
| `forall: {X: Dim}` | 行内给出整数；否则按统一规则求解（见下） |
| `forall: {S: Shape}` | 行内给出整数列表，`*S` 按轴拼接 |
| `forall: {T: DType[a \| b]}` | 取该行 `dtype_cases[i]` 的赋值；未在该分支出现的 dtype 索引可以缺省 |
| `forall: {seq: Seq[Int]}` | 行内整数列表或值原语（`repeat/balanced_sizes/...`） |
| `params` | 行值 > `default` > 报错（无默认且缺省） |
| `optional: true` | 按行的 `some` 列表判定是否传入 |
| `optional: "<expr>"` | 求值布尔表达式，支持 `present(t)` 链式依赖 |
| `values` / `requires` | 原样保留到 `TensorSpec.values` / `.requires`，`role=metadata` |
| `mutated` | `role=state`（输入/输出被写回） |
| `types`（type family） | 按 `match` 选 `cases` 分支；未用到的分支参数**惰性求值** |
| `let` | 按依赖顺序反复求值，失败则跳过（只在用到的分支才必须可解） |
| `shape_rules` | 逐条求值，结果写入 `manifest.shape_rules`（`true/false/null`） |

### 未写出的 `Dim` 如何求解

manifest 允许行里省略由统一（unification）决定的维度。aiop 用两条事实：

1. **元数据生成器定长**：承载 `values` 的 1 维（或首轴）张量长度由生成器决定
   - `as_tensor(seq)` → `len(seq)`
   - `prefix_sum(seq)` / `exclusive_prefix_sum(seq)` / `chunk_offsets(seq, ...)` → `len(seq)+1`
   - `packed_positions(seq)` / `token_indices(seq)` → `sum(seq)`
   - `chunk_indices(seq, chunk)` → `sum(ceil(l/chunk))`
2. **批次/请求数**：所有 `Seq[Int]` 长度一致时，未绑定的 `B`（或 `N`）取该长度。

例：`MLAPagedFwdOp` 行 `{S_q: 1, H: 128, DK: 576, NP: 4096, PS: 64, W: 64,
cache_lens: "repeat(4096, 64)"}` 没有写 `B`，但 `cache_seqlens: {shape: "[B]",
values: "as_tensor(cache_lens)"}` 与 `block_table: "[B, W]"` 一起把 `B` 定为 64；
`cache_seqlens` 的 `requires`（`in_range(S_q, W * PS + 1)`）作为调用方义务保留。

## 能生成的 family

`OP_FAMILY` 把 manifest 条目映射到已有 A5 模板：

| manifest 条目 | aiop family | 约束 |
|---|---|---|
| `Relu/Abs/Exp/Neg/Sqrt/Rsqrt/Log/Sigmoid/TanhFwdOp` | elementwise_unary | 输入为 float16/bfloat16/float32，输出 dtype 与输入一致 |
| `Add/Sub/Mul/Div/Maximum/MinimumFwdOp` | elementwise_binary | 两个操作数形状、dtype 完全一致；`alpha == 1` |
| `Sum/Mean/Amax/AminFwdOp` | row_reduce | 归约轴必须是最后一维；`keepdim=False` |
| `SoftmaxFwdOp` | softmax | `dim` 为最后一维；不覆盖输出 dtype |
| `RMSNormFwdOp` / `LayerNormFwdOp` | rmsnorm / layernorm | `normalized_shape` 秩为 1；weight（和 bias）必须传入 |
| `GemmFwdOp` | gemm | `trans_a=False` 且 `trans_b=True`（B 为 `[N, K]`） |

其余条目（attention / MoE / conv / pool / FFT / 量化 ...）能解析、能出 roofline。
未配置 LLM 时 `build_plan` 抛 `UnsupportedOperator`，Agent 返回
`blocked_unsupported`，只写 `DESIGN.md`（契约表 + 元数据张量 + 不支持原因 + 补齐路径），
**不生成假 kernel、不伪造测量**；配置 LLM 后走下一节的 open 模式。

## GEMM 的非 fp32 输出

manifest 的 `GemmFwdOp` 输出 dtype 等于操作数 dtype（fp16/bf16）。L0C 累加器是 fp32，
因此 `Plan.out_cast=True` 时模板走：

```text
L0C(fp32) --dual_copy--> UB(fp32) --SimtVF cast--> UB(dtype) --dual_copy--> GM
```

静态检查 `HW005`/`SEM017` 强制该路径存在；`check_plan` 要求 `block_m` 为偶数。

## LLM 开放模式（无模板 family）

没有 A5 模板的算子（attention/MoE/conv/...）在配置了 LLM 之后不会被直接判死，
而是进入 **open 模式**：

```bash
export AIOP_LLM_BASE_URL=https://api.deepseek.com/v1
export AIOP_LLM_MODEL=<model id>
export AIOP_LLM_API_KEY=<key>
python -m aiop generate --op GQADenseFwdOp --workload 0
```

与模板模式的差别只在“谁来写 kernel”，门禁一条不放松：

| 项 | template 模式 | open 模式 |
|---|---|---|
| tiling 计划 | 按 family 精确规划（UB/L1/L0C 预算） | `Plan.variant="llm_open"`，只固定 grid 上限；容量由 lowering/编译事后验证 |
| kernel 来源 | `aiop/generation/templates.py` | LLM 草稿（`--strategy llm` 自动生效） |
| 静态门禁 | 语法/API/语义/约束 | 完全相同 |
| L1/L2 | 必须通过 | 必须通过 |
| 失败修复 | 回退保守 plan | 带着编译错误回到 LLM repair（最多 `agent.max_attempts` 轮） |
| golden | aiop 内置 torch 参考 | 无 metadata 输入时由 LLM 写参考，标记 `llm-self-referenced` |
| `verified` | L3 通过即为 `True` | **恒为 `False`**（自指参考不算独立证据） |

判定规则：

- 有 `role=metadata` 的输入（如 `block_table`/`cache_seqlens`）→ 不生成 golden，
  L3 直接标记 `unavailable`，只交付 L0–L2 结论；
- 没有 metadata 输入 → LLM 写一份纯 torch 参考，`golden_source="llm-self-referenced"`，
  报告里注明“与该参考一致，不是独立证据”；
- 成功状态永远是 `success_static_only`，不要当成“算子已验证”。

配置项（`config/aiop.yaml`）：`agent.llm.enabled/base_url/model/api_key_env`；
`--strategy auto` 只对**没有模板**的 family 走 LLM，已有模板的仍然用确定性模板。

## 已知边界

- 算术数据类型的 type family（MoE 的 `layout`）不绑定 → `BindError`。
- `complex64/complex128`（FFT）不在 A5 dtype 集合内 → `BindError`。
- 个别条目的参数在行里缺席（如 `LongRoPEFwdOp.rescale_factors`）→ `BindError`，
  错误信息会点名缺的参数。
- 绑定失败不会被静默降级：CLI 直接报错，不会产出伪造的 TaskSpec。

## 全量自检

```bash
python - <<'EOF'
from aiop.manifest import bind_operator, load_manifest
ok = fail = 0
for name, entry in load_manifest().items():
    for i, row in enumerate(entry.get("workloads") or []):
        for c in range(len(row.get("dtype_cases") or [{}])):
            try:
                bind_operator(None, name, i, c); ok += 1
            except Exception:
                fail += 1
print("bound", ok, "failed", fail)
EOF
```

当前：2125/2208 行绑定成功；`attention.yaml` 235/235 全部绑定成功。
