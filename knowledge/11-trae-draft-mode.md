# 用 Trae（IDE Agent）当 LLM：draft 模式

适用场景：**没有 API key / 不能直连模型端点**，但手上有 Trae（或其它 IDE Agent）账号，
想让它承担"写 kernel"这一步，aiop 只做确定性门禁。

核心约定：aiop 不调用任何模型。`prompt` 产出请求 → 外部 Agent 写 `draft.py` →
`generate --strategy draft` 跑 L0–L2 门禁，失败写 `FIX_REQUEST.md` 回传。

## 前提

```bash
cd /home/cjl/workspace/aiop
python -m aiop env                       # 确认 CANN / tilelang 探测正常
python -m aiop manifest list | head      # 需求来自 manifest 时确认条目名
```

需要 `bisheng`（CANN 9.2+）才能过 L2；没有设备编译时状态会停在 L1。

## 第 1 步：产出起草请求

```bash
python -m aiop prompt --op HardsigmoidFwdOp --workload 0 --dtype-case 0
```

`prompt` 支持与 `generate` 相同的输入选择器：

| 输入方式 | 示例 |
|---|---|
| manifest 条目 | `--op RMSNormFwdOp --workload 0 --dtype-case 1` |
| TaskSpec 文件 | `--spec examples/softmax.yaml` |
| 自然语言 | `--requirement "rmsnorm on [128, 4096] float32"` |

产物落在 `workspace/<Op>-<hash>/`：

| 文件 | 作用 |
|---|---|
| `spec.yaml` | 绑定后的张量契约（形状 / dtype / role，测试按原样传参） |
| `DESIGN.md` | 契约表 + roofline + 开放模式边界 |
| `PROMPT.md` | 给 Agent 的硬性 API 规则与交付方式 |

命令末尾会打印下一步该执行的 `generate` 命令，直接复制即可。

## 第 2 步：让 Trae Agent 按 PROMPT.md 写 draft.py

把任务目录的 `PROMPT.md` 交给 Trae Agent，要求它只在**同一目录**新建 `draft.py`，
不要改动 `spec.yaml` / `DESIGN.md` / 已有 kernel。

`PROMPT.md` 里已固化的硬约束（违反会在 L0 被拒）：

- 只 import `tilelang` 与 `tilelang.ascend.language as T`；禁 torch / numpy / 文件 IO / print / eval；
- 模块必须提供 `def build()`，返回 `@T.prim_func`；
- `T.Kernel(blocks)` 只接受一维 grid，不允许 `threads=`；线程域只写在 `T.SimtVF(threads=...)`；
- 片上内存：`T.alloc_shared`（UB）、`T.alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c`（Cube）；
- 归约只能用 `T.alloc_reducer(op="sum"|"max"|"min")` + `reducer_init` / `reducer_update` / `finalize_reducer`，数量顺序必须配对；
- 禁 `T.print` / `T.device_assert` / 动态循环边界；
- UB 208 KiB / L1 512 KiB / L0C 256 KiB；grid Vector ≤ 64、Cube ≤ 32；
- 张量签名与 `spec.yaml` 逐字一致；只声明一个输出。

```bash
ls workspace/HardsigmoidFwdOp-*/draft.py    # 确认已生成
```

## 第 3 步：门禁校验（无 key、无网络）

```bash
python -m aiop generate --op HardsigmoidFwdOp --workload 0 --dtype-case 0 --strategy draft
```

- 不指定 `--draft` 时默认读 `workspace/<task>/draft.py`；
- `--draft /path/to/kernel.py` 可指向任意文件；
- 门禁与模板模式完全相同：语法 → API 白名单 → reducer 生命周期 → 语义 →
  `tilelang.lower` → `bisheng --npu-arch=dav-3510` →（可选）L3 仿真。

通过时的实际输出：

```text
status: success_static_only  verified: False
note:   open-ended mode: no A5 template family; the kernel was drafted by the LLM
        and still passed static + lowering + device compile gates.
note:   L1/L2 verified: lowering + bisheng device compile passed (1.8s).
note:   L3 skipped: no reference implementation is available for this operator
        (golden_source=unavailable).
```

`success_static_only` 表示 **L0–L2 通过**，不代表数值正确。

## 第 4 步：失败则按 FIX_REQUEST.md 修 draft.py，重跑第 3 步

门禁失败时任务目录出现 `FIX_REQUEST.md`：

```markdown
# 修复请求 — HardsigmoidFwdOp

> 阶段：**static** · 布局：`llm_open` / blocks=64

## 错误
unknown Ascend dialect API `T.hardsigmoid`; did you mean T.sigmoid?

## 静态检查 findings
- `API001` error: unknown Ascend dialect API `T.hardsigmoid`; did you mean T.sigmoid?
```

处理方式：把 `FIX_REQUEST.md` 原文贴回 Trae，让它只改 `draft.py`，然后重复第 3 步，
直到状态变成 `success_static_only`。

## 状态语义

| 状态 | 含义 | 下一步 |
|---|---|---|
| `success_static_only` | L0 静态 + L1 lowering + L2 设备编译全过 | 需要数值证据时补 golden 或上板 |
| `blocked_static` | 语法 / API 白名单 / reducer / 语义未过 | 读 `FIX_REQUEST.md`，改 `draft.py` |
| `blocked_compile` | lowering 或 bisheng 失败 | 同上，错误在 `compile_result.json` |
| `blocked_unsupported` | 无模板且未进入 open/draft 分支 | 用 `--strategy draft` 重跑 |

报告中固定出现 `generator: draft`、`mode: open`；`verified` 恒为 `False`。

## draft vs 配置 LLM

| 项 | `--strategy draft`（Trae 当 LLM） | `--strategy llm`（DeepSeek/vLLM/网关） |
|---|---|---|
| 需要 key / 网络 | 否 | 是（`AIOP_LLM_BASE_URL` / `MODEL` / `API_KEY`） |
| 谁起草 | 外部 IDE Agent | OpenAI 兼容 `/chat/completions` |
| 修复方式 | 人工回传 `FIX_REQUEST.md` | 进程内 `repair()` 自动重试（`agent.max_attempts`） |
| 门禁 | 完全相同 | 完全相同 |
| 适用 | 公司账号、内网、离线 | 全自动批跑、CI |

## 可选：让 L3 有参考实现

在任务目录放 `golden_draft.py`（纯 torch 参考，结果赋给 `out`），draft 模式会自动用它跑 L3：

- 报告标记 `golden_source: agent-self-referenced`；
- `verified` 仍为 `False` —— 自指参考不是独立证据；
- 要拿到独立证据，必须补 aiop 内置 golden 或做 L4 上板。

## 常见问题

1. **同一任务目录的 kernel 被坏草稿覆盖**
   `generate` 会把 `draft.py` 内容原样写到 `workspace/<task>/<Op>.py`。
   用坏草稿跑过之后，重新执行第 3 步（不带 `--draft`）即可从 `draft.py` 恢复正常版本。

2. **改了 spec 但任务目录没更新**
   任务 id 由 spec 内容哈希生成；改了 shape/dtype 会落到新目录，重跑第 1 步即可。

3. **`draft not found`**
   `draft.py` 不在 `<task>/draft.py` 且未给 `--draft`。补文件或显式指定路径。

4. **`draft has no build()`**
   约定入口是 `def build()` 返回 `@T.prim_func`，不是 `main()` 或 `kernel()`。

## 命令速查

```bash
# 1) 起草请求
python -m aiop prompt --op <Op> --workload <N> --dtype-case <M>

# 2) Trae Agent 按 PROMPT.md 写 workspace/<task>/draft.py

# 3) 门禁
python -m aiop generate --op <Op> --workload <N> --dtype-case <M> --strategy draft

# 4) 失败 → 读 FIX_REQUEST.md → 改 draft.py → 重复 3)

# 指定外部草稿
python -m aiop generate --op <Op> --workload <N> --dtype-case <M> \
  --strategy draft --draft /path/to/kernel.py
```

相关实现：`aiop/cli.py`（`cmd_prompt`）、`aiop/agent/design.py`（`build_draft_prompt`）、
`aiop/generation/generator.py`（draft 分支）、`aiop/agent/orchestrator.py`（`FIX_REQUEST.md`）。
