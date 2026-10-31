# aiop — Ascend A5 × TileLang 算子自动生成 Agent

面向 **Ascend 950PR/DT（Atlas A5，`dav-3510`，NpuArch 3510）** 的端到端算子生成系统：
把「算子需求」变成**可编译、可验证、可上板**的 TileLang A5 kernel，并如实记录每一层的证据与缺口。

> 文档基线：`main` @ `242a6ad`（`feat:llm`）+ 工作区中的 draft 模式与 Campaign 架构变更（未提交）。
> 包版本 `0.1.0`，Python ≥ 3.10。

```text
需求 / TaskSpec / TileOPs manifest
  → TaskContract（语义 / ABI / 精度 / 性能目标 / 资源上限 / 检查义务）
  → SkillRegistry（技能选择 + 内容锁定 + 知识召回）→ DESIGN.md
  → Planner / SearchController（族感知候选计划，可选有界搜索）
  → kernel 来源（三选一）
       ├─ 已验证模板（确定性）
       ├─ LLM open 模式（无模板 family）
       └─ 外部 Agent 草稿 draft.py（Trae / IDE，无需 API Key）
  → 静态检查 L0（语法 / API / 容量约束 / 语义）
  → 设备编译 L1/L2（tilelang.lower + bisheng dav-3510，隔离子进程）
  → LoweringAudit（计划声明 vs lowering 可观测事实）
  → 仿真验证 L3（normal/zeros/ones/large/tiny 多场景 + 负例门禁）
  → 跨服务器上板 L4（SSH，可选）
  → EvidenceStore / Diagnoser → campaign_report.md
```

## 1. 项目概述

### 1.1 定位

aiop 不是「让模型直接写代码」的代码生成器，而是**带门禁的算子生产流水线**：

- **确定性优先**：7 个算子家族由知识驱动的模板 + 容量规划器生成，无网络、可复现。
- **模型只在写代码这一步**：LLM 或外部 IDE Agent 只负责起草/修复 kernel，
  静态检查、lowering、设备编译、精度门禁全部由 aiop 本地执行，不能绕过。
- **诚实性约束**：未实际执行的编译/仿真/上板不会被写成「通过」；
  L3 不可用时标记 `unavailable`，交付状态为 `success_static_only`；
  `verified` 只在**独立 golden**（aiop 内置参考）通过时为 `True`。

### 1.2 核心能力

| 能力 | 说明 |
|---|---|
| 需求接入 | 自然语言、TaskSpec YAML/JSON、TileOPs manifest（`--op/--workload/--dtype-case`） |
| 知识召回 | BM25（中英混合分词）+ 标题加权，索引 `.index/knowledge.json`，按 mtime 失效 |
| 设计文档 | `DESIGN.md`：契约表、容量预算、API 映射、roofline、验证方案 |
| 确定性生成 | 7 个 family 模板；规划器给出 UB/L1/L0C 预算与 grid/tile/stages |
| 开放生成 | LLM open 模式（无模板 family）与 draft 模式（外部 Agent 起草，无 API Key） |
| 静态门禁 | 语法、Ascend dialect API 白名单、reducer 生命周期、容量/语义规则（规则号稳定） |
| 编译门禁 | 隔离 worker 内 `tilelang.lower` + `bisheng`，错误自动分类（CANN 过旧/容量/lowering…） |
| 修复闭环 | 模板回退（保守 plan）/ LLM repair / draft 模式的 `FIX_REQUEST.md` 回传 |
| 仿真验证 | `auto` 探测 `local_npu` → `npusim` → `static`；混合容差精度判定 |
| 上板实测 | SSH 预检、打包上传、远端运行、指标/Profile 采集，生成 `board_report.json` |
| 服务化 | `python -m aiop serve` HTTP API（可选 Bearer 鉴权）+ CLI + pytest |

### 1.3 适用场景

- 在 A5 上快速产出并验证 elementwise / reduce / softmax / norm / GEMM 类算子。
- 把上游 TileOPs 的算子契约（shape/dtype/optional/metadata）直接变成可编译 kernel。
- 没有外网 / 没有 API Key 的内网环境：用 Trae 等 IDE Agent 起草，aiop 只做门禁。
- 需要可审计证据链的交付：每层都有 JSON/Markdown 产物，失败原因可分类定位。

**不适用**：动态 shape、需要运行时多输出/复杂别名的算子、attention/MoE/conv 等尚无模板的
高性能实现（只能走 LLM/draft，且只保证 L0–L2）；也不提供自动 autotune 与绝对性能承诺。

## 2. 功能特性与版本

| 功能 | 说明 | 引入版本 |
|---|---|---|
| 7 家族确定性模板 | elementwise unary/binary、row_reduce、softmax、rmsnorm、layernorm、gemm | 基线 `3f7d3ca` |
| 容量规划与审计 | UB 208 KiB / L1 512 KiB / L0C 256 KiB 预算，越界在生成前拒绝 | 基线 |
| 四类静态检查 | `SYN*` 语法、`API*` 白名单、`HW*` 约束、`SEM*` 语义 | 基线 |
| 隔离编译 worker | `aiop.runtime.compile_worker` 子进程，避免污染主进程 | 基线 |
| 三后端仿真 | `local_npu` / `npusim` / `static`，精度混合容差 | 基线 |
| SSH 上板 | 预检、打包、远端运行、`profile_cmd`、`board_report.json` | 基线 |
| HTTP 服务与基准 | `serve` + `eval`（28 个手写用例） | 基线 |
| TileOPs manifest 接入 | `forall`/`params`/`optional`/`values`/`requires`/`mutated`/type family/`shape_rules` 全量绑定；205 条目中 23 个映射到模板 | `242a6ad` |
| LLM open 模式 | 无模板 family 由 OpenAI 兼容端点起草，门禁不放松；失败进 repair 循环 | `242a6ad` |
| GEMM 非 fp32 输出 | L0C(fp32) → UB → cast → GM（`Plan.out_cast`、`T.dual_copy`）；`HW005`/`SEM017` 强制 | `242a6ad` |
| manifest 基准用例 | `aiop/eval/cases.yaml` 增加 8 个 `{op, workload, dtype_case}` 用例（总数 36） | `242a6ad` |
| **Trae / IDE draft 模式** | `aiop prompt` 产出 `PROMPT.md`；`--strategy draft` 读 `draft.py`；失败写 `FIX_REQUEST.md`；可选 `golden_draft.py` | 工作区（未提交） |
| **TaskContract 契约层** | 语义/ABI/精度/性能目标/资源上限/检查义务冻结为 `contract.json`，下游只消费契约 | 工作区（未提交） |
| **SkillRegistry 技能适配** | 按 family/mode/stage 选技能，sha256 锁定，映射为检查义务，产出 `skill.lock` / `context_manifest.json` | 工作区（未提交） |
| **有界计划搜索** | 族感知候选（tile/grid/threads/stages/GEMM blocks），模型评分 + 静态/编译门禁，产出 `frontier.json` | 工作区（未提交） |
| **LoweringAudit** | 计划声明 vs lowering 可观测事实（VF/DMA/sync/printf），产出 `lowering_audit.json` | 工作区（未提交） |
| **证据链与诊断** | 追加式 `evidence.jsonl` + 失败聚类/回路路由 `diagnosis.json` | 工作区（未提交） |
| **多场景仿真 + 负例门禁** | normal/zeros/ones/large/tiny（按算子域裁剪）；坏 kernel 必须被 `SEM004` 拒绝 | 工作区（未提交） |
| **Campaign CLI** | `aiop campaign` 串起契约→技能→搜索→实现→仿真→上板并输出 `campaign_report.md` | 工作区（未提交） |
| **统一进度日志** | 每个任务写 `run.log`；CLI/服务任务逐行打印阶段进度（`--no-progress` 可关） | 工作区（未提交） |
| **HTTP 命令提交与作业日志** | `/v1/commands` 提交白名单 CLI 命令，`/v1/jobs` 查状态/日志；`/v1/tasks` 支持 manifest `op` | 工作区（未提交） |
| **可搬运测试单文件** | `aiop export` 生成内嵌 kernel、从环境变量发现路径的 `test_<op>.py`，单文件拷到服务器即可运行 | 工作区（未提交） |
| **结构化用例矩阵** | manifest workload×dtype 全枚举；每个 `(shape, datatype)` 组合独立 kernel 变体；逐用例输出精度/性能指标与覆盖率 | 工作区（未提交） |
| 知识库 10/11 | manifest 契约规则、Trae draft 全流程 | `242a6ad` / 工作区 |

### 2.1 端到端能力对照

| 要求 | 落地 |
|---|---|
| 复用参考 Skill / A3 Agent 逻辑 | [.opencode/agents/](.opencode/agents/README.md) 6 个 A5 Agent；[.agent/skills/](.agent/skills/) 8 个 skill（含 `aiop-op-pipeline`） |
| A5 知识库 | `knowledge/01–11`，覆盖语法/硬件/性能/样例/仿真/上板/契约/draft；BM25 实时检索 |
| 端到端生成 | 7 个家族、36 个 benchmark 用例（含 8 个 manifest 绑定用例），静态+编译 100% |
| 规格输入 | TaskSpec YAML/JSON、自然语言、TileOPs manifest |
| kernel 来源 | 模板 / LLM open / 外部 draft，三选一 |
| 错误自修正 | 静态/编译分类 + 保守 plan 回退 + LLM repair / `FIX_REQUEST.md` |
| 仿真验证 | `npusim`/`local_npu`/`static` 三后端，精度混合容差 |
| 跨服务器上板 | SSH 预检/部署/运行/指标/Profile，生成 `board_report.json` |
| 可运行服务 | `python -m aiop serve` + HTTP API + CLI + pytest（124 个测试） |

## 3. 环境依赖

### 3.1 依赖矩阵

| 组件 | 版本 / 要求 | 必需 | 用途与说明 |
|---|---|---|---|
| 操作系统 | Linux（本机工具链按 `x86_64-linux` 布局探测） | ✅ | CANN 的 `bin` / `lib64` / `x86_64-linux` 路径 |
| Python | ≥ 3.10（本机验证 3.11.16） | ✅ | `pyproject.toml` 的 `requires-python` |
| PyYAML | ≥ 6.0 | ✅ | TaskSpec / manifest / 配置解析 |
| setuptools | ≥ 68 | ✅（构建） | `pyproject.toml` 的 `build-system` |
| CANN | **9.2+**，含 `bisheng` 与 A5 头文件/指令 | ✅（L1/L2） | 低于 9.2 会报 `unknown type name 'asc_'`（`blocked_environment`） |
| TileLang A5 原生仓 | 含 `tilelang/ascend/language` 与 `dav-3510` 支持 | ✅（L1/L2） | 路径由 `paths.tilelang_repo` 指定 |
| pytest | 任意近期版本 | ⬜（测试） | `python -m pytest` |
| torch | 与 CANN 匹配 | ⬜（L3） | 参考实现与计时 |
| torch_npu | 与 CANN/torch 版本匹配 | ⬜（L3） | `local_npu` / `npusim` 的运行时 |
| npusim | 随 CANN 提供 | ⬜（L3） | 仅接受 `-s Ascend950` / `Ascend950DT` |
| ssh / scp | OpenSSH 客户端 | ⬜（L4） | 远端 A5 主机（需免交互登录） |
| OpenAI 兼容端点 | `/chat/completions` | ⬜ | `--strategy llm` / 自动 repair；draft 模式不需要 |
| TileOPs 仓 | `src/tileops/manifest/spec/*.yaml` | ⬜ | manifest 输入；也可用仓内 [examples/spec/](examples/spec/) 副本 |

### 3.2 版本探测

```bash
cd aiop

python --version                  # ≥ 3.10
python -c "import yaml; print(yaml.__version__)"   # ≥ 6.0

# 平台事实（CANN / bisheng / npusim / torch_npu / tilelang / /dev/davinci*）
python -m aiop env
python -m aiop env --probe-compile                # 额外编一个最小 A5 kernel 验证工具链
```

`env` 只**如实报告**探测结果，不会因为缺组件而伪造可用。

## 4. 安装部署

以下步骤在仓库根目录执行，按顺序可零障碍跑通。

### 4.1 获取代码与虚拟环境

```bash
git clone <repo-url> aiop
cd aiop

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 4.2 安装 aiop

```bash
pip install -e .            # 安装 PyYAML，并注册 `aiop` 控制台命令
pip install pytest          # 可选：跑测试
```

安装后两种调用等价：`aiop <cmd>` 与 `python -m aiop <cmd>`。

### 4.3 配置路径

编辑 [config/aiop.yaml](config/aiop.yaml)。所有路径都可以留空自动探测：

```yaml
paths:
  tilelang_repo: ""        # 留空：AIOP_TILELANG_REPO / 同级 tilelang / 已安装 A5 包
  tileops_manifest: ""     # 留空：同级 TileOPs / 仓内 examples/spec
cann:
  home: ""                 # 留空：ASCEND_HOME_PATH / /usr/local/Ascend/{latest,cann-*}
  soc_version: Ascend950
```

也可以用环境变量覆盖（优先级：**环境变量 > config YAML > 自动探测**）：

```bash
export AIOP_TILELANG_REPO=/path/to/tilelang
export AIOP_CANN_HOME=/path/to/ascend-toolkit
export AIOP_TILEOPS_MANIFEST=/path/to/manifest/spec
```

### 4.4 自检与冒烟

```bash
# 1) 环境事实
python -m aiop env --probe-compile

# 2) 生成 + 编译（跳过仿真；无仿真后端时 L3 会自动降级并如实标记）
python -m aiop generate --spec examples/softmax.yaml --no-simulate

# 3) 跑一遍基准（36 用例，静态 + 设备编译）
python -m aiop eval --cases aiop/eval/cases.yaml --no-simulate --out reports/eval_local.json
```

期望：第 2 步输出 `status: success_static_only` 与 `note: L1/L2 verified: ...`；
第 3 步 `static_pass=36`、`compile_pass=36`（无 CANN 时编译会失败，属环境缺失而非生成错误）。

### 4.5 可选组件

```bash
# LLM（OpenAI 兼容）
export AIOP_LLM_BASE_URL=https://api.deepseek.com/v1
export AIOP_LLM_MODEL=<model id>
export AIOP_LLM_API_KEY=<key>

# L4 上板
export AIOP_BOARD_HOST=user@a5-host
export AIOP_BOARD_KEY=~/.ssh/id_rsa          # 可选
export AIOP_BOARD_ENV_SCRIPT="source /usr/local/Ascend/ascend-toolkit/set_env.sh"

# HTTP 服务鉴权（可选）
export AIOP_SERVICE_TOKEN=<token>
```

## 5. 快速开始

```bash
# 0. 环境探测（A5 编译需要 CANN 9.2+）
python -m aiop env --probe-compile

# 1. 自然语言生成
python -m aiop generate --requirement "softmax on [128, 1000] float32"

# 2. 用 TaskSpec 精确控制
python -m aiop generate --spec examples/softmax.yaml

# 2b. 直接用 TileOPs manifest 契约
python -m aiop manifest list
python -m aiop manifest show MLAPagedFwdOp
python -m aiop generate --op RMSNormFwdOp --workload 0 --dtype-case 1

# 2c. 没有模板的算子：配置 LLM 走 open 模式（门禁不放松）
python -m aiop generate --op GQADenseFwdOp --workload 0

# 2d. 没有 API Key：用 Trae / IDE Agent 起草
python -m aiop prompt --op HardsigmoidFwdOp --workload 0 --dtype-case 0
#    → 让 Agent 按 PROMPT.md 写 workspace/<task>/draft.py
python -m aiop generate --op HardsigmoidFwdOp --workload 0 --dtype-case 0 --strategy draft

# 3. 已有任务重跑门禁
python -m aiop check    workspace/<task>
python -m aiop compile  workspace/<task>
python -m aiop simulate workspace/<task>
python -m aiop board    workspace/<task> --host user@a5-host

# 4. 基准评测（36 个用例，含 8 个 manifest 绑定用例）
python -m aiop eval --cases aiop/eval/cases.yaml

# 4b. 全链路 campaign（契约 → 技能 → 搜索 → 门禁 → 仿真 → 上板 → 报告）
python -m aiop campaign --spec examples/softmax.yaml --search-budget 3 --board
python -m aiop campaign --op GemmFwdOp --workload 1 --dtype-case 0 --out reports/campaign_gemm.md

# 4c. 导出单文件测试（内嵌 kernel，可直接拷到服务器）
python -m aiop export workspace/<task> --out /tmp/bundle

# 5. HTTP 服务
python -m aiop serve --host 0.0.0.0 --port 8790
```

> `--manifest` 的两种位置不同：`aiop generate --manifest <file|dir>` 直接可用；
> `manifest` 子命令的 `--manifest` 是**父级选项**，必须写成
> `aiop manifest --manifest <file|dir> list`。

## 6. 使用指南

### 6.1 三种输入方式

| 输入 | 用法 | 适用 |
|---|---|---|
| 自然语言 | `--requirement "rmsnorm on [128, 4096] float32"` | 快速试验；只识别算子名 + 首个 shape |
| TaskSpec | `--spec examples/softmax.yaml`（YAML/JSON） | 精确控制 I/O、dtype、精度、attrs |
| manifest | `--op RMSNormFwdOp --workload 0 --dtype-case 1` | 契约来自上游，避免手写漂移 |

`--workload` 接受行下标或行 `label`；`--dtype-case` 是该行 `dtype_cases` 的下标。

### 6.2 生成策略

| `--strategy` | kernel 来源 | 失败处理 | 需要网络/Key |
|---|---|---|---|
| `auto`（默认） | 有模板 → 模板；无模板 → LLM open（未配置 LLM 则 `blocked_unsupported`） | 保守 plan 回退 / LLM repair | 视 family 而定 |
| `template` | 只渲染已验证模板 | 保守 plan 回退（最多 `agent.max_attempts` 轮） | 否 |
| `llm` | 先让 LLM 起草，失败回退模板 | LLM repair 循环 | 是 |
| `draft` | 读外部 `draft.py`（默认 `<task>/draft.py`，可用 `--draft <path>` 指定） | 静态/编译失败写 `FIX_REQUEST.md` 并停止 | **否** |

`draft` 只替换「谁写 kernel」，门禁与模板模式完全相同（语法 → API 白名单 → reducer 生命周期 →
语义 → `tilelang.lower` → `bisheng` → 可选 L3）。完整流程见
[knowledge/11-trae-draft-mode.md](knowledge/11-trae-draft-mode.md)。

`--search` 是独立于 `--strategy` 的计划搜索开关：对确定模板先生成一小批合法候选，
按模型评分排序后逐个过 L0 静态与 L1/L2 编译门禁，选第一个通过的最优计划；预算由
`--search-budget`（默认 4）控制。没有真机测量时 `frontier.json` 明确记录
`objective: model score`，不会把模型评分当成实测性能。

### 6.3 Trae / IDE Agent 起草流程（draft 模式）

```bash
# 第 1 步：产出起草请求（spec.yaml / DESIGN.md / PROMPT.md）
python -m aiop prompt --op HardsigmoidFwdOp --workload 0 --dtype-case 0

# 第 2 步：外部 Agent 按 PROMPT.md 在任务目录写 draft.py（入口必须是 def build()）

# 第 3 步：门禁校验（无 key、无网络）
python -m aiop generate --op HardsigmoidFwdOp --workload 0 --dtype-case 0 --strategy draft

# 第 4 步：失败 → 读 FIX_REQUEST.md → 改 draft.py → 重跑第 3 步
```

- `PROMPT.md` 固化了硬性 API 规则（一维 grid、无 `threads=`、reducer 必须配对、禁打印/断言、
  UB 208 KiB / L1 512 KiB / L0C 256 KiB、张量签名与 `spec.yaml` 逐字一致）。
- `--draft` 指向任意文件；省略时读 `<task>/draft.py`，产物目录会记录 `draft_source.txt`。
- 失败产物：`FIX_REQUEST.md`（阶段 + 错误 + 静态 findings + 约束回顾）。
- 想在 L3 有参考实现：在任务目录放 `golden_draft.py`（纯 torch，结果赋给 `out`）。
  该文件**只对无模板 family（open 模式）生效**；报告标记 `golden_source: agent-self-referenced`，
  `verified` 仍为 `False`（自指参考不是独立证据）。
- 注意：若同时配置了 LLM，静态/编译失败会优先走进程内 `repair()`，而不是写 `FIX_REQUEST.md`；
  若 `draft.py` 缺失或没有 `build()`，任务直接以 `blocked_generate` 结束（不会生成修复请求）。

### 6.4 TileOPs manifest 接入

```bash
python -m aiop manifest list                          # 205 条目，23 个映射到模板
python -m aiop manifest show GemmFwdOp                # 签名 + 所有 workload 行
python -m aiop manifest convert RMSNormFwdOp --workload 0 --dtype-case 1 --out /tmp/spec.json

# 用仓内副本（无需 TileOPs checkout）
python -m aiop manifest --manifest examples/spec/norm.yaml list
python -m aiop generate --manifest examples/spec/norm.yaml --op RMSNormFwdOp --workload 0 --dtype-case 1
```

绑定规则见 [knowledge/10-manifest-spec.md](knowledge/10-manifest-spec.md)：
`forall` 索引、`params` 默认值、`optional`+`some`、`values`/`requires`（`role=metadata`）、
`mutated`（`role=state`）、`contiguous`、多输出契约（可绑定，模板只写主输出）、`let`、
`shape_rules`、`dtype_cases`、type family（`Mat[trans_b, K, N]` 等）全部按 manifest 语义求值；
行内省略的维度按统一规则求解
（`B = len(seq)`、`prefix_sum(seq)` → `len+1`、`packed_positions/token_indices(seq)` → `sum(seq)`）。

### 6.5 重跑与续跑

```bash
python -m aiop check    workspace/<task>       # 只跑 L0 静态检查
python -m aiop compile  workspace/<task>       # 只跑 L1/L2
python -m aiop simulate workspace/<task>       # 只跑 L3（读 test_<op>.py）
python -m aiop board    workspace/<task> --host user@a5-host [--force]

python -m aiop generate --spec examples/softmax.yaml --reuse   # 保留 history/npusim 续跑
python -m aiop generate --spec examples/softmax.yaml --task-id my-task
```

任务 id 默认由 spec 内容哈希生成（`<op>-<sha8>`）；修改 shape/dtype 会落到新目录。

### 6.6 任务产物

```text
workspace/<task>/
├── spec.yaml                    # 绑定后的张量契约（测试按原样传参）
├── DESIGN.md                    # 契约表 + 容量预算 + roofline + 验证方案
├── <op>.py                      # 生成的 kernel（def build() -> @T.prim_func）
├── test_<op>.py                 # 自包含验证脚本（不依赖 aiop 安装）
├── static_report.json           # L0 findings（规则号 + severity）
├── compile_result.json          # L1/L2 结果 + 错误分类 + CCE audit
├── compile_result.cce.c         # lowering 产物（lowering 成功后保留，即使设备编译失败）
├── simulation_report.json       # L3（status/backend/metrics）
├── board_report.json            # L4（可选）
├── result.json / report.md      # 汇总结果与人类可读报告
├── manifest.json / state.json   # 任务清单与阶段状态
├── run.log                      # 统一进度日志（逐行、带时间戳/阶段）
├── contract.json                # TaskContract（语义/ABI/精度/性能目标/资源上限/义务）
├── skill.lock / context_manifest.json   # 技能选择、内容锁定、知识召回
├── frontier.json                # 计划搜索候选与门禁结果（--search）
├── lowering_audit.json          # 计划声明 vs lowering 可观测事实
├── negative_probe.json          # 负例门禁结果（campaign）
├── evidence.jsonl / diagnosis.json      # 追加式事实 + 失败聚类/回路路由
├── campaign_report.md           # campaign 六步汇总报告
├── debug_log.md                 # 每次尝试的分类与错误
├── draft.py / PROMPT.md / draft_source.txt / FIX_REQUEST.md   # draft 模式产物
├── golden_draft.py / golden_ref.py                              # 可选参考实现
└── history/                     # 失败快照（<op>_attemptN.py）
```

### 6.7 状态与 `verified` 语义

| status | 含义 | 下一步 |
|---|---|---|
| `success` | 模板模式，L0–L3 全过（独立 golden） | 可上板 |
| `success_static_only` | L0–L2 通过；L3 不可用或为自指参考 | 补 L3 环境或上板 |
| `blocked_static` | 语法/API/reducer/语义/约束未过 | 看 `static_report.json` / `FIX_REQUEST.md` |
| `blocked_compile` | lowering 或 bisheng 失败 | 看 `compile_result.json` 的 `category` |
| `blocked_generate` | kernel 来源异常（如 draft 缺失/无 `build()`） | 修正输入后重跑 |
| `blocked_environment` | CANN 过旧或主机头文件不兼容 | 升级 CANN / 调整 `compile.device_flags` |
| `blocked_accuracy` | L3 精度未过（重试后） | 看 `simulation_report.json` |
| `blocked_board` | 上板失败/预检失败 | 看 `board_report.json` 与 `board_run.log` |
| `blocked_unsupported` | 无模板且未配置 LLM/draft | 配置 LLM 或 `--strategy draft` |

`verified=True` 仅在 **模板模式 + 编译通过 + L3 通过 + golden 为 aiop 内置参考** 时成立。
open 模式（LLM/draft 起草的无模板 family）恒为 `False`；draft 用在**有内置 golden 的 family**
时仍使用独立参考，L3 真通过即可 `verified=True`（本机无 L3 时为 `success_static_only`）。

`campaign` 会额外跑一次负例门禁：把交付 kernel 复制到 `negative/` 并注入 `T.device_assert`，
必须被 `SEM004` 拒绝；负例失败时 campaign 退出码为 1，但不会改写任务的 `status`。

### 6.8 精度门限

按输出 dtype 取默认值（`aiop/specs.py`），也可在 TaskSpec 的 `precision` 中覆盖：

| dtype | atol | rtol | max_abs_error | matched_ratio |
|---|---|---|---|---|
| float32 | 1e-5 | 1e-5 | 1e-4 | 0.9999 |
| float16 | 1e-3 | 1e-3 | 1e-2 | 0.999 |
| bfloat16 | 1e-2 | 1e-2 | 1e-1 | 0.999 |
| float8 / hifloat8 | 1e-1 | 1e-1 | 2e-1 | 0.99 |
| int / bool | 0 | 0 | 0 | 1.0 |

判定：`matched_ratio ≥ matched_ratio 阈值` 且 `max_abs_error ≤ max_abs_error`。

### 6.9 进度查看与统一日志

每个任务都会写 `workspace/<task>/run.log`，CLI 与 HTTP 服务任务默认把同样的行**实时打印到终端（stderr）**：

```text
[10:16:57] [+  0.0s] [relu-demo] [task] start op=relu family=elementwise_unary strategy=auto ...
[10:16:57] [+  0.0s] [relu-demo] [design] done mode=template variant=simt_ub_staged blocks=32 ...
[10:16:57] [+  0.0s] [relu-demo] [contract] frozen hash=7973e22e236f29cf skills=4 obligations=18
[10:17:04] [+  7.0s] [relu-demo] [search] candidate=e-t2000-b64 status=passed static=True compile=True
[10:17:04] [+  7.0s] [relu-demo] [generate] start attempt=1 max_attempts=3 plan=simt_ub_staged
[10:17:04] [+  7.0s] [relu-demo] [static] passed rules=none
[10:17:04] [+  7.0s] [relu-demo] [compile] start timeout_s=600
[10:17:06] [+  8.9s] [relu-demo] [compile] passed category=none elapsed_s=1.88
[10:17:06] [+  8.9s] [relu-demo] [audit] passed checks=5
[10:17:07] [+ 10.1s] [relu-demo] [simulate] unavailable backend=static cases=0
[10:17:07] [+ 10.1s] [relu-demo] [done] finished status=success_static_only verified=False
```

- `--no-progress` 关闭终端输出（`run.log` 仍会写）；`--json` 的 stdout 不受影响。
- 另开终端跟踪：`tail -f workspace/<task>/run.log`；详细诊断仍在 `debug_log.md`。
- HTTP 服务：任务进度打印在**服务端终端**，也可通过 `GET /v1/tasks/{id}/log` 取完整日志；
  终端命令可通过 `POST /v1/commands` 提交，作业日志用 `GET /v1/jobs/{job_id}/log`。
- `state.json` 仍只在任务开始/结束落盘，阶段级进度以 `run.log` 为准。

### 6.10 导出可搬运的测试文件

```bash
python -m aiop export workspace/<task> --out /tmp/bundle
```

产物是**单个** `test_<op>.py`：

- kernel 源码**逐行铺在文件顶层**（不是字符串、不是 base64），可直接阅读/断点调试；
  测试侧符号统一加 `_aiop` / `_AIOP` 后缀（`golden_aiop_N`、`compare_aiop`、`main_aiop`、
  `CASE_SPECS_AIOP` 等），避免与 kernel 中定义的 `build`/辅助函数重名。
  脚本中**没有**任何指向外部 kernel 文件的 import / `spec_from_file_location` / `KERNEL_PATH`。
- **manifest 全组合枚举**：模板类任务会遍历 TileOPs manifest 的全部 workload 行 × dtype case，
  每个合法的 `(shape, datatype)` 组合渲染一个独立 kernel 变体（`build_0`、`build_1`…），
  因为 TileLang kernel 是按固定 shape/dtype 编译的，不能复用一个 kernel 跑其他组合。
  LLM/draft 等无法确定性再生成的算子只保留其绑定的单组合。
- **结构化用例**：`CASE_SPECS_AIOP` 是扁平列表，每一项显式包含
  `shape`、`datatype`、`inputs`、`output`、`precision`、`bytes`、`flops`、`kernel`、`golden`
  以及数据变体 `normal/zeros/ones/large/tiny`；不读取任何外部测试数据或基准文件。
- **逐用例指标**：每个 case 输出两组可直接解析的指标：
  - `precision`：`max_abs_error`、`mean_abs_error`、`max_rel_error`、`mean_rel_error`、
    `matched_ratio`、`mismatched_elements`、`total_elements`
  - `performance`：`latency_ms`、`compute_ms`（NPU Event 设备耗时）、`bandwidth_gbps`、
    `throughput_gflops`、`throughput_melems_s`、`bytes`、`flops`
- **覆盖率**：`coverage` 给出 `declared/executable/executed/ratio/complete/missing/unsupported`；
  模板无法生成 manifest 行（如广播 shape、非末轴 reduce、UB 超限）仍以
  `available: false` 保留在 `CASE_SPECS_AIOP` 并汇总到 `UNSUPPORTED_CONTRACTS_AIOP`，
  运行时标记 `status: unsupported` 而非静默丢弃；其余可执行 case 全部通过时脚本退出码仍为 0。
- 路径自动发现，不依赖生成机器的绝对路径：
  - tilelang：`AIOP_TILELANG_REPO` > 生成时路径（仅当目录存在才加入 `sys.path`）
  - CANN：`AIOP_CANN_HOME` > `ASCEND_HOME_PATH` > `ASCEND_TOOLKIT_HOME` > 生成时路径 > `/usr/local/Ascend/ascend-toolkit/latest` > `/usr/local/Ascend/cann-*`
- 只使用内嵌 kernel：修改 kernel 后需重新 `aiop generate` 或 `aiop export`，旧测试文件不会感知改动。

输出 JSON 结构（`--json result.json` 同内容）：

```json
{
  "status": "passed",
  "coverage": {"declared": 3, "executable": 3, "executed": 3,
               "ratio": 1.0, "complete": true, "unsupported": 0},
  "cases": [{
    "name": "v0_hidden_state_prefill_float16_normal",
    "shape": [[2048, 4096]],
    "datatype": ["float16"],
    "kernel": "build_0",
    "status": "passed",
    "precision": {"max_abs_error": 0.0, "mean_abs_error": 0.0,
                  "max_rel_error": 0.0, "mean_rel_error": 0.0,
                  "matched_ratio": 1.0, "mismatched_elements": 0,
                  "total_elements": 8388608},
    "performance": {"latency_ms": 0.12, "compute_ms": 0.10,
                    "bandwidth_gbps": 279.6, "throughput_gflops": 0.17,
                    "throughput_melems_s": 69.9, "bytes": 33554432,
                    "flops": 16777216, "samples": 20}
  }],
  "summary": {"total": 15, "passed": 15, "failed": 0,
              "latency_ms": 0.12, "max_abs_error": 0.0, "matched_ratio": 1.0}
}
```

服务器上执行：

```bash
scp /tmp/bundle/test_softmax_2.py user@a5-host:/tmp/
ssh user@a5-host
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 或设置 ASCEND_HOME_PATH
python /tmp/test_softmax_2.py --device npu --json /tmp/result.json
```

退出码：`0` 通过，`1` 精度/运行失败，`3` 环境不可用（无 torch_npu/设备）。

## 7. 架构与目录

```text
aiop/
├── aiop/                       # 可执行 Agent 服务
│   ├── agent/                  # 编排、契约、技能、计划搜索、证据、审计、进度、campaign
│   ├── generation/             # 模板渲染、LLM 适配、修复、draft 读取
│   ├── checks/                 # 静态检查、隔离编译门禁
│   ├── simulation/             # 用例矩阵、测试脚本生成、npusim/local NPU
│   ├── board/                  # SSH 预检/部署/实测
│   ├── knowledge/              # BM25 检索
│   ├── eval/                   # 36 用例基准套件
│   ├── runtime/                # A5 引导、硬件事实、编译 worker
│   ├── service/                # HTTP API
│   ├── manifest.py             # TileOPs manifest 绑定
│   └── cli.py
├── knowledge/                  # A5 知识库（01–11）
├── .agent/skills/              # 8 个可加载 skill（含 aiop-op-pipeline）
├── .opencode/agents/           # A5 多代理定义（orchestrator + 5 subagents）
├── examples/                   # softmax.yaml + spec/（25 个 manifest YAML）
├── samples/                    # 8 个已验证算子样例 + index.json
├── scripts/                    # build_samples.py / sync_skills.sh
├── tests/                      # 124 个单元 + 端到端测试
├── config/aiop.yaml
├── reports/                    # 基准报告
└── workspace/                  # 运行时任务产物（git 忽略）
```

关键模块：

- [aiop/cli.py](aiop/cli.py)：CLI 入口（`generate/campaign/prompt/check/compile/simulate/board/search/env/manifest/eval/serve`）。
- [aiop/agent/orchestrator.py](aiop/agent/orchestrator.py)：六阶段状态机、门禁、重试、`FIX_REQUEST.md`。
- [aiop/agent/contract.py](aiop/agent/contract.py)：TaskContract（ABI/精度/性能目标/资源上限/检查义务）。
- [aiop/agent/skills.py](aiop/agent/skills.py)：SkillRegistry/SkillAdapter、`skill.lock`、`context_manifest.json`。
- [aiop/agent/search.py](aiop/agent/search.py)：族感知候选计划、模型评分、`frontier.json`。
- [aiop/agent/audit.py](aiop/agent/audit.py)：LoweringAudit（计划 vs 实际）。
- [aiop/agent/evidence.py](aiop/agent/evidence.py)：EvidenceStore + Diagnoser。
- [aiop/agent/progress.py](aiop/agent/progress.py)：统一 `run.log` + 终端逐行进度。
- [aiop/agent/campaign.py](aiop/agent/campaign.py)：六步报告与负例门禁。
- [aiop/generation/templates.py](aiop/generation/templates.py)：7 家族模板。
- [aiop/checks/static.py](aiop/checks/static.py)：稳定规则号（`SYN/API/HW/SEM`）。
- [aiop/checks/compiler.py](aiop/checks/compiler.py)：错误分类（`cann_too_old`、`capacity`…）。
- [aiop/runtime/hardware.py](aiop/runtime/hardware.py)：硬件事实、`Plan`、`build_plan`/`build_open_plan`。
- [aiop/manifest.py](aiop/manifest.py)：manifest → `TaskSpec` 绑定。

## 8. 算子支持范围

| 家族 | 变体 | 模板要点 | 状态 |
|---|---|---|---|
| `elementwise_unary` | relu/abs/exp/neg/sigmoid/tanh/rsqrt/sqrt/log/cast | UB 暂存 + `T.Pipelined`；不可整除时走 guarded 标量 GM 路径 | 已验证编译 |
| `elementwise_binary` | add/sub/mul/div/maximum/minimum | 同上，双输入三缓冲 | 已验证编译 |
| `row_reduce` | sum/mean/max/min（最后一维） | `T.alloc_reducer` + fp32 累加，64 AIV，rows ≤ 4 | 已验证编译 |
| `softmax` | 数值稳定（max-shift） | 双 reducer（max/sum）+ `T.exp` | 已验证编译 |
| `rmsnorm` | 行级 reducer + weight | weight 常驻 UB，`T.rsqrt` | 已验证编译 |
| `layernorm` | 双 reducer + weight/bias | 均值/方差双 reducer，方差 clamp | 已验证编译 |
| `gemm` / `gemm_relu` | 持久化 L1→L0C | `transpose_B=True`、`T.Persistent`；可选 relu epilogue；非 fp32 输出走 UB cast | 已验证编译 |

**不支持**：动态 shape、跨级 GM→L0、非 A5 dtype、多输出 kernel、attention/MoE/conv/pool/FFT/
量化/sampling/rope/mamba 等无模板家族（可解析契约并出 roofline，但只有 LLM/draft 路径，且仅保证 L0–L2）。

### 8.1 manifest → 模板映射（23 条目）

| aiop family | manifest 条目 | 约束 |
|---|---|---|
| `elementwise_unary` | Relu/Abs/Neg/Exp/Sqrt/Rsqrt/Log/Sigmoid/TanhFwdOp | 输出 dtype 与输入一致 |
| `elementwise_binary` | Add/Sub/Mul/Div/Maximum/MinimumFwdOp | 两操作数形状/dtype 一致；`alpha == 1` |
| `row_reduce` | Sum/Mean/Amax/AminFwdOp | 归约轴必须是最后一维；`keepdim=False` |
| `softmax` | SoftmaxFwdOp | `dim` 为最后一维；不覆盖输出 dtype |
| `rmsnorm` / `layernorm` | RMSNormFwdOp / LayerNormFwdOp | `normalized_shape` 秩为 1；weight（和 bias）必须传入 |
| `gemm` | GemmFwdOp | `trans_a=False` 且 `trans_b=True`（B 为 `[N, K]`） |

全量自检：2208 个 workload×dtype 组合中 **2125 个可绑定**；`attention.yaml` **256/256** 全部绑定。
其余条目返回 `blocked_unsupported` 并输出契约级 `DESIGN.md`，不生成假 kernel。

## 9. 验证层级与当前实测状态

| 层 | 内容 | 本机状态 |
|---|---|---|
| L0 | 语法 / API 白名单 / 容量约束 / 语义 | ✅ 36/36 用例通过 |
| L1 | `tilelang.lower` 产出 CCE 源码 | ✅ 36/36 |
| L2 | `bisheng --npu-arch=dav-3510` 设备编译 | ✅ 36/36 |
| L3 | npusim / local NPU 数值验证 | ⚠️ 本机无匹配 torch_npu/NPU；自动探测并标记 `unavailable` |
| L4 | 远端 A5 实测 | ⚪ 未配置 host；`board` 返回 `skipped` |

L3 需要**与 CANN 版本匹配的 torch_npu** 以及可见的 `/dev/davinci*`（版本对应关系以官方发布矩阵为准）；
`npusim` 只接受 `-s Ascend950`/`Ascend950DT`。缺少条件时系统不会伪造通过，
而是在 `simulation_report.json` 与最终报告中披露缺口。

基准结果：[reports/eval_manifest.json](reports/eval_manifest.json)（报告：
[reports/eval_manifest.md](reports/eval_manifest.md)），生成（静态+编译）通过率 **100%（36/36）**，
其中 8 个用例直接绑定自 TileOPs manifest。该次基准以 `--no-simulate` 运行，
因此 `simulate_requested=false`、`simulation_pass_rate=null`（未做数值断言，也不冒充通过）。

Campaign 实测（本机）：`softmax_2`（[examples/softmax.yaml](examples/softmax.yaml)）与
`GemmFwdOp`（workload 1 / dtype-case 0）均完成契约 → 技能 → 计划搜索 → L0–L2；
多场景仿真因无 torch_npu 标记 `unavailable`，上板因未配置 host 标记 `skipped`，
负例门禁均按 `SEM004` 拒绝坏 kernel。报告生成在 `reports/campaign_softmax.md` 与
`reports/campaign_gemm.md`（生成物，git 忽略）。

## 10. 配置项说明

配置文件默认 [config/aiop.yaml](config/aiop.yaml)，可用 `--config <path>` 或 `AIOP_CONFIG` 指定。
优先级：**环境变量 > config YAML > 内置默认值**。

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `paths.tilelang_repo` | 自动探测 | A5 原生 tilelang 仓；`AIOP_TILELANG_REPO` / 同级 `tilelang` / 已安装包 |
| `paths.workspace` | `<repo>/workspace` | 任务产物根目录 |
| `paths.knowledge` | `<repo>/knowledge` | 知识库目录（检索与索引） |
| `paths.samples` | `<repo>/samples` | 样例库目录 |
| `paths.reports` | `<repo>/reports` | 基准报告目录 |
| `paths.tileops_manifest` | 自动探测 | `AIOP_TILEOPS_MANIFEST` / 同级 TileOPs / `<repo>/examples/spec` |
| `cann.home` | 自动探测 | `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` / `/usr/local/Ascend/...` |
| `cann.npusim` | `""` | 为空时取 `<cann.home>/x86_64-linux/bin/npusim` |
| `cann.soc_version` | `Ascend950` | `npusim record -s` 的目标芯片 |
| `compile.target` | `ascend` | `tilelang.lower` 的 target |
| `compile.npu_arch` | `dav-3510` | 目标架构标记（预留；实际编译由 `target=ascend` + CANN 工具链决定） |
| `compile.device_flags` | `-D__INTEL_LLVM_COMPILER=1`、`-DSYCL_LANGUAGE_VERSION=1` | 传给设备编译的宏 |
| `compile.timeout_s` | `600` | 单次编译超时 |
| `simulation.backend` | `auto` | `auto`/`npusim`/`local_npu`/`static`；`static` 强制标记 `unavailable` |
| `simulation.timeout_s` | `1800` | L3 超时 |
| `simulation.repeats` | `20` | 计时重复次数 |
| `simulation.warmup` | `5` | 计时预热次数 |
| `board.host` | `""` | SSH 目标（`user@host`），为空则跳过 L4 |
| `board.port` | `22` | SSH 端口 |
| `board.key` | `""` | SSH 私钥路径 |
| `board.remote_dir` | `/tmp/aiop_runs` | 远端工作目录 |
| `board.python` | `python3` | 远端解释器 |
| `board.env_script` | `""` | 远端环境脚本（如 `set_env.sh`） |
| `board.tilelang_repo` | `""` | 远端 tilelang 路径（非空时纳入预检） |
| `board.profile_cmd` | `""` | 实测后的可选 profile 命令（如 `msprof ...`） |
| `board.timeout_s` | `1800` | 远端运行超时 |
| `board.repeats` / `board.warmup` | `50` / `10` | 上板计时参数 |
| `agent.max_attempts` | `3` | 生成/静态/编译的最大尝试轮数 |
| `agent.max_sim_attempts` | `2` | 预留配置项（当前实现仿真失败后固定再试 1 次） |
| `agent.llm.enabled` | `false` | 配置了 `base_url` + `model` 时自动置为 `true` |
| `agent.llm.base_url` | `""` | OpenAI 兼容端点（自动追加 `/chat/completions`） |
| `agent.llm.model` | `""` | 模型 id |
| `agent.llm.api_key_env` | `AIOP_LLM_API_KEY` | 读取 Key 的环境变量名 |
| `agent.llm.timeout_s` | `120` | LLM 请求超时 |

### 10.1 环境变量

| 变量 | 覆盖 | 说明 |
|---|---|---|
| `AIOP_CONFIG` | 配置文件路径 | 默认 `config/aiop.yaml` |
| `AIOP_TILELANG_REPO` | `paths.tilelang_repo` | A5 原生仓 |
| `AIOP_WORKSPACE` | `paths.workspace` | 产物根目录 |
| `AIOP_TILEOPS_MANIFEST` | `paths.tileops_manifest` | manifest 目录/文件 |
| `AIOP_CANN_HOME` | `cann.home` | CANN 安装目录 |
| `AIOP_NPUSIM` | `cann.npusim` | npusim 可执行文件 |
| `AIOP_SOC_VERSION` | `cann.soc_version` | 目标芯片 |
| `AIOP_BOARD_HOST` | `board.host` | SSH 目标 |
| `AIOP_BOARD_KEY` | `board.key` | SSH 私钥 |
| `AIOP_BOARD_DIR` | `board.remote_dir` | 远端工作目录 |
| `AIOP_BOARD_PYTHON` | `board.python` | 远端解释器 |
| `AIOP_BOARD_ENV_SCRIPT` | `board.env_script` | 远端环境脚本 |
| `AIOP_LLM_BASE_URL` | `agent.llm.base_url` | OpenAI 兼容端点 |
| `AIOP_LLM_MODEL` | `agent.llm.model` | 模型 id |
| `AIOP_LLM_API_KEY` | —（`api_key_env` 指定） | 模型 Key |
| `AIOP_SERVICE_TOKEN` | —（服务进程读取） | 设置后 HTTP 需 `Authorization: Bearer <token>` |

## 11. 接口说明

### 11.1 CLI

| 命令 | 作用 | 主要参数 |
|---|---|---|
| `generate` | 跑完整流水线 | `--spec/--requirement/--manifest+--op`、`--workload`、`--dtype-case`、`--strategy`、`--draft`、`--search`、`--search-budget`、`--no-progress`、`--task-id`、`--simulate/--no-simulate`、`--board`、`--force-board`、`--reuse`、`--json` |
| `campaign` | 六步全链路 + 汇总报告 | 与 `generate` 相同的输入选择器、`--search/--no-search`、`--search-budget`、`--no-progress`、`--simulate/--no-simulate`、`--board`、`--out` |
| `prompt` | 写 `PROMPT.md` 供 IDE Agent 起草 | 与 `generate` 相同的输入选择器 |
| `export` | 导出单文件可搬运测试 | `task`、`--out`（默认 `<task>/export`） |
| `check` | 只跑 L0 静态 | `task` |
| `compile` | 只跑 L1/L2 | `task` |
| `simulate` | 只跑 L3 | `task` |
| `board` | 只跑 L4 | `task`、`--host`、`--force` |
| `search` | 检索知识库 | `query`、`-k` |
| `env` | 探测工具链 | `--probe-compile` |
| `manifest` | 查看/转换 manifest | `manifest list/show/convert`，父级 `--manifest` |
| `eval` | 基准评测 | `--cases`、`--out`、`--limit`、`--no-simulate` |
| `serve` | 启动 HTTP 服务 | `--host`、`--port` |

全局选项 `--config <path>` 必须写在子命令之前：`aiop --config cfg.yaml generate ...`。

### 11.2 HTTP API

```text
GET  /                          # 服务索引 + 可用命令清单
GET  /healthz
GET  /v1/env
POST /v1/search                 {"query": "...", "k": 5}
POST /v1/tasks                  {"spec"|"spec_yaml"|"requirement"|"op",
                                 "workload"?, "dtype_case"?, "simulate"?, "board"?,
                                 "strategy"?, "search"?, "search_budget"?, "draft"?, "task_id"?}
GET  /v1/tasks
GET  /v1/tasks/{id}
GET  /v1/tasks/{id}/report
GET  /v1/tasks/{id}/log
POST /v1/tasks/{id}/board       {"host"?, "force"?}
POST /v1/commands               {"command": "...", "args"?: [...], "config"?: "..."}
GET  /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/log      ?tail=N
```

### 11.2.1 提交任务（`/v1/tasks`）

- 输入四选一：`spec`（TaskSpec JSON）、`spec_yaml`（YAML 字符串）、`requirement`（自然语言）、
  `op` + `workload` + `dtype_case`（TileOPs manifest，可选 `manifest` 路径）。
- 其余字段：`simulate`、`board`、`strategy`、`search`、`search_budget`、`draft`、`task_id`。
- 返回 `202 {"task_id": "...", "status": "queued"}`；不传 `task_id` 时**直接返回真实哈希 id**，可直接轮询。
- 运行中 `GET /v1/tasks/{id}` 返回 `202` + `state.json`；结束后返回 `200` + `result.json`。
- `GET /v1/tasks/{id}/report` 取 `report.md`，`GET /v1/tasks/{id}/log` 取统一 `run.log`。

```bash
curl -s localhost:8790/v1/tasks -H 'Content-Type: application/json' \
  -d '{"requirement":"relu on [128, 1000] float32","task_id":"relu-demo","simulate":false}'

curl -s localhost:8790/v1/tasks -H 'Content-Type: application/json' \
  -d '{"op":"GQADenseFwdOp","workload":0,"dtype_case":0,"task_id":"gqa-demo","simulate":false}'
```

### 11.2.2 提交终端命令（`/v1/commands`）

把 CLI 命令作为子进程提交，参数原样透传；stdout/stderr 合并写入**作业日志**，并实时打印到服务端终端。

允许的命令：`generate`、`campaign`、`prompt`、`export`、`check`、`compile`、`simulate`、`board`、
`search`、`env`、`manifest`、`eval`（`serve` 被禁用，避免递归启动）。

```bash
# 提交 generate（等价于终端里的 python -m aiop generate ...）
curl -s localhost:8790/v1/commands -H 'Content-Type: application/json' -d '{
  "command": "generate",
  "args": ["--spec", "examples/softmax.yaml", "--search", "--search-budget", "3"]
}'

# args 也可以是一个字符串（按 shell 规则拆分）
curl -s localhost:8790/v1/commands -H 'Content-Type: application/json' -d '{
  "command": "manifest",
  "args": "list --json"
}'

# 可选：为这个 job 指定 config
curl -s localhost:8790/v1/commands -H 'Content-Type: application/json' -d '{
  "command": "eval",
  "args": ["--cases", "aiop/eval/cases.yaml", "--no-simulate"],
  "config": "config/aiop.yaml"
}'
```

返回：

```json
{"job_id": "job-1a2b3c4d", "command": "generate", "status": "queued",
 "exit_code": null, "task_id": null, "log": "workspace/_jobs/job-1a2b3c4d.log"}
```

查询与日志：

```bash
curl -s localhost:8790/v1/jobs                     # 作业列表（含重启前落盘的元数据）
curl -s localhost:8790/v1/jobs/job-1a2b3c4d        # status/exit_code/task_id
curl -s "localhost:8790/v1/jobs/job-1a2b3c4d/log?tail=100"   # 最后 100 行
```

- `status`：`queued → running → passed|failed`；`exit_code` 即 CLI 退出码。
- `task_id`：`generate`/`campaign` 从 `task:` 行解析；`check/compile/simulate/board` 从第一个参数解析。
- 作业元数据落盘到 `workspace/_jobs/<job_id>.json`，日志落盘到 `<job_id>.log`，服务重启后仍可查询。
- 不提供取消/杀进程接口；长任务会占用 2 个 worker 之一。

### 11.2.3 鉴权与安全

- 设置 `AIOP_SERVICE_TOKEN` 后所有请求需 `Authorization: Bearer <token>`。
- `POST /v1/commands` 只允许白名单命令，且不经过 shell（参数按列表传给子进程）。
- 默认建议绑定 `127.0.0.1`；开放到网络时务必设置 token，并限制可写的 `config`/`args` 路径。

### 11.3 Python API

```python
from aiop.config import load_config
from aiop.agent.orchestrator import OperatorAgent
from aiop.manifest import bind_operator
from aiop.runtime.hardware import build_plan

cfg = load_config()
spec = bind_operator(None, "RMSNormFwdOp", workload=0, dtype_case=1)

plan = build_plan(spec)                      # 无模板 family 抛 UnsupportedOperator
result = OperatorAgent(cfg).run(spec, simulate=False, strategy="auto")

result["status"]        # success_static_only / ...
result["verified"]      # 是否拿到独立证据
result["artifacts"]     # kernel / design / compile_report ...
```

`OperatorAgent.run(spec, *, task_id=None, simulate=True, board=False, force_board=False,
strategy="auto", reuse_workspace=False, draft=None, search=False, search_budget=4,
progress=False)`。`progress=True` 时写 `run.log` 并把阶段行打印到 stderr。

契约 / 技能 / 搜索 / 证据 / 审计 / campaign 可直接复用：

```python
from aiop.agent.contract import build_contract
from aiop.agent.skills import SkillRegistry
from aiop.agent.search import propose_candidates
from aiop.agent.evidence import diagnose
from aiop.agent.campaign import run_negative_probe, write_campaign_report

bundle = SkillRegistry().select(spec)
contract = build_contract(spec, plan, obligations=bundle.obligations)
candidates = propose_candidates(spec, plan)      # 已按模型评分排序
negative = run_negative_probe(task_dir, spec, plan, cfg)
write_campaign_report(task_dir, result, negative=negative)
```

### 11.4 接口变更记录（相对基线 `3f7d3ca`）

| 变更 | 位置 | 说明 |
|---|---|---|
| 新增 `prompt` 子命令 | `aiop/cli.py` | 产出 `PROMPT.md` / `spec.yaml` / `DESIGN.md`，打印下一步命令 |
| `generate` 新增 `--draft`，`--strategy` 增加 `draft` | `aiop/cli.py` | 外部草稿路径，默认 `<task>/draft.py` |
| `OperatorAgent.run(..., draft=None)` | `aiop/agent/orchestrator.py` | draft 模式在静态/编译失败时写 `FIX_REQUEST.md` |
| `generate(..., draft_path=None)` | `aiop/generation/generator.py` | 读取外部 kernel，记录 `draft_source.txt` |
| 新增 `build_open_plan` | `aiop/runtime/hardware.py` | `Plan.variant="llm_open"`，只固定 grid 上限 |
| `Plan.out_cast` | `aiop/runtime/hardware.py` | GEMM 非 fp32 输出的 UB cast epilogue |
| 静态规则更新 | `aiop/checks/static.py` | `HW005` 允许 fp16/bf16 输出（需 `out_cast`）；新增 `SEM017` 强制 `T.dual_copy` |
| 新增 manifest 绑定 | `aiop/manifest.py` | `bind_operator`/`list_operators`/`describe_operator`；23 条目映射 |
| TaskSpec/TensorSpec 扩展 | `aiop/specs.py` | `optional/values/requires/mutated/contiguous/role/raw_shape`、`TaskSpec.manifest` |
| 新增配置项 | `aiop/config.py` | `paths.tileops_manifest`、`AIOP_TILEOPS_MANIFEST` |
| 仿真 golden 扩展 | `aiop/simulation/testgen.py` | `GOLDEN_FAMILIES`、`golden_override`；golden 来源标记 |
| 基准用例扩展 | `aiop/eval/cases.yaml`、`aiop/eval/run_benchmark.py` | `{op, workload, dtype_case}` manifest 用例（28 → 36） |
| 知识库扩展 | `knowledge/10`、`knowledge/11` | manifest 契约、Trae draft 流程 |
| 新增 `campaign` 子命令 | `aiop/cli.py`、`aiop/agent/campaign.py` | 六步链路 + 汇总报告 + 负例门禁 |
| 新增 `--search/--search-budget` | `aiop/cli.py`、`aiop/agent/search.py` | 有界候选计划与 `frontier.json` |
| 新增契约/技能层 | `aiop/agent/contract.py`、`aiop/agent/skills.py` | `contract.json`、`skill.lock`、`context_manifest.json` |
| 新增证据/审计层 | `aiop/agent/evidence.py`、`aiop/agent/audit.py` | `evidence.jsonl`、`diagnosis.json`、`lowering_audit.json` |
| 仿真多场景化 | `aiop/simulation/testgen.py` | 结构化 `shape`/`datatype` 用例；`make_inputs_aiop(case, device)`；逐用例精度/性能指标 |
| 用例矩阵与 kernel 变体 | `aiop/simulation/casematrix.py` | manifest workload×dtype 枚举；每组合独立 `build_N`；覆盖率与 unsupported 报告 |
| 统一进度日志 | `aiop/agent/progress.py`、`orchestrator.py`、`service/server.py` | `run.log`、终端 stderr 逐行、`GET /v1/tasks/{id}/log` |
| HTTP 命令提交 | `aiop/service/server.py` | `POST /v1/commands`、`GET /v1/jobs`、`/v1/jobs/{id}/log` |
| 可搬运测试导出 | `aiop/simulation/testgen.py`、`aiop/cli.py` | `aiop export` → 内嵌 kernel 的单文件 `test_<op>.py` |

## 12. 知识库与多代理

| 文档 | 内容 |
|---|---|
| [00 README](knowledge/README.md) | 索引、检索协议、上游路径映射 |
| [01 平台](knowledge/01-ascend950-platform.md) | 950/3510 架构、AIC/AIV、内存层级、dtype、GEMM/对齐约束 |
| [02 DSL](knowledge/02-tilelang-ascend-dsl.md) | Ascend dialect API 速查 |
| [03 编译调试](knowledge/03-compile-run-debug.md) | 构建、lower、IR trace、profiling、测试 |
| [04 样例骨架](knowledge/04-examples-and-patterns.md) | 官方示例地图 + 可复用骨架 |
| [05 限制](knowledge/05-limits-and-pitfalls.md) | 语言/向量化拒绝点/自检清单 |
| [06 性能](knowledge/06-performance-optimization.md) | Roofline、容量硬数字、搬运/流水/SIMD、autotune 边界 |
| [07 仿真与精度](knowledge/07-simulation-and-accuracy.md) | 五层验证、npusim、精度门限、失败分层 |
| [08 上板](knowledge/08-board-deployment.md) | SSH 部署、指标、门禁、失败模式 |
| [09 样例库](knowledge/09-verified-samples.md) | 已验证样例索引与验证状态 |
| [10 manifest 规格](knowledge/10-manifest-spec.md) | 契约绑定规则、可生成 family、`blocked_unsupported` |
| [11 Trae draft](knowledge/11-trae-draft-mode.md) | 无 Key 场景的 prompt → draft → 门禁 → `FIX_REQUEST.md` |

检索：`python -m aiop search "UB 容量 预算"`。索引自动构建到 `.index/knowledge.json`，
覆盖 `knowledge/*.md`、`.agent/skills/*/SKILL.md`、skill references、`.opencode/agents/*.md`、
`samples/*.md`；按文件 mtime 失效，tokenizer 支持中文 bigram 与英文标识符。

多代理体系（[.opencode/agents/](.opencode/agents/README.md)）：

- `aiop-op-orchestrator`（primary）：六阶段状态机、门禁、重试、设计回退
- `aiop-op-designer`：需求 → DESIGN.md
- `aiop-op-implementer`：生成 + 静态 + 编译修复
- `aiop-op-simulator`：npusim/local NPU/静态降级
- `aiop-op-board-runner`：SSH 上板
- `aiop-op-tuner`：可选性能调优

Skill：[.agent/skills/aiop-op-pipeline/SKILL.md](.agent/skills/aiop-op-pipeline/SKILL.md)。

## 13. 测试与基准

```bash
python -m pytest -q              # 124 个测试（静态/规划/manifest/open/draft/契约/搜索/审计/进度/HTTP/导出/端到端）
python -m pytest -q -m slow      # 仅端到端（会调用 bisheng）

python -m aiop eval --cases aiop/eval/cases.yaml --no-simulate \
  --out reports/eval_local.json

# 六步 campaign（契约 → 技能 → 搜索 → 门禁 → 仿真 → 上板）
python -m aiop campaign --spec examples/softmax.yaml --search-budget 3 --board
python -m aiop campaign --op GemmFwdOp --workload 1 --dtype-case 0 --out reports/campaign_gemm.md
```

- 基准用例：36 个（28 个手写 TaskSpec + 8 个 manifest 绑定），覆盖尾部/素数/非 2 幂形状。
- 最近一次本机结果（`--no-simulate`）：静态 36/36、编译 36/36、总耗时 ~71s，
  详见 [reports/eval_manifest.md](reports/eval_manifest.md)。
- `samples/` 下的 8 个算子样例与 [samples/index.json](samples/index.json) 由
  `scripts/build_samples.py` 生成/维护。

## 14. 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `blocked_environment`，错误含 `unknown type name 'asc_'` | CANN < 9.2，缺 A5 头/指令 | 升级 CANN 到 9.2+，并核对 `cann.home` |
| 错误含 `floatn.h` / `__float128` | 主机头文件与设备编译冲突 | 检查 `compile.device_flags` 与 CANN 版本 |
| `category: capacity` | tile/rows/num_stages 超 UB/L1/L0C | 降低 `tile`/`rows`，或让模板走 guarded 路径 |
| `blocked_unsupported` | family 无模板且未配置 LLM/draft | 配 `AIOP_LLM_*`，或 `aiop prompt` + `--strategy draft` |
| `draft not found` | 任务目录无 `draft.py` 且未给 `--draft` | 先跑 `aiop prompt`，或显式 `--draft <path>` |
| `draft has no build()` | 入口名不是 `build()` | 按 `PROMPT.md` 提供 `def build()` 返回 `@T.prim_func` |
| 静态/编译失败但没有 `FIX_REQUEST.md` | 配置了 LLM（走 repair）或属 generator 错误 | 检查 `attempts`/`debug_log.md`；draft 缺失时状态为 `blocked_generate` |
| `verified` 一直是 `False` | open 模式自指 golden，或本机 L3 不可用 | 补内置 golden 或上板实测；看 `golden_source` |
| L3 `unavailable` | 无 torch_npu / 无 `/dev/davinci*` / npusim 缺失或版本不匹配 | 安装与 CANN 匹配的 torch_npu；或接受 `success_static_only` |
| L4 `skipped` / `blocked` | `board.host` 为空 / SSH 预检失败 | 配置 `AIOP_BOARD_HOST`，确认免密登录与 `npu-smi` |
| `op 'XXX' not found in manifest` | manifest 路径不对或条目名拼错 | `aiop manifest list`，确认 `paths.tileops_manifest` |
| `BindError`（type family / complex / 缺参数） | manifest 语义超出当前绑定能力 | 看错误中点名的条目/参数；改用其它 workload 行 |
| 改了 shape 后找不到旧任务 | 任务 id 由 spec 哈希生成 | 用 `--task-id` 固定目录，或去新目录重跑 |
| `tilelang was already imported from ...` | 进程内先导入了另一个 tilelang checkout | 用新进程运行；确认 `paths.tilelang_repo` 指向 A5 原生仓 |
| 知识检索结果陈旧 | 索引按 mtime 指纹失效 | 删除 `.index/knowledge.json` 后重跑，或让它自动重建 |
| HTTP 401 | 设置了 `AIOP_SERVICE_TOKEN` | 加 `Authorization: Bearer <token>` |
| 端口被占用 | 默认 8790 已有服务 | `--port` 换端口 |
| `frontier.json` 一直选 base/高分候选 | 无 L3/L4 时只能按模型评分排序 | 在可用设备上跑 `aiop simulate`/`board` 后用实测数据复核 |
| `campaign` 报 `negative: failed` | 静态门禁没有拒绝注入的 `T.device_assert` | 检查 `SEM004` 规则是否被改动，属于门禁回归 |
| `contract.json` 里 `perf.enforced=false` | spec 未设置 `perf.target` | 在 TaskSpec 中显式给出 `perf: {metric: latency, target: ..., unit: ms}` |
| 终端看不到逐行进度 | 用了 `--no-progress`，或 stderr 被重定向 | 去掉开关；`tail -f workspace/<task>/run.log` 可补看 |
| HTTP 任务看不到进度 | 进度打印在服务端终端，不在响应里 | 服务端 `tail -f` 或 `GET /v1/tasks/{id}/log` |

## 15. 已知限制

1. 动态 shape、多输出 kernel、attention 模板不在确定性生成范围；配置 LLM 或 draft 后可起草，
   但只保证 L0–L2，不保证数值正确。
2. L3 依赖本机 torch_npu 与 CANN 版本匹配；不匹配时只做 L1/L2，并标记 `unavailable`。
3. L4 需要可 SSH 的 A5 主机与一致的工具链（`board.tilelang_repo` / `board.env_script`）。
4. 性能目标未配置时只记录数据（`roofline` / 时延），不做绝对峰值利用率断言。
5. `.runtime/` 是可选的本地 torch_npu 安装目录（`npusim` 探测用），已被 `.gitignore` 忽略，可安全删除。
6. manifest 中依赖代数数据类型 type family 的条目（MoE 的 `layout`）和 `complex64/128`（FFT）
   不绑定，会以 `BindError` 明确报出。
7. open 模式的 golden 由 LLM 生成时属于自指参考，`verified` 恒为 `False`；
   要拿到独立证据必须补 aiop 内置参考或上板实测。
8. draft 模式不会自动多轮重试：失败即停并写 `FIX_REQUEST.md`，由外部 Agent/人修正后重跑。
9. `--strategy draft` 作用于有模板 family 时，`mode` 仍为 `template`（沿用内置 golden），
   只有无模板 family 才进入 `mode="open"`。
10. 计划搜索（`--search`）的排序依据是**模型评分**，没有真机测量时不能当成 autotune
    结论；`frontier.json` 明确标注 `objective: model score`。
11. SkillEvolutionService / CompilerEvolutionService / CompatibilityGate 与 ReleaseRegistry
    尚未落地（映射见第 17 节），当前技能与编译器规则仍靠人工维护。

## 16. 版本与变更记录

| 版本标记 | 对应提交 | 主要内容 |
|---|---|---|
| `v0.1.0`（基线） | `3f7d3ca` Initial commit | 7 家族模板、规划器、L0–L4 流水线、HTTP 服务、28 用例基准、知识库 01–09 |
| `v0.1.0+manifest/llm` | `242a6ad` `feat:llm` | TileOPs manifest 绑定（205 条目/23 映射）、LLM open 模式与 repair、GEMM 非 fp32 输出 cast、8 个 manifest 基准用例（36 总数）、知识库 10、`examples/spec/` 25 个 manifest YAML（含 `types.yaml`） |
| `v0.1.0+draft`（工作区，未提交） | — | `prompt` 命令、`--strategy draft`/`--draft`、`FIX_REQUEST.md`、`golden_draft.py`、知识库 11 |
| `v0.1.0+campaign`（工作区，未提交） | — | TaskContract、SkillRegistry、有界计划搜索、LoweringAudit、EvidenceStore/Diagnoser、多场景仿真 + 负例门禁、`campaign` 命令、统一 `run.log` 与终端进度、HTTP `/v1/commands` + `/v1/jobs`、`export` 单文件测试；测试 71 → 90 |

> `pyproject.toml` 当前版本号仍为 `0.1.0`；上表用提交批次标记功能适用版本，便于对应代码状态。

## 17. Campaign 架构与 some_ref.md 映射

本次把 aiop 从「一次性生成脚本」升级为可审计的 campaign 流水线：

```text
TaskContract → SkillBundle → 候选计划 → 静态/编译门禁 → LoweringAudit
            → 多场景仿真 + 负例门禁 → EvidenceStore/Diagnoser → campaign_report.md
```

| `some_ref.md` 模块 | aiop 落地 | 产物 | 状态与边界 |
|---|---|---|---|
| TaskRegistry | [agent/contract.py](aiop/agent/contract.py) | `contract.json` | 语义/ABI/精度/性能目标/资源上限/检查义务 |
| CampaignSupervisor | `OperatorAgent` + `--search-budget` | `state.json` / `frontier.json` | 单任务阶段与预算；无多 campaign 调度/设备池 |
| SkillRegistry 与 SkillAdapter | [agent/skills.py](aiop/agent/skills.py) | `skill.lock` / `context_manifest.json` | 技能选择、sha256 锁定、义务映射 |
| Planner 与 SearchController | [agent/search.py](aiop/agent/search.py) | `frontier.json` | 族感知候选 + 门禁；评分为模型估计 |
| Implementer | [generation/](aiop/generation/) | `<op>.py` / `history/` | 模板 / LLM / 外部 draft |
| AutoTuneAdapter | `SearchController` 候选 | `frontier.json` | tile/grid/threads/stages/GEMM block；未接 `T.Stage`/latency 标注 |
| CompilerBridge | [checks/compiler.py](aiop/checks/compiler.py) + `compile_worker` | `compile_result.json` / `.cce.c` | 隔离 lowering + bisheng |
| IndependentEvaluator | [simulation/testgen.py](aiop/simulation/testgen.py) + [simulation/casematrix.py](aiop/simulation/casematrix.py) | `simulation_report.json` | shape/datatype 全组合；逐用例精度/性能；负例门禁 `SEM004` |
| EvidenceStore 与 Diagnoser | [agent/evidence.py](aiop/agent/evidence.py) | `evidence.jsonl` / `diagnosis.json` | 追加事实 + 失败聚类 + 回路路由 |
| LoweringAudit | [agent/audit.py](aiop/agent/audit.py) | `lowering_audit.json` | 计划声明 vs lowering 可观测事实 |
| SkillEvolutionService | — | — | 未落地：技能仍为静态 markdown |
| CompilerEvolutionService | — | — | 未落地 |
| CompatibilityGate 与 ReleaseRegistry | — | — | 未落地 |

六步流程与产物的对应：

| 步骤 | 产物 |
|---|---|
| 1. 规格与契约 | `contract.json`（含性能目标是否强制、资源余量） |
| 2. 知识/技能匹配 | `skill.lock`、`context_manifest.json`、`DESIGN.md` 的知识召回 |
| 3. 初步实现 | `<op>.py`、`frontier.json`、`compile_result.json` |
| 4. 仿真验证 | `simulation_report.json`（多场景）、`negative_probe.json` |
| 5. 上板实测 | `board_report.json`（无 host 时明确 `skipped`） |
| 6. 设计融合 | `campaign_report.md` + 本节映射表 |

## 18. 参考与复用

- A3 多代理逻辑：`tilelang-ascend` 仓的 `.opencode/agents`
- Skill：`cannbot-skills`（tilelang-op-*、ops-simulator、ops-precision-standard 等）
- A5 接口权威：`TILELANG_ROOT`（`tilelang/ascend/README.md`、`examples/ascend/`）
- TileOPs 契约：`TILEOPS_MANIFEST`（设计文档 `docs/design/manifest.md`）
- 设计要求：[some_ref.md](some_ref.md)

本仓不修改上述参考仓；`scripts/sync_skills.sh` 用于从上游同步 skill 副本，
`scripts/build_samples.py` 用于重新生成 `samples/` 与索引。
