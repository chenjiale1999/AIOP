# aiop — Ascend A5 × TileLang 算子自动生成 Agent

面向 **Ascend 950PR/DT（Atlas A5，`dav-3510`）** 的端到端算子生成系统：

```text
需求 / TaskSpec
  → 知识召回 + 设计 (DESIGN.md)
  → TileLang A5 kernel + 自包含测试
  → 静态检查 (AST / API / 容量 / 语义)
  → 设备编译 (tilelang.lower + bisheng dav-3510)
  → 仿真验证 (npusim / local NPU / static fallback)
  → 跨服务器上板实测 (SSH)
  → 报告归档 (result.json / report.md)
```

## 快速开始

```bash
cd /home/cjl/workspace/aiop

# 0. 环境探测（A5 编译需要 CANN 9.2+）
python -m aiop env --probe-compile

# 1. 生成 + 编译 + 仿真（本机无仿真后端时自动降级并如实标记）
python -m aiop generate --requirement "softmax on [128, 1000] float32" 

# 2. 用 TaskSpec 精确控制
python -m aiop generate --spec examples/softmax.yaml

# 2b. 直接用 TileOPs manifest 契约（src/tileops/manifest/spec/*.yaml）
python -m aiop manifest list
python -m aiop manifest show MLAPagedFwdOp
python -m aiop generate --op RMSNormFwdOp --workload 0 --dtype-case 1

# 2c. 没有模板的算子：配置 LLM 后走 open 模式（门禁不放松）
export AIOP_LLM_BASE_URL=https://api.deepseek.com/v1
export AIOP_LLM_MODEL=<model id>
export AIOP_LLM_API_KEY=<key>
python -m aiop generate --op GQADenseFwdOp --workload 0

# 3. 已有任务重跑门禁
python -m aiop check    workspace/<task>
python -m aiop compile  workspace/<task>
python -m aiop simulate workspace/<task>
python -m aiop board    workspace/<task> --host user@a5-host

# 4. 基准评测（36 个算子用例，含 8 个 manifest 绑定用例）
python -m aiop eval --cases aiop/eval/cases.yaml

# 5. HTTP 服务
python -m aiop serve --host 0.0.0.0 --port 8790
```

## 架构

```text
aiop/
├── aiop/                     # 可执行 Agent 服务
│   ├── agent/                # 编排、状态机、设计文档
│   ├── generation/           # 模板生成、LLM 适配、修复
│   ├── checks/               # 静态检查、编译门禁
│   ├── simulation/           # 测试生成、npusim/local NPU
│   ├── board/                # SSH 部署与上板实测
│   ├── knowledge/            # BM25 知识检索
│   ├── eval/                 # 28 用例基准套件
│   ├── runtime/              # A5 TileLang 引导、硬件事实
│   ├── service/              # HTTP API
│   └── cli.py
├── knowledge/                # A5 知识库（01–09）
├── .agent/skills/            # 可加载 skill（含 aiop-op-pipeline）
├── .opencode/agents/         # A5 多代理定义（orchestrator + 5 subagents）
├── samples/                  # 已验证算子样例库
├── scripts/build_samples.py
├── tests/                    # 单元 + 端到端测试
├── config/aiop.yaml
└── workspace/                # 运行时任务产物
```

## 算子支持范围

| 家族 | 变体 | 状态 |
|---|---|---|
| `elementwise_unary` | relu/abs/exp/neg/sigmoid/tanh/rsqrt/sqrt/log/cast | 已验证编译 |
| `elementwise_binary` | add/sub/mul/div/maximum/minimum | 已验证编译 |
| `row_reduce` | sum/mean/max/min（沿最后一维） | 已验证编译 |
| `softmax` | 数值稳定（max-shift） | 已验证编译 |
| `rmsnorm` | 行级 reducer + 权重 | 已验证编译 |
| `layernorm` | 双 reducer + weight/bias | 已验证编译 |
| `gemm` / `gemm_relu` | 持久化 L1→L0C，可选 Cube→Vector epilogue（含非 fp32 输出 cast） | 已验证编译 |

不支持：动态 shape、attention（需专用模板）、非 A5 dtype、跨级 GM→L0。

### TileOPs manifest 接入

`src/tileops/manifest/spec/*.yaml` 的算子契约可以直接驱动本流程（详见
[knowledge/10-manifest-spec.md](knowledge/10-manifest-spec.md)）：

- `forall` 索引 / `params` / `optional` + `some` / `values` + `requires` /
  `mutated` / `contiguous` / 多输出 / `let` / `shape_rules` / `dtype_cases` /
  type family（`Mat[trans_b, K, N]`、`PoolOut[...]` 等）全部按 manifest 语义绑定；
- 行里省略的维度按统一规则求解（`B = len(seq)`、`prefix_sum(seq)` → `len+1`、
  `packed_positions/token_indices(seq)` → `sum(seq)`）；
- 全量自检：2208 个 workload×dtype 组合中 2125 个可绑定，`attention.yaml`
  235/235 绑定成功；
- 有模板的 23 个条目直接生成；其余（attention/MoE/conv/pool/FFT/量化）返回
  `blocked_unsupported` 并输出契约级 `DESIGN.md`，不生成假 kernel。

## 知识库

| 文档 | 内容 |
|---|---|
| [00 README](knowledge/README.md) | 索引、检索协议、上游路径映射 |
| [01 平台](knowledge/01-ascend950-platform.md) | 950/3510 架构、AIC/AIV、内存层级、dtype、GEMM/对齐约束 |
| [02 DSL](knowledge/02-tilelang-ascend-dsl.md) | Ascend dialect API 速查 |
| [03 编译调试](knowledge/03-compile-run-debug.md) | 构建、lower、IR trace、profiling、测试 |
| [04 样例骨架](knowledge/04-examples-and-patterns.md) | 官方示例地图 + 6 类可复用骨架 |
| [05 限制](knowledge/05-limits-and-pitfalls.md) | 语言/向量化拒绝点/自检清单 |
| [06 性能](knowledge/06-performance-optimization.md) | Roofline、容量硬数字、搬运/流水/SIMD、autotune 边界 |
| [07 仿真与精度](knowledge/07-simulation-and-accuracy.md) | 五层验证、npusim、精度门限、失败分层 |
| [08 上板](knowledge/08-board-deployment.md) | SSH 部署、指标、门禁、失败模式 |
| [09 样例库](knowledge/09-verified-samples.md) | 已验证样例索引与验证状态 |
| [10 manifest 规格](knowledge/10-manifest-spec.md) | TileOPs 契约绑定规则、可生成 family、`blocked_unsupported` |

检索：`python -m aiop search "UB 容量 预算"`。索引自动构建到 `.index/knowledge.json`，
按文件 mtime 失效；tokenizer 支持中文 bigram 与英文标识符。

## 验证层级与当前实测状态

| 层 | 内容 | 本机状态 |
|---|---|---|
| L0 | 语法/API/约束/语义 | ✅ 36/36 用例通过 |
| L1 | `tilelang.lower` 出 CCE 源码 | ✅ 36/36 |
| L2 | `bisheng --npu-arch=dav-3510` | ✅ 36/36 |
| L3 | npusim / local NPU 数值验证 | ⚠️ 本机无匹配 torch_npu/NPU；自动探测并标记 `unavailable` |
| L4 | 远端 A5 实测 | ⚪ 未配置 host；`board` 返回 `skipped` |

L3 需要 **与 CANN 版本匹配的 torch_npu**（例如 CANN 9.1 + torch_npu 2.9.x）以及可见的
`/dev/davinci*`；`npusim` 只接受 `-s Ascend950`/`Ascend950DT`。缺少条件时系统不会伪造通过，
而是在 `simulation_report.json` 与最终报告中披露缺口。

基准结果：`reports/eval_manifest.json`，生成（静态+编译）通过率 **100%（36/36）**，
其中 8 个用例直接绑定自 TileOPs manifest；每个用例的 `simulation.report` 均如实标记
`unavailable`。

## 配置

`config/aiop.yaml`：

```yaml
paths:
  tilelang_repo: /home/cjl/workspace/tilelang   # A5 原生仓
cann:
  home: /usr/local/Ascend/cann-9.2.0            # A5 头文件/指令
  soc_version: Ascend950
board:
  host: ""                                      # 为空则跳过 L4
  remote_dir: /tmp/aiop_runs
  python: python3
  env_script: ""
agent:
  max_attempts: 3
  llm:
    enabled: false                              # 配置 base_url/model 后启用
```

环境变量覆盖：`AIOP_TILELANG_REPO`、`AIOP_CANN_HOME`、`AIOP_BOARD_HOST`、
`AIOP_BOARD_KEY`、`AIOP_LLM_BASE_URL`、`AIOP_LLM_MODEL`、`AIOP_SERVICE_TOKEN`。

### LLM 模式

默认使用**已验证模板**，无需网络。配置任意 OpenAI-compatible 端点（DeepSeek、vLLM、网关均可）后：

- `--strategy auto`（默认）：**有模板的 family 仍走模板**，没模板的 family 自动走 LLM 开放模式；
- `--strategy llm`：强制先让模型起草，失败再回退模板；
- `--strategy template`：完全离线、确定性。

LLM 只负责“写 kernel / 修 kernel”，不参与判定：静态、lowering、bisheng 设备编译、
精度门禁全部照旧，任何候选不通过就带着错误回到修复循环（`agent.max_attempts` 轮）。

开放模式的可信度边界：

| 项 | 行为 |
|---|---|
| plan | `variant="llm_open"`，只固定 grid ≤ 64（Cube ≤ 32），tiling/UB 不做断言 |
| golden | 有 `role=metadata` 输入时不生成（L3 = `unavailable`）；否则 LLM 写纯 torch 参考，标记 `llm-self-referenced` |
| `verified` | 恒为 `False`：自指参考不算独立证据 |
| 状态 | 最多 `success_static_only`（L0–L2 通过） |

## HTTP API

```text
GET  /healthz
GET  /v1/env
POST /v1/search                 {"query": "...", "k": 5}
POST /v1/tasks                  {"spec": {...}} 或 {"requirement": "..."}
GET  /v1/tasks
GET  /v1/tasks/{id}
GET  /v1/tasks/{id}/report
POST /v1/tasks/{id}/board       {"host": "user@a5-host"}
```

示例：

```bash
curl -s localhost:8790/v1/tasks -H 'Content-Type: application/json' \
  -d '{"requirement":"rmsnorm on [128, 4096] float32","simulate":true}'
```

## 测试

```bash
python -m pytest -q                 # 69 个测试（golden 执行、修复循环、manifest 绑定、open 模式、端到端编译）
python -m pytest -q -m slow         # 仅端到端（会调用 bisheng）
```

## 多代理体系

[`.opencode/agents/`](.opencode/agents/README.md)：

- `aiop-op-orchestrator`（primary）：六阶段状态机、门禁、重试、设计回退
- `aiop-op-designer`：需求 → DESIGN.md
- `aiop-op-implementer`：生成 + 静态 + 编译修复
- `aiop-op-simulator`：npusim/local NPU/静态降级
- `aiop-op-board-runner`：SSH 上板
- `aiop-op-tuner`：可选性能调优

Skill：[`.agent/skills/aiop-op-pipeline/SKILL.md`](.agent/skills/aiop-op-pipeline/SKILL.md)。

## 参考与复用

- A3 多代理逻辑：`/home/cjl/workspace/tilelang-ascend/.opencode/agents`
- Skill：`cannbot-skills`（tilelang-op-*、ops-simulator、ops-precision-standard 等）
- A5 接口权威：`/home/cjl/workspace/tilelang`（`tilelang/ascend/README.md`、`examples/ascend/`）
- 设计要求：`/home/cjl/workspace/aiop/some_ref.md`

本仓不修改上述参考仓；`scripts/sync_skills.sh` 用于从上游同步 skill 副本。

## 已知限制

1. 动态 shape / 多输出 kernel / attention 模板暂不在确定性生成范围；配置 LLM 后可走
   open 模式起草（只保证 L0–L2，不保证数值正确）。
2. L3 数值仿真依赖本机 torch_npu 与 CANN 的版本匹配；不匹配时只做 L1/L2。
3. L4 需要用户提供可 SSH 的 A5 主机与一致的工具链。
4. 性能目标未配置时只记录数据，不做绝对峰值利用率断言。
5. `.runtime/` 是可选的本地 torch_npu 安装目录（探测用），已被 `.gitignore` 忽略，可安全删除。
6. manifest 中依赖代数数据类型 type family 的条目（MoE 的 `layout`）和 `complex64/128`
   （FFT）不绑定，会以 `BindError` 明确报出。
7. open 模式的 golden 由 LLM 生成时属于自指参考，`verified` 恒为 `False`；要拿到独立
   证据必须补 aiop 内置参考或上板实测。

## 验收对照

| 要求 | 落地 |
|---|---|
| 复用参考 Skill / A3 Agent 逻辑 | `.opencode/agents/` 6 个 A5 Agent；`.agent/skills/` 8 个 skill（含 `aiop-op-pipeline`） |
| A5 知识库 | `knowledge/01–10`，覆盖语法/硬件/性能/样例/仿真/上板/契约；BM25 实时检索 |
| 端到端生成 | 7 个家族、36 个 benchmark 用例（含 8 个 manifest 绑定用例），静态+编译 100% |
| 规格输入 | TaskSpec YAML/JSON、自然语言、TileOPs manifest（`--op/--workload/--dtype-case`） |
| LLM 能力 | OpenAI 兼容端点：无模板 family 的 open 模式起草 + 编译失败 repair 循环 |
| 错误自修正 | 静态/编译分类 + 保守 plan 回退 + 可选 LLM 修复 |
| 仿真验证 | `npusim`/`local_npu`/`static` 三后端，精度混合容差，失败触发重生成 |
| 跨服务器上板 | SSH 预检/部署/运行/指标/Profile 命令，生成 `board_report.json` |
| 可运行服务 | `python -m aiop serve` + HTTP API + CLI + pytest |
