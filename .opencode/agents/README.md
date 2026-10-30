---
disable: true
---

# aiop — Ascend A5 算子生成代理体系

面向 **Ascend 950PR/DT（`dav-3510`，Atlas A5）+ TileLang Ascend dialect** 的端到端算子生成
代理体系。由 a3 版 `tilelang-op-orchestrator` 三阶段状态机迁移而来，新增**知识库检索**、
**设备编译门禁**、**npusim 仿真门禁**与**跨服务器上板实测**。

## 与 a3 版本的关系

| a3（tilelang-ascend/.opencode） | a5（本目录） | 迁移点 |
|---|---|---|
| `tilelang-op-orchestrator` | `aiop-op-orchestrator` | 三阶段 → 六阶段；新增仿真/上板门禁；状态机由 `aiop` CLI 落地 |
| `tilelang-op-analyst` | `aiop-op-designer` | 增加 A5 容量事实、知识库召回、A5 dtype/尾部约束 |
| `tilelang-op-developer` | `aiop-op-implementer` | 模板优先 + LLM 可选；静态/编译分类修复 |
| `tilelang-op-perf-tuner` | `aiop-op-tuner` | 增加 roofline、tile/stages/cache 搜索边界 |
| — | `aiop-op-simulator` | npusim / local NPU / static fallback |
| — | `aiop-op-board-runner` | SSH 部署、实测、指标采集 |

## 拓扑

```text
                 ┌──────────────────────────────┐
                 │ aiop-op-orchestrator (primary)│
                 │ 状态机 · 门禁 · 重试 · 报告    │
                 └──┬───────┬───────┬───────┬────┘
                    │       │       │       │
              ┌─────▼──┐ ┌──▼────┐ ┌▼──────┐ ┌▼──────────────┐
              │designer│ │implementer│ │simulator│ │board-runner │
              │  S1    │ │ S2–S4  │ │  S5   │ │  S6          │
              └────────┘ └────────┘ └───────┘ └──────────────┘
                    │       │        │            │
                    └───────┴────────┴────────────┘
                         可选：aiop-op-tuner (S7)
```

## 六阶段

| Stage | 名称 | 负责 Agent | 产物 | 门禁 |
|---|---|---|---|---|
| 0 | 环境/需求预检 | orchestrator | `env.json` | CANN 9.2、tilelang、规格齐全 |
| 1 | 设计 | designer | `DESIGN.md`、`spec.yaml` | 11 章节 + 容量预算 |
| 2 | 生成 | implementer | `{op}.py`、`test_{op}.py` | 语法/AST |
| 3 | 静态检查 | implementer | `static_report.json` | 无 error |
| 4 | 设备编译 | implementer | `compile_result.cce.c` | `bisheng` 通过 |
| 5 | 仿真 | simulator | `simulation_report.json` | 精度通过；不可用需如实标记 |
| 6 | 上板 | board-runner | `board_report.json` | 精度 + 性能门禁 |
| 7 | 调优（可选） | tuner | `perf_tuning/` | 不退化 + 目标达成 |

## 可执行入口

Markdown agent 负责流程与判断；**实际执行由 `aiop` 包完成**，两者共享同一套门禁：

```bash
python -m aiop env --probe-compile
python -m aiop generate --spec case.yaml [--simulate/--no-simulate] [--board]
python -m aiop check|compile|simulate|board workspace/<task>
python -m aiop eval --cases aiop/eval/cases.yaml
python -m aiop serve --host 0.0.0.0 --port 8790
```

## 关键机制

### 状态文件

`workspace/{task_id}/state.json` 仅 orchestrator 可写；schema 见
`aiop/agent/state.py`。Subagent 只返回阶段结果，不修改全局状态。

### 设计回退

implementer 发现设计层错误（API 不存在、容量不可行、内存层级违规、同步模式冲突）时返回
`[DESIGN_ERROR]`；orchestrator 备份 DESIGN.md 到 `history/design_rev{N}.md` 并回退 Stage 1。
设计回退不计入 Stage 2 重试次数，但受 Stage 2 总尝试上限兜底。

### 重试与中止

| Stage | 上限 | 超限状态 |
|---|---|---|
| 1 设计 | 3 | `BLOCKED_DESIGN` |
| 2–4 生成/静态/编译 | 3 次尝试 | `BLOCKED_IMPL` / `BLOCKED_COMPILE` |
| 5 仿真 | 2 次尝试 | `BLOCKED_ACCURACY` |
| 6 上板 | 1 次（不自动重试设备问题） | `BLOCKED_BOARD` |
| 环境 | 不重试 | `BLOCKED_ENVIRONMENT` |

### 诚实性约束

- 未实际执行的编译/仿真/上板不得写成“通过”。
- 仿真后端不可用时标记 `unavailable`，报告 `success_static_only`，并说明缺口。
- 性能数据必须附测量协议（warmup/repeats/设备/revision）；仿真时延不算真机性能。
- 不修改 `tilelang` / `cannbot-skills` / `tilelang-ascend` 等参考仓。

## 产物目录

```text
workspace/{task_id}/
├── DESIGN.md / spec.yaml
├── {op}.py / test_{op}.py
├── static_report.json
├── compile_result.json / compile_result.cce.c
├── simulation_report.json
├── board_report.json
├── result.json / report.md / manifest.json
├── state.json / debug_log.md
└── history/
```

## 使用示例

```text
@aiop-op-orchestrator 生成一个 softmax 算子，输入 [128, 1000] float32，
沿最后一维归一化，精度 1e-5，先在本机编译+仿真，仿真通过后提交到 A5 服务器实测。
```

Orchestrator 会先补齐缺失字段（算子名、公式、I/O、精度、性能目标、是否上板），
再按 Stage 1–6 推进；中断后用同一 task_id 续跑。
