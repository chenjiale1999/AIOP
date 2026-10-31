"""Campaign report: the six-step, end-to-end view of one operator run.

Renders a single markdown document from ``result.json`` plus the archived
contract / skills / frontier / audit / diagnosis artifacts. It deliberately
separates *executed* evidence from *unavailable* layers instead of collapsing
everything into a pass/fail headline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..checks.static import run_static
from ..config import Config
from ..runtime.hardware import Plan
from ..specs import TaskSpec


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _arch_map() -> list[list[str]]:
    return [
        ["TaskRegistry", "`aiop/agent/contract.py`", "`contract.json`", "已落地：语义/ABI/精度/性能目标/资源上限/检查义务"],
        ["CampaignSupervisor", "`OperatorAgent` + `--search-budget`", "`state.json` / `frontier.json`", "部分落地：单任务阶段与预算；无多 campaign 调度/设备池"],
        ["SkillRegistry 与 SkillAdapter", "`aiop/agent/skills.py`", "`skill.lock` / `context_manifest.json`", "已落地：技能选择、内容锁定、义务映射"],
        ["Planner 与 SearchController", "`aiop/agent/search.py`", "`frontier.json`", "已落地：族感知候选 + 静态/编译门禁；评分为模型估计"],
        ["Implementer", "`aiop/generation/`", "`<op>.py` / `history/`", "已有：模板 / LLM / 外部 draft"],
        ["AutoTuneAdapter", "`SearchController` 候选", "`frontier.json`", "部分落地：tile/grid/threads/stages/GEMM block；未接 T.Stage/latency 标注"],
        ["CompilerBridge", "`aiop/checks/compiler.py` + `compile_worker`", "`compile_result.json` / `.cce.c`", "已有：隔离 lowering + bisheng"],
        ["IndependentEvaluator", "`aiop/simulation/testgen.py`", "`simulation_report.json`", "已落地：正常/零/一/大/小 多场景；异常场景由静态负例覆盖"],
        ["EvidenceStore 与 Diagnoser", "`aiop/agent/evidence.py`", "`evidence.jsonl` / `diagnosis.json`", "已落地：追加事实 + 失败聚类 + 回路路由"],
        ["LoweringAudit", "`aiop/agent/audit.py`", "`lowering_audit.json`", "已落地：计划声明 vs lowering 可观测事实"],
        ["SkillEvolutionService", "—", "—", "未落地：技能仍为静态 markdown"],
        ["CompilerEvolutionService", "—", "—", "未落地"],
        ["CompatibilityGate 与 ReleaseRegistry", "—", "—", "未落地"],
    ]


def run_negative_probe(
    task_dir: str | Path,
    spec: TaskSpec,
    plan: Plan,
    cfg: Config,
) -> dict[str, Any]:
    """Prove the negative gate is live: a deliberately broken copy must be rejected.

    This is the error-scenario half of the evaluator. It mutates the delivered
    kernel with a forbidden debug API (``SEM004``) in a separate directory and
    asserts the static gate refuses it; the delivered kernel is never touched.
    """
    task_dir = Path(task_dir)
    kernel = task_dir / f"{spec.name}.py"
    if not kernel.is_file():
        return {"status": "skipped", "reason": "no delivered kernel to mutate"}
    source = kernel.read_text(encoding="utf-8")
    marker = "def build():"
    if marker not in source:
        return {"status": "skipped", "reason": "build() entry not found"}
    bad_dir = task_dir / "negative"
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad_path = bad_dir / f"bad_{spec.name}.py"
    bad_path.write_text(
        source.replace(marker, marker + "\n    T.device_assert(False)", 1),
        encoding="utf-8",
    )
    report = run_static(spec, plan, bad_path, cfg)
    rules = sorted({f.rule for f in report.findings if f.severity == "error"})
    status = "passed" if (not report.ok and "SEM004" in rules) else "failed"
    data = {
        "status": status,
        "case": "forbidden_debug_api",
        "expected_rule": "SEM004",
        "observed_rules": rules,
        "static_ok": report.ok,
        "bad_kernel": str(bad_path),
    }
    (task_dir / "negative_probe.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return data


def render_campaign_report(
    result: dict[str, Any], negative: dict[str, Any] | None = None
) -> str:
    spec = result.get("spec") or {}
    plan = result.get("plan") or {}
    contract = result.get("contract") or {}
    skills = result.get("skills") or {}
    sim = result.get("simulation") or {}
    board = result.get("board") or {}
    diagnosis = result.get("diagnosis") or {}
    audit = result.get("lowering_audit") or {}
    lines: list[str] = [
        f"# {spec.get('name', '?')} — aiop Campaign 报告",
        "",
        f"- status: **{result.get('status')}** · verified: `{result.get('verified')}`",
        f"- mode: `{result.get('mode')}` · generator: `{result.get('generator')}` · golden: `{result.get('golden_source')}`",
        f"- task: `{result.get('task_id')}` · elapsed: `{result.get('elapsed_s')}s`",
        "",
        "## 1. 算子契约（TaskRegistry）",
        "",
    ]
    if contract:
        perf = contract.get("perf") or {}
        target = perf.get("target")
        target_text = "未设置（只记录不判定）" if target is None else f"`{target}{perf.get('unit', '')}`"
        lines += _table(
            ["项", "值"],
            [
                ["contract hash", f"`{contract.get('contract_hash')}`"],
                ["来源", f"`{contract.get('source')}`"],
                ["语义", f"`{contract.get('semantics')}`"],
                ["确定性", f"`{contract.get('determinism')}`（open_mode={contract.get('open_mode')}）"],
                ["精度", f"`{json.dumps(contract.get('precision', {}))}`"],
                ["性能目标", f"{perf.get('metric')}={target_text} · enforced=`{perf.get('enforced')}`"],
                ["检查义务", ", ".join(f"`{item}`" for item in contract.get("obligations", [])) or "—"],
            ],
        )
        lines += ["", "### ABI", ""]
        abi_rows = []
        for tensor in contract.get("inputs", []) + contract.get("outputs", []):
            abi_rows.append(
                [
                    tensor.get("direction"),
                    tensor.get("name"),
                    tensor.get("shape"),
                    f"`{tensor.get('dtype')}`",
                    tensor.get("role"),
                    tensor.get("nbytes"),
                ]
            )
        lines += _table(["方向", "张量", "形状", "dtype", "role", "bytes"], abi_rows)
        limits = contract.get("limits") or {}
        facts = contract.get("plan_facts") or {}
        headroom = contract.get("headroom") or {}
        lines += ["", "### 资源上限与计划占用", ""]
        lines += _table(
            ["资源", "上限", "计划占用", "余量"],
            [
                ["UB", limits.get("ub_bytes"), facts.get("ub_used"), headroom.get("ub_bytes")],
                ["L1", limits.get("l1_bytes"), facts.get("l1_used"), headroom.get("l1_bytes")],
                ["L0C", limits.get("l0c_bytes"), facts.get("l0c_used"), headroom.get("l0c_bytes")],
                ["grid", f"AIV {limits.get('aiv_cores')} / AIC {limits.get('aic_cores')}", facts.get("blocks"), "—"],
            ],
        )
    else:
        lines += ["- contract 不可用（任务在契约阶段前终止）", ""]

    lines += ["## 2. 可复用技能与知识资产（SkillRegistry）", ""]
    if skills:
        reasons = skills.get("reasons") or {}
        rows = [
            [
                f"`{item.get('name')}`",
                reasons.get(item.get("name"), "—"),
                f"`{str(item.get('sha256', ''))[:12]}`",
            ]
            for item in skills.get("skills", [])
        ]
        lines += _table(["技能", "选择原因", "内容锁定"], rows)
        if skills.get("missing"):
            lines += ["", f"- 缺失技能：{', '.join(skills['missing'])}"]
        lines += [
            "",
            f"- 检查义务：{', '.join(f'`{item}`' for item in skills.get('obligations', [])) or '—'}",
            "",
        ]
    else:
        lines += ["- 技能选择不可用", ""]
    hits = result.get("knowledge") or []
    if hits:
        lines += ["知识召回：", ""]
        lines += _table(
            ["文档", "章节", "score"],
            [[f"`{h.get('doc')}`", h.get("heading"), h.get("score")] for h in hits[:5]],
        )
        lines += [""]

    lines += ["## 3. 初步实现", ""]
    geometry = (
        f"rows={plan.get('rows')} / extent={plan.get('tile_extent')} / threads={plan.get('threads')}"
        if plan.get("rows")
        else f"blocks={plan.get('blocks')} / tile={plan.get('tile')} / threads={plan.get('threads')}"
    )
    lines += _table(
        ["项", "值"],
        [
            ["generator", f"`{result.get('generator')}`"],
            ["plan variant", f"`{plan.get('variant')}`"],
            ["几何", geometry],
            ["stages / iterations", f"{plan.get('num_stages')} / {plan.get('iterations')}"],
            ["GEMM blocks", f"m={plan.get('block_m')} n={plan.get('block_n')} k={plan.get('block_k')}"],
            ["epilogue / out_cast", f"`{plan.get('epilogue')}` / `{plan.get('out_cast')}`"],
            ["kernel", f"`{(result.get('artifacts') or {}).get('kernel', '')}`"],
        ],
    )
    attempts = result.get("attempts") or []
    if attempts:
        lines += ["", "门禁尝试：", ""]
        lines += _table(
            ["#", "阶段", "ok", "category", "error"],
            [
                [
                    item.get("attempt"),
                    item.get("stage"),
                    item.get("ok"),
                    item.get("category", "—"),
                    (item.get("error") or "")[:120].replace("|", "\\|"),
                ]
                for item in attempts
            ],
        )
    lines += [""]

    lines += ["## 4. 仿真验证（IndependentEvaluator）", ""]
    if sim:
        lines += [
            f"- status: `{sim.get('status')}` · backend: `{sim.get('backend')}`",
            f"- error: `{sim.get('error', '')}`",
            "",
        ]
        cases = (sim.get("result") or {}).get("cases") or []
        if cases:
            lines += _table(
                ["场景", "类型", "结果", "matched_ratio", "max_abs_error"],
                [
                    [
                        c.get("name"),
                        c.get("kind"),
                        c.get("status"),
                        c.get("matched_ratio"),
                        c.get("max_abs_error"),
                    ]
                    for c in cases
                ],
            )
        else:
            lines += [
                "- 已生成的多场景测试脚本包含 normal / zeros / ones / large / tiny（按算子域自动裁剪），"
                "本机没有可执行的数值参考/设备：`unavailable`，不冒充通过；"
                "需在匹配 CANN + torch_npu 的机器上重跑 `aiop simulate <task>`。",
            ]
        lines += [""]
    else:
        lines += [
            "- 本次未执行 L3（`--no-simulate`、无 golden，或任务未编译通过）。",
            "",
        ]

    if negative:
        lines += ["## 4b. 异常/负例门禁", ""]
        lines += _table(
            ["项", "值"],
            [
                ["case", negative.get("case", "")],
                ["status", f"`{negative.get('status')}`"],
                ["expected rule", f"`{negative.get('expected_rule', '')}`"],
                ["observed rules", ", ".join(negative.get("observed_rules") or []) or "—"],
                ["static ok", f"`{negative.get('static_ok')}`"],
                ["bad kernel", f"`{negative.get('bad_kernel', '')}`"],
            ],
        )
        lines += [""]

    lines += ["## 5. 硬件上板（L4）", ""]
    if board:
        lines += [
            f"- status: `{board.get('status')}` · host: `{board.get('host', '')}`",
            f"- metrics: `{json.dumps(board.get('result', {}), ensure_ascii=False)[:800]}`",
            f"- error: `{board.get('error', '')}`",
            "",
        ]
    else:
        lines += [
            "- 未执行上板：未配置 `board.host`，或仿真未达可上板条件。",
            "- 配置 `AIOP_BOARD_HOST` 后重跑 `aiop board <task> --host user@a5-host`。",
            "",
        ]

    if audit:
        lines += ["## Lowering 审计", "", f"- status: `{audit.get('status')}`"]
        for item in audit.get("checks", []):
            lines.append(f"- `{item.get('id')}` {item.get('status')}: {item.get('detail')}")
        lines += [""]

    lines += ["## 6. 对齐 some_ref.md 的架构映射", ""]
    lines += _table(["参考模块", "aiop 实现", "产物", "状态"], _arch_map())
    lines += [""]

    if diagnosis:
        lines += ["## 诊断与回路路由（Diagnoser）", ""]
        lines += [f"- verdict: `{diagnosis.get('verdict')}`"]
        for cluster in diagnosis.get("clusters", []):
            lines.append(
                f"- `{cluster.get('loop')}` ×{cluster.get('count')}: "
                f"{cluster.get('reason')} → {cluster.get('recommendation')} "
                f"（{', '.join(cluster.get('evidence', []))}）"
            )
        lines += [""]

    lines += ["## 事实与缺口", ""]
    lines += [f"- {note}" for note in result.get("notes", [])] or ["- none"]
    lines += [
        "",
        "## 产物",
        "",
    ]
    for name, path in (result.get("artifacts") or {}).items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def write_campaign_report(
    task_dir: str | Path,
    result: dict[str, Any],
    out: str | Path | None = None,
    negative: dict[str, Any] | None = None,
) -> Path:
    text = render_campaign_report(result, negative)
    path = Path(task_dir) / "campaign_report.md"
    path.write_text(text, encoding="utf-8")
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return out_path
    return path
