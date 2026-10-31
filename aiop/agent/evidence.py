"""EvidenceStore and Diagnoser: append-only facts and failure routing.

Every stage appends a typed record to ``evidence.jsonl``. The diagnoser groups
failed records into clusters and routes each cluster to the loop that can fix
it (plan/schedule, kernel/skill, environment, dataflow, board), mirroring the
``EvidenceStore`` + ``Diagnoser`` split in ``some_ref.md``.

Records are facts, not claims: only executed stages are recorded, and an
unavailable stage is recorded as ``unavailable`` rather than passed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOOPS = {
    "environment": "修复工具链/CANN/设备，或调整编译 flags",
    "kernel": "回到生成/技能层重写 kernel，或修正 API/语义",
    "schedule": "回到 Planner/Search：调整 tile/rows/stages/grid",
    "dataflow": "回到数据流与 golden：检查归约/精度/边界处理",
    "board": "回到上板流程：检查 SSH/远端环境/测量协议",
    "search": "扩大或收缩候选预算，重新评估计划",
}


def _route(record: dict[str, Any]) -> tuple[str, str]:
    stage = str(record.get("stage", ""))
    category = str(record.get("category") or "none")
    facts = record.get("facts") or {}
    rules = facts.get("rules") or []
    if stage == "board":
        return "board", "上板失败或预检未过"
    if category in ("cann_too_old", "toolchain_host_headers", "timeout", "environment"):
        return "environment", "编译环境不满足 A5 要求"
    if category == "capacity":
        return "schedule", "片上容量超预算"
    if category in ("lowering_unsupported", "python_syntax", "generator"):
        return "kernel", "kernel 源码/lowering 不被接受"
    if category in ("precision", "accuracy"):
        return "dataflow", "数值精度未达门限"
    if category == "search":
        return "search", "计划搜索失败"
    if any(str(rule).startswith(("HW002", "HW003", "HW008", "HW009", "HW010")) for rule in rules):
        return "schedule", "硬件容量/对齐约束不满足"
    if any(str(rule).startswith(("API", "SYN", "SEM")) for rule in rules):
        return "kernel", "静态 API/语义规则不满足"
    if stage == "static":
        return "kernel", "静态检查未通过"
    if stage == "compile":
        return "kernel", "设备编译未通过"
    return "kernel", "未分类失败"


class EvidenceStore:
    def __init__(self, task_dir: str | Path, task_id: str, *, reset: bool = True) -> None:
        self.task_dir = Path(task_dir)
        self.task_id = task_id
        self.path = self.task_dir / "evidence.jsonl"
        self.records: list[dict[str, Any]] = []
        if reset:
            self.task_dir.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def record(
        self,
        stage: str,
        kind: str,
        status: str,
        *,
        category: str = "none",
        facts: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        parent: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "id": f"ev-{len(self.records) + 1:04d}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_id": self.task_id,
            "stage": stage,
            "kind": kind,
            "status": status,
            "category": category,
            "facts": facts or {},
            "artifacts": artifacts or [],
            "parent": parent,
        }
        self.records.append(record)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def diagnose(records: list[dict[str, Any]]) -> dict[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    failed = [r for r in records if str(r.get("status")) in ("failed", "blocked", "error")]
    for record in failed:
        loop, reason = _route(record)
        category = str(record.get("category") or "none")
        rules = (record.get("facts") or {}).get("rules") or []
        key = f"{loop}:{category if category != 'none' else (rules[0] if rules else record.get('stage'))}"
        cluster = clusters.setdefault(
            key,
            {
                "key": key,
                "loop": loop,
                "stage": record.get("stage"),
                "category": category,
                "rules": sorted({str(rule) for rule in rules}),
                "count": 0,
                "reason": reason,
                "recommendation": LOOPS.get(loop, "人工分析"),
                "evidence": [],
            },
        )
        cluster["count"] += 1
        cluster["evidence"].append(record.get("id"))
    ordered = sorted(clusters.values(), key=lambda c: (-c["count"], c["key"]))
    return {
        "task_id": records[0].get("task_id") if records else None,
        "total_records": len(records),
        "failed_records": len(failed),
        "verdict": "no_failure" if not failed else "needs_attention",
        "clusters": ordered,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_diagnosis(task_dir: str | Path, diagnosis: dict[str, Any]) -> Path:
    path = Path(task_dir) / "diagnosis.json"
    path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
