# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""derive-overrides 兜底机制（references/index_derivation_contract.md 第 3 节）。

脚本推导不了的残留进 unresolved.json；单轮 LLM/人工裁决后固化为
derive-overrides/<surface>.json（落盘、可审查、可回放）；构建确定性消费。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def collect_unresolved(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "okf_derive_unresolved",
        "records": sorted(records, key=lambda r: (r.get("document", ""), r.get("field", ""))),
    }


def write_unresolved(run_output: Path, payload: dict[str, Any]) -> Path:
    target = run_output / "unresolved.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_overrides(path: Path) -> dict[str, list[dict[str, Any]]]:
    """derive-overrides/<surface>.json → {document: [adjudication, ...]}。"""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"invalid overrides schema: {path}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in payload.get("adjudications", []):
        document = entry.get("document")
        if not document or "field" not in entry or "value" not in entry:
            raise ValueError(f"invalid adjudication entry in {path}: {entry}")
        if not entry.get("reason"):
            raise ValueError(f"adjudication without reason: {entry}")
        grouped.setdefault(document, []).append(entry)
    return grouped


def apply_overrides(
    document: str, derived: dict[str, Any], overrides: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """将裁决覆盖应用到单文档推导结果；返回应用的裁决（供审计输出）。"""
    applied = []
    for entry in overrides.get(document, []):
        field, value = entry["field"], entry["value"]
        if field == "signatures.archs" and isinstance(derived.get("signatures"), list):
            for signature in derived["signatures"]:
                if not signature.get("archs"):
                    signature["archs"] = list(value)
        else:
            derived[field] = value
        applied.append({"field": field, "value": value, "reason": entry["reason"]})
    return applied
