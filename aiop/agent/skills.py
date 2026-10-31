"""SkillRegistry / SkillAdapter: deterministic skill selection and locking.

Skills are markdown procedures under ``.agent/skills``. This module turns them
from passive documentation into an explicit generation input:

* select the skills that apply to the task family / mode / stage,
* pin their content with a sha256 lock so a run is reproducible,
* map each skill onto concrete check obligations consumed by the contract,
* emit ``context_manifest.json`` for auditability.

It implements the ``SkillRegistry`` + ``SkillAdapter`` responsibilities from the
reference architecture (``some_ref.md``) without pretending that prose skills
are executable code.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import PROJECT_ROOT
from ..specs import TaskSpec

SKILL_OBLIGATIONS: dict[str, list[str]] = {
    "aiop-op-pipeline": [
        "L0-static",
        "L1-lowering",
        "L2-device-compile",
        "L3-simulate",
        "L4-board",
        "evidence-archive",
    ],
    "tilelang-ascend": [
        "API-whitelist",
        "reducer-lifecycle",
        "sync-balance",
        "guard-coverage",
    ],
    "asc-npu-arch": ["dtype-set", "grid-limit", "ub-budget", "alignment"],
    "tilelang-semantic": [
        "dependency",
        "buffer-ownership",
        "race-check",
        "sync-protocol",
    ],
    "tilelang-layout": ["layout-inference", "alignment", "l2-cache-policy"],
    "tilelang-build": ["toolchain-probe", "rebuild-protocol"],
    "asc-search": ["api-evidence"],
    "ascendc-api-kernel-debug": ["no-debug-apis", "debug-artifacts"],
}


@dataclass
class SkillMeta:
    name: str
    description: str
    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass
class SkillBundle:
    skills: list[SkillMeta] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    obligations: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": [s.to_dict() for s in self.skills],
            "reasons": dict(self.reasons),
            "obligations": list(self.obligations),
            "missing": list(self.missing),
        }


def _parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    block: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        block.append(line)
    if not block:
        return {}
    try:
        data = yaml.safe_load("\n".join(block)) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


class SkillRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or (PROJECT_ROOT / ".agent" / "skills"))
        self._cache: dict[str, SkillMeta] | None = None

    def scan(self) -> dict[str, SkillMeta]:
        if self._cache is not None:
            return self._cache
        found: dict[str, SkillMeta] = {}
        if self.root.is_dir():
            for path in sorted(self.root.glob("*/SKILL.md")):
                text = path.read_text(encoding="utf-8", errors="ignore")
                meta = _parse_frontmatter(text)
                name = str(meta.get("name") or path.parent.name)
                description = " ".join(str(meta.get("description") or "").split())
                found[name] = SkillMeta(
                    name=name,
                    description=description,
                    path=str(path.relative_to(PROJECT_ROOT)),
                    sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    bytes=len(text.encode("utf-8")),
                )
        self._cache = found
        return found

    def select(
        self, spec: TaskSpec, *, open_mode: bool = False, board: bool = False
    ) -> SkillBundle:
        registry = self.scan()
        wanted: list[tuple[str, str]] = [
            ("aiop-op-pipeline", "端到端门禁与产物协议"),
            ("tilelang-ascend", "Ascend dialect 编程约束"),
            ("asc-npu-arch", "A5 容量/核心/dtype 硬件事实"),
            ("tilelang-semantic", "循环嵌套/依赖/缓冲区/同步语义"),
        ]
        if spec.family == "gemm":
            wanted.append(("tilelang-layout", "GEMM 的 layout 推断与对齐约束"))
        if open_mode:
            wanted.append(("asc-search", "无模板算子需要 API 能力证据"))
        if board:
            wanted.append(("ascendc-api-kernel-debug", "上板调试接口与证据采集"))

        bundle = SkillBundle()
        seen: set[str] = set()
        for name, reason in wanted:
            if name in seen:
                continue
            seen.add(name)
            meta = registry.get(name)
            if meta is None:
                bundle.missing.append(name)
                continue
            bundle.skills.append(meta)
            bundle.reasons[name] = reason
        obligations: list[str] = []
        for meta in bundle.skills:
            for item in SKILL_OBLIGATIONS.get(meta.name, []):
                if item not in obligations:
                    obligations.append(item)
        bundle.obligations = obligations
        return bundle


def write_skill_lock(task_dir: str | Path, bundle: SkillBundle) -> Path:
    path = Path(task_dir) / "skill.lock"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "skills": {
            meta.name: {
                "sha256": meta.sha256,
                "path": meta.path,
                "bytes": meta.bytes,
            }
            for meta in bundle.skills
        },
        "obligations": list(bundle.obligations),
        "missing": list(bundle.missing),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_context_manifest(
    task_dir: str | Path,
    *,
    task_id: str,
    spec: TaskSpec,
    bundle: SkillBundle,
    knowledge_hits: list[dict[str, Any]],
    open_mode: bool,
) -> Path:
    path = Path(task_dir) / "context_manifest.json"
    spec_digest = hashlib.sha256(
        json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    payload = {
        "task_id": task_id,
        "op": spec.name,
        "family": spec.family,
        "spec_sha": spec_digest,
        "open_mode": open_mode,
        "skills": [
            {**meta.to_dict(), "reason": bundle.reasons.get(meta.name, "")}
            for meta in bundle.skills
        ],
        "obligations": list(bundle.obligations),
        "missing_skills": list(bundle.missing),
        "knowledge": [
            {
                "doc": hit.get("doc"),
                "heading": hit.get("heading"),
                "score": hit.get("score"),
            }
            for hit in knowledge_hits
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
