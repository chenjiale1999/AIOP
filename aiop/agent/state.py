"""Task state persistence for the A5 operator agent."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskState:
    task_id: str
    status: str = "running"
    stages: dict[str, str] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    last_error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    artifacts: dict[str, str] = field(default_factory=dict)

    def enter(self, stage: str) -> None:
        self.stages[stage] = "in_progress"
        self.attempts.setdefault(stage, 0)
        self.touch()

    def complete(self, stage: str) -> None:
        self.stages[stage] = "completed"
        self.touch()

    def fail(self, stage: str, error: str = "") -> None:
        self.stages[stage] = "failed"
        self.attempts[stage] = self.attempts.get(stage, 0) + 1
        self.last_error = error
        self.touch()

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "stages": self.stages,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": self.artifacts,
        }

    def save(self, path: Path) -> None:
        self.touch()
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, task_id: str) -> "TaskState":
        data = json.loads(path.read_text(encoding="utf-8"))
        state = cls(task_id=data.get("task_id", task_id), status=data.get("status", "running"))
        state.stages = data.get("stages", {})
        state.attempts = data.get("attempts", {})
        state.last_error = data.get("last_error", "")
        state.created_at = data.get("created_at", time.time())
        state.updated_at = data.get("updated_at", time.time())
        state.artifacts = data.get("artifacts", {})
        return state
