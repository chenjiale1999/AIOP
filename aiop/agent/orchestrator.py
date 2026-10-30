"""End-to-end operator generation agent (design → generate → verify → report).

The orchestrator is intentionally deterministic: every stage has a concrete
artifact and a machine-checkable gate. An LLM, when configured, only supplies
source drafts; it cannot bypass the static/compile/simulation gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from ..board.runner import run_board
from ..checks.compiler import CompileResult, compile_kernel
from ..checks.static import Report as StaticReport
from ..checks.static import run_static
from ..config import Config, load_config
from ..generation.generator import generate
from ..knowledge.index import KnowledgeIndex, default_index
from ..runtime.hardware import Plan, build_plan, roofline
from ..simulation.runner import SimResult, run_simulation
from ..simulation.testgen import write_test
from ..specs import TaskSpec
from .design import build_design
from .state import TaskState


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def task_id_for(spec: TaskSpec) -> str:
    return f"{spec.name}-{_sha8(json.dumps(spec.to_dict(), sort_keys=True))}"


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _fallback_plan(spec: TaskSpec, plan: Plan, attempt: int) -> Plan:
    """Conservative variant used after a failed compile/simulation attempt."""
    if attempt <= 1:
        return plan
    if spec.family in ("elementwise_unary", "elementwise_binary"):
        elem = 2 if spec.inputs[0].dtype in ("float16", "bfloat16") else 4
        tile = 8192 if elem <= 2 else 4096
        total = spec.output.numel
        return Plan(
            family=spec.family,
            variant="simt_direct_global",
            blocks=64,
            tile=tile,
            threads=min(256, tile),
            num_stages=1,
            iterations=math.ceil(total / (64 * tile)),
            tail_mode="guarded",
        )
    if spec.family in ("row_reduce", "softmax", "rmsnorm", "layernorm"):
        total = spec.input.numel // spec.input.shape[-1]
        return Plan(
            family=spec.family,
            variant="simt_row_reducer",
            blocks=plan.blocks,
            rows=1,
            tile_extent=plan.tile_extent,
            threads=min(plan.threads, 128),
            num_stages=1,
            iterations=math.ceil(total / plan.blocks),
            tail_mode="exact" if total % plan.blocks == 0 else "guarded",
            buffer_bytes={"x": plan.tile_extent * 4, "acc": 8},
        )
    if spec.family == "gemm":
        from ..runtime.hardware import largest_divisor

        k = spec.inputs[0].shape[1]
        n = spec.inputs[1].shape[0]
        block_n = largest_divisor(n, max(16, plan.block_n // 2))
        block_k = largest_divisor(k, max(16, plan.block_k // 2))
        return Plan(
            family="gemm",
            variant="persistent_gemm",
            blocks=plan.blocks,
            num_stages=1,
            iterations=plan.iterations,
            block_m=plan.block_m,
            block_n=block_n,
            block_k=block_k,
            epilogue="none",
            tail_mode="exact",
        )
    return plan


class OperatorAgent:
    def __init__(self, cfg: Config | None = None, kb: KnowledgeIndex | None = None) -> None:
        self.cfg = cfg or load_config()
        self.kb = kb or default_index()

    # ------------------------------------------------------------------ helpers
    def _context(self, spec: TaskSpec) -> str:
        query = (
            f"{spec.family} {spec.name} tiling 内存 容量 精度 性能 "
            f"{' '.join(t.dtype for t in spec.inputs)}"
        )
        return self.kb.context(query, k=5)

    def _write_debug(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n\n")

    # -------------------------------------------------------------------- run
    def run(
        self,
        spec: TaskSpec,
        *,
        task_id: str | None = None,
        simulate: bool = True,
        board: bool = False,
        force_board: bool = False,
        strategy: str = "auto",
        reuse_workspace: bool = False,
    ) -> dict[str, Any]:
        started = time.time()
        task_id = task_id or task_id_for(spec)
        task_dir = self.cfg.workspace / task_id
        if reuse_workspace and task_dir.is_dir():
            pass
        else:
            task_dir.mkdir(parents=True, exist_ok=True)
            for stale in ("history", "npusim"):
                if (task_dir / stale).is_dir():
                    shutil.rmtree(task_dir / stale)
        history = task_dir / "history"
        history.mkdir(exist_ok=True)
        state = TaskState(task_id=task_id)
        state.save(task_dir / "state.json")
        spec.save(task_dir / "spec.yaml")

        knowledge_hits = self.kb.search(
            f"{spec.family} {spec.name} tiling 内存 精度 性能", k=5
        )
        knowledge_context = self._context(spec)

        # Stage 1: design
        state.enter("design")
        plan = build_plan(spec)
        design = build_design(spec, plan, knowledge_hits)
        (task_dir / "DESIGN.md").write_text(design, encoding="utf-8")
        state.complete("design")

        artifacts: dict[str, str] = {
            "design": str(task_dir / "DESIGN.md"),
            "spec": str(task_dir / "spec.yaml"),
        }

        static_report: StaticReport | None = None
        compile_result: CompileResult | None = None
        kernel_path = task_dir / f"{spec.name}.py"
        test_path = task_dir / f"test_{spec.name}.py"
        generator_name = "template"
        replaced_plan = None

        # Stage 2/3: generate + static + compile with repair loop
        state.enter("generate")
        attempts: list[dict[str, Any]] = []
        current_plan = plan
        for attempt in range(1, self.cfg.agent.max_attempts + 1):
            try:
                kernel_path, generator_name = generate(
                    spec,
                    current_plan,
                    task_dir,
                    self.cfg,
                    design=design,
                    knowledge=knowledge_context,
                    strategy=strategy,
                )
            except Exception as exc:  # noqa: BLE001 - generator bugs become task failures
                error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    {"attempt": attempt, "stage": "generate", "ok": False, "error": error}
                )
                self._write_debug(
                    task_dir / "debug_log.md",
                    f"## Attempt {attempt} — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                    f"- mode: generate\n- classification: generator_error\n- error: {error}",
                )
                compile_result = CompileResult(ok=False, error=error, category="generator")
                break
            write_test(
                spec,
                current_plan,
                task_dir,
                cann_home=self.cfg.cann.home,
                device_flags=self.cfg.compile.device_flags,
                warmup=self.cfg.simulation.warmup,
                repeats=self.cfg.simulation.repeats,
                tilelang_repo=self.cfg.paths.tilelang_repo,
            )
            static_report = run_static(spec, current_plan, kernel_path, self.cfg)
            (task_dir / "static_report.json").write_text(
                json.dumps(static_report.to_dict(), indent=2), encoding="utf-8"
            )
            if not static_report.ok:
                error = "; ".join(f.message for f in static_report.findings if f.severity == "error")
                attempts.append({"attempt": attempt, "stage": "static", "ok": False, "error": error})
                self._write_debug(
                    task_dir / "debug_log.md",
                    f"## Attempt {attempt} — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                    f"- mode: generate\n- classification: static_fail\n- error: {error}",
                )
                current_plan = _fallback_plan(spec, current_plan, attempt + 1)
                continue

            state.enter("compile")
            compile_result = compile_kernel(kernel_path, task_dir, self.cfg)
            attempts.append(
                {
                    "attempt": attempt,
                    "stage": "compile",
                    "ok": compile_result.ok,
                    "error": compile_result.error[-1200:],
                    "category": compile_result.category,
                    "generator": generator_name,
                }
            )
            self._write_debug(
                task_dir / "debug_log.md",
                f"## Attempt {attempt} — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                f"- mode: generate\n- generator: {generator_name}\n"
                f"- classification: {'compile_pass' if compile_result.ok else 'compile_fail'}\n"
                f"- error: {compile_result.error[-800:] or 'none'}\n"
                f"- plan: {json.dumps(current_plan.to_dict(), sort_keys=True)}",
            )
            if compile_result.ok:
                state.complete("compile")
                break
            if compile_result.category in ("cann_too_old", "toolchain_host_headers"):
                break
            # keep the failing source for audit, then try a conservative variant
            shutil.copyfile(kernel_path, history / f"{spec.name}_attempt{attempt}.py")
            current_plan = _fallback_plan(spec, current_plan, attempt + 1)

        artifacts["kernel"] = str(kernel_path)
        artifacts["test"] = str(test_path)
        artifacts["compile_report"] = str(task_dir / "compile_result.json")
        state.artifacts = artifacts

        compile_ok = bool(compile_result and compile_result.ok)
        if compile_ok and current_plan is not plan:
            # Documents must reflect the plan that actually compiled.
            design = build_design(spec, current_plan, knowledge_hits)
            (task_dir / "DESIGN.md").write_text(design, encoding="utf-8")
            replaced_plan = current_plan

        sim_result: SimResult | None = None
        if compile_ok and simulate:
            state.enter("simulate")
            sim_result = run_simulation(
                task_dir,
                test_path,
                spec,
                self.cfg,
                compile_audit=compile_result.audit if compile_result else None,
            )
            (task_dir / "simulation_report.json").write_text(
                json.dumps(sim_result.to_dict(), indent=2), encoding="utf-8"
            )
            if sim_result.status == "failed":
                self._write_debug(
                    task_dir / "debug_log.md",
                    f"## Simulation retry — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                    f"- classification: precision_fail\n- error: {sim_result.error}\n"
                    f"- metrics: {json.dumps(sim_result.result, ensure_ascii=False)[:1200]}",
                )
                current_plan = _fallback_plan(spec, current_plan, 2)
                try:
                    kernel_path, generator_name = generate(
                        spec,
                        current_plan,
                        task_dir,
                        self.cfg,
                        design=design,
                        knowledge=knowledge_context,
                        strategy=strategy,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._write_debug(
                        task_dir / "debug_log.md",
                        f"## Simulation retry generator error — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                        f"- error: {type(exc).__name__}: {exc}",
                    )
                    kernel_path = task_dir / f"{spec.name}.py"
                write_test(
                    spec,
                    current_plan,
                    task_dir,
                    cann_home=self.cfg.cann.home,
                    device_flags=self.cfg.compile.device_flags,
                    warmup=self.cfg.simulation.warmup,
                    repeats=self.cfg.simulation.repeats,
                    tilelang_repo=self.cfg.paths.tilelang_repo,
                )
                static_report = run_static(spec, current_plan, kernel_path, self.cfg)
                compile_result = compile_kernel(kernel_path, task_dir, self.cfg)
                if compile_result.ok:
                    sim_result = run_simulation(
                        task_dir,
                        test_path,
                        spec,
                        self.cfg,
                        compile_audit=compile_result.audit,
                    )
                    (task_dir / "simulation_report.json").write_text(
                        json.dumps(sim_result.to_dict(), indent=2), encoding="utf-8"
                    )
            if sim_result.status == "passed":
                state.complete("simulate")
            else:
                state.stages["simulate"] = sim_result.status

        board_result = None
        sim_ok_for_board = not simulate or (
            sim_result is not None and sim_result.status in ("passed", "unavailable")
        )
        if board and compile_ok and sim_ok_for_board:
            # Board packaging expects manifest.json before the run.
            (task_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "spec": spec.to_dict(),
                        "plan": current_plan.to_dict(),
                        "generator": generator_name,
                        "tilelang_repo": self.cfg.paths.tilelang_repo,
                        "cann_home": self.cfg.cann.home,
                        "device_flags": self.cfg.compile.device_flags,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            state.enter("board")
            board_result = run_board(
                task_dir, spec, self.cfg, force=force_board
            )
            (task_dir / "board_report.json").write_text(
                json.dumps(board_result.to_dict(), indent=2), encoding="utf-8"
            )
            if board_result.status == "passed":
                state.complete("board")
            else:
                state.stages["board"] = board_result.status
        elif board and not sim_ok_for_board:
            board_result = None
            self._write_debug(
                task_dir / "debug_log.md",
                f"## Board skipped — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                "- classification: simulation_not_passed\n"
                "- reason: board requires a passing (or explicitly unavailable) simulation",
            )

        # Final status and report
        if not compile_ok:
            if compile_result is None:
                status = "blocked_static"
            elif compile_result.category == "generator":
                status = "blocked_generate"
            elif compile_result.category in ("cann_too_old", "toolchain_host_headers"):
                status = "blocked_environment"
            else:
                status = "blocked_compile"
        elif sim_result and sim_result.status == "failed":
            status = "blocked_accuracy"
        elif board_result and board_result.status in ("failed", "blocked"):
            status = "blocked_board"
        elif sim_result and sim_result.passed:
            status = "success"
        else:
            status = "success_static_only"

        verified = bool(compile_ok and sim_result and sim_result.passed)
        state.status = status
        state.save(task_dir / "state.json")

        result = {
            "task_id": task_id,
            "status": status,
            "verified": verified,
            "simulation_requested": simulate,
            "generator": generator_name,
            "spec": spec.to_dict(),
            "plan": (current_plan if compile_ok else plan).to_dict(),
            "roofline": roofline(spec),
            "static": static_report.to_dict() if static_report else None,
            "compile": compile_result.to_dict() if compile_result else None,
            "simulation": sim_result.to_dict() if sim_result else None,
            "board": board_result.to_dict() if board_result else None,
            "attempts": attempts,
            "knowledge": knowledge_hits,
            "artifacts": artifacts,
            "environment": self._env_snapshot(),
            "elapsed_s": round(time.time() - started, 2),
            "notes": self._notes(sim_result, board_result, compile_result),
        }
        (task_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (task_dir / "report.md").write_text(self._report_markdown(result), encoding="utf-8")
        (task_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "spec": spec.to_dict(),
                    "plan": result["plan"],
                    "generator": generator_name,
                    "tilelang_repo": self.cfg.paths.tilelang_repo,
                    "cann_home": self.cfg.cann.home,
                    "device_flags": self.cfg.compile.device_flags,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def _env_snapshot(self) -> dict[str, Any]:
        from ..runtime.tilelang_a5 import environment_probe

        try:
            return environment_probe(self.cfg)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _notes(self, sim: SimResult | None, board, compile_result: CompileResult | None) -> list[str]:
        notes: list[str] = []
        if compile_result and compile_result.ok:
            notes.append(
                "L1/L2 verified: lowering + bisheng device compile passed "
                f"({compile_result.elapsed_s:.1f}s)."
            )
        if sim and sim.status == "unavailable":
            notes.append(
                "L3 functional simulation unavailable: "
                + "; ".join(sim.notes or [sim.error or "no functional simulator"])
                + ". Numeric verification must run on a machine with a matching CANN + torch_npu."
            )
        if sim and sim.passed:
            backend = sim.backend
            notes.append(f"L3 passed on backend={backend}.")
        if board and board.status == "skipped":
            notes.append("L4 skipped: configure board.host to enable remote measurement.")
        if board and board.status == "blocked":
            notes.append("L4 blocked: " + board.error)
        return notes

    def _report_markdown(self, result: dict[str, Any]) -> str:
        spec = result["spec"]
        plan = result["plan"]
        sim = result.get("simulation") or {}
        comp = result.get("compile") or {}
        lines = [
            f"# {spec['name']} — aiop 验证报告",
            "",
            f"- status: **{result['status']}**",
            f"- verified: `{result['verified']}`",
            f"- generator: `{result['generator']}`",
            f"- task: `{result['task_id']}`",
            "",
            "## 计划",
            "",
            "```json",
            json.dumps(plan, indent=2),
            "```",
            "",
            "## 编译（L1/L2）",
            "",
            f"- ok: `{comp.get('ok')}`",
            f"- category: `{comp.get('category')}`",
            f"- CCE source: `{comp.get('source_path', '')}`",
            f"- audit: `{json.dumps(comp.get('audit', {}))}`",
            "",
        ]
        if comp.get("error"):
            lines += ["```text", comp["error"][-2000:], "```", ""]
        lines += [
            "## 仿真（L3）",
            "",
            f"- status: `{sim.get('status', 'not_run')}`",
            f"- backend: `{sim.get('backend', 'n/a')}`",
            f"- metrics: `{json.dumps(sim.get('result', {}))[:1500]}`",
            "",
        ]
        if sim.get("notes"):
            lines += ["- notes: " + "; ".join(sim["notes"]), ""]
        board = result.get("board") or {}
        lines += [
            "## 上板（L4）",
            "",
            f"- status: `{board.get('status', 'not_run')}`",
            f"- host: `{board.get('host', '')}`",
            f"- metrics: `{json.dumps(board.get('result', {}))[:1500]}`",
            "",
            "## 事实与缺口",
            "",
        ]
        lines += [f"- {note}" for note in result.get("notes", [])] or ["- none"]
        lines += ["", "## 产物", ""]
        lines += [f"- {name}: `{path}`" for name, path in result.get("artifacts", {}).items()]
        return "\n".join(lines) + "\n"
