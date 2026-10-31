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
from ..generation.generator import generate, repair
from ..generation.llm import LLMClient
from ..knowledge.index import KnowledgeIndex, default_index
from ..runtime.hardware import Plan, build_open_plan, build_plan, roofline
from ..simulation.runner import SimResult, run_simulation
from ..simulation.testgen import GOLDEN_FAMILIES, write_test
from ..specs import TaskSpec, UnsupportedOperator
from .audit import build_lowering_audit, write_lowering_audit
from .contract import build_contract, write_contract
from .design import build_blocked_design, build_design, build_open_design
from .evidence import EvidenceStore, diagnose, write_diagnosis
from .search import SearchController
from .skills import SkillRegistry, write_context_manifest, write_skill_lock
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
            out_cast=plan.out_cast,
            tail_mode="exact",
        )
    return plan


class OperatorAgent:
    def __init__(self, cfg: Config | None = None, kb: KnowledgeIndex | None = None) -> None:
        self.cfg = cfg or load_config()
        self.kb = kb or default_index()

    # ------------------------------------------------------------------ helpers
    def _llm_client(self) -> LLMClient | None:
        """Return a usable client, or None when the LLM is not configured."""
        client = LLMClient(self.cfg)
        return client if client.available else None

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
        draft: str | None = None,
        search: bool = False,
        search_budget: int = 4,
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
        evidence = EvidenceStore(task_dir, task_id)

        knowledge_hits = self.kb.search(
            f"{spec.family} {spec.name} tiling 内存 精度 性能", k=5
        )
        knowledge_context = self._context(spec)

        # Stage 1: design
        state.enter("design")
        mode = "template"
        golden_source = "aiop-builtin"
        golden_override: str | None = None
        open_reasons: list[str] = []
        try:
            plan = build_plan(spec)
            design = build_design(spec, plan, knowledge_hits)
        except UnsupportedOperator as exc:
            client = self._llm_client()
            if strategy != "draft" and client is None:
                return self._run_unsupported(
                    spec,
                    exc,
                    task_id=task_id,
                    task_dir=task_dir,
                    state=state,
                    knowledge_hits=knowledge_hits,
                    started=started,
                )
            # Open-ended mode: no template family, so an external writer owns
            # the tiling — the LLM directly, or a draft file from an IDE agent.
            mode = "open"
            open_reasons = list(exc.reasons)
            plan = build_open_plan(spec)
            golden_source = "unavailable"
            if strategy == "draft":
                golden_path = task_dir / "golden_draft.py"
                if golden_path.is_file() and not spec.metadata_inputs:
                    golden_override = golden_path.read_text(encoding="utf-8")
                    golden_source = "agent-self-referenced"
            else:
                strategy = "llm"
                if not spec.metadata_inputs:
                    golden_source = "llm-self-referenced"
            design = build_open_design(
                spec, open_reasons, knowledge_hits, golden_source=golden_source
            )
            if client is not None and strategy == "llm" and golden_source != "unavailable":
                try:
                    golden_override = client.generate_golden(spec.to_dict(), design)
                except Exception as exc:  # noqa: BLE001 - reference is optional
                    golden_override = None
                    golden_source = "unavailable"
                    design = build_open_design(
                        spec, open_reasons, knowledge_hits, golden_source=golden_source
                    )
                    self._write_debug(
                        task_dir / "debug_log.md",
                        f"## Golden unavailable — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                        f"- error: {type(exc).__name__}: {exc}",
                    )
            if golden_override:
                (task_dir / "golden_ref.py").write_text(
                    "def golden_body():\n" + golden_override + "\n", encoding="utf-8"
                )
        (task_dir / "DESIGN.md").write_text(design, encoding="utf-8")
        state.complete("design")

        artifacts: dict[str, str] = {
            "design": str(task_dir / "DESIGN.md"),
            "spec": str(task_dir / "spec.yaml"),
        }

        # Stage 1b: freeze the task contract and the skill bundle it relies on.
        contract = None
        skill_bundle = None
        try:
            skill_bundle = SkillRegistry().select(
                spec, open_mode=(mode == "open"), board=board
            )
            contract = build_contract(
                spec,
                plan,
                obligations=skill_bundle.obligations,
                open_mode=(mode == "open"),
            )
            artifacts["contract"] = str(write_contract(task_dir, contract))
            artifacts["skill_lock"] = str(write_skill_lock(task_dir, skill_bundle))
            artifacts["context_manifest"] = str(
                write_context_manifest(
                    task_dir,
                    task_id=task_id,
                    spec=spec,
                    bundle=skill_bundle,
                    knowledge_hits=knowledge_hits,
                    open_mode=(mode == "open"),
                )
            )
            evidence.record(
                "design",
                "contract",
                "passed",
                facts={
                    "contract_hash": contract.contract_hash,
                    "skills": [meta.name for meta in skill_bundle.skills],
                    "obligations": list(skill_bundle.obligations),
                },
                artifacts=[artifacts["contract"], artifacts["skill_lock"]],
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not kill a run
            evidence.record(
                "design",
                "contract",
                "failed",
                category="generator",
                facts={"error": f"{type(exc).__name__}: {exc}"},
            )

        # Stage 1c: bounded plan search (opt-in; deterministic templates only).
        if search and mode == "template":
            try:
                controller = SearchController(self.cfg)
                chosen_plan, frontier = controller.search(
                    spec,
                    plan,
                    task_dir,
                    design=design,
                    knowledge=knowledge_context,
                    budget=search_budget,
                )
                artifacts["frontier"] = str(frontier)
                if chosen_plan.to_dict() != plan.to_dict():
                    plan = chosen_plan
                    design = build_design(spec, plan, knowledge_hits)
                    (task_dir / "DESIGN.md").write_text(design, encoding="utf-8")
                    if contract is not None:
                        contract = build_contract(
                            spec,
                            plan,
                            obligations=contract.obligations,
                            open_mode=(mode == "open"),
                        )
                        artifacts["contract"] = str(write_contract(task_dir, contract))
                evidence.record(
                    "search",
                    "plan_search",
                    "passed",
                    facts={"chosen": plan.to_dict(), "frontier": str(frontier)},
                    artifacts=[str(frontier)],
                )
            except Exception as exc:  # noqa: BLE001 - fall back to the base plan
                evidence.record(
                    "search",
                    "plan_search",
                    "failed",
                    category="search",
                    facts={"error": f"{type(exc).__name__}: {exc}"},
                )

        static_report: StaticReport | None = None
        compile_result: CompileResult | None = None
        kernel_path = task_dir / f"{spec.name}.py"
        test_path = task_dir / f"test_{spec.name}.py"
        generator_name = "template"
        can_test = spec.family in GOLDEN_FAMILIES or golden_override is not None
        replaced_plan = None

        # Stage 2/3: generate + static + compile with repair loop
        state.enter("generate")
        attempts: list[dict[str, Any]] = []
        current_plan = plan
        pending_repair: str | None = None
        for attempt in range(1, self.cfg.agent.max_attempts + 1):
            try:
                if pending_repair is not None and self._llm_client() is not None:
                    kernel_path, generator_name = repair(
                        spec,
                        current_plan,
                        kernel_path,
                        pending_repair,
                        self.cfg,
                        design=design,
                    )
                else:
                    kernel_path, generator_name = generate(
                        spec,
                        current_plan,
                        task_dir,
                        self.cfg,
                        design=design,
                        knowledge=knowledge_context,
                        strategy=strategy,
                        open_mode=mode == "open",
                        draft_path=draft,
                    )
            except Exception as exc:  # noqa: BLE001 - generator bugs become task failures
                error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    {"attempt": attempt, "stage": "generate", "ok": False, "error": error}
                )
                evidence.record(
                    "generate",
                    "kernel_source",
                    "failed",
                    category="generator",
                    facts={"attempt": attempt, "error": error[-1200:]},
                )
                self._write_debug(
                    task_dir / "debug_log.md",
                    f"## Attempt {attempt} — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                    f"- mode: generate\n- classification: generator_error\n- error: {error}",
                )
                compile_result = CompileResult(ok=False, error=error, category="generator")
                break
            pending_repair = None
            evidence.record(
                "generate",
                "kernel_source",
                "passed",
                facts={
                    "attempt": attempt,
                    "generator": generator_name,
                    "plan_variant": current_plan.variant,
                },
                artifacts=[str(kernel_path)],
            )
            if can_test:
                write_test(
                    spec,
                    current_plan,
                    task_dir,
                    cann_home=self.cfg.cann.home,
                    device_flags=self.cfg.compile.device_flags,
                    warmup=self.cfg.simulation.warmup,
                    repeats=self.cfg.simulation.repeats,
                    tilelang_repo=self.cfg.paths.tilelang_repo,
                    golden_override=golden_override,
                )
            static_report = run_static(spec, current_plan, kernel_path, self.cfg)
            (task_dir / "static_report.json").write_text(
                json.dumps(static_report.to_dict(), indent=2), encoding="utf-8"
            )
            static_rules = sorted({f.rule for f in static_report.findings})
            if static_report.ok:
                evidence.record(
                    "static",
                    "static_gate",
                    "passed",
                    facts={"attempt": attempt, "rules": static_rules},
                    artifacts=[str(task_dir / "static_report.json")],
                )
            else:
                evidence.record(
                    "static",
                    "static_gate",
                    "failed",
                    category="static",
                    facts={
                        "attempt": attempt,
                        "rules": static_rules,
                        "errors": [
                            f.message for f in static_report.findings if f.severity == "error"
                        ][:8],
                    },
                    artifacts=[str(task_dir / "static_report.json")],
                )
            if not static_report.ok:
                error = "; ".join(f.message for f in static_report.findings if f.severity == "error")
                attempts.append({"attempt": attempt, "stage": "static", "ok": False, "error": error})
                self._write_debug(
                    task_dir / "debug_log.md",
                    f"## Attempt {attempt} — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                    f"- mode: generate\n- classification: static_fail\n- error: {error}",
                )
                if self._llm_client() is not None:
                    pending_repair = error
                elif strategy == "draft":
                    self._write_fix_request(
                        task_dir, spec, current_plan, "static", error, static_report
                    )
                    break
                else:
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
            evidence.record(
                "compile",
                "compile_gate",
                "passed" if compile_result.ok else "failed",
                category=compile_result.category,
                facts={
                    "attempt": attempt,
                    "elapsed_s": round(compile_result.elapsed_s, 2),
                    "error": compile_result.error[-1200:] if not compile_result.ok else "",
                    "audit": compile_result.audit,
                },
                artifacts=[compile_result.source_path] if compile_result.source_path else [],
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
            # keep the failing source for audit, then let the LLM repair it, or
            # fall back to a conservative template variant when no LLM is set.
            shutil.copyfile(kernel_path, history / f"{spec.name}_attempt{attempt}.py")
            if self._llm_client() is not None:
                pending_repair = compile_result.error or compile_result.category
            elif strategy == "draft":
                self._write_fix_request(
                    task_dir, spec, current_plan, "compile", compile_result.error, None
                )
                break
            else:
                current_plan = _fallback_plan(spec, current_plan, attempt + 1)

        artifacts["kernel"] = str(kernel_path)
        if can_test:
            artifacts["test"] = str(test_path)
        artifacts["compile_report"] = str(task_dir / "compile_result.json")
        state.artifacts = artifacts

        compile_ok = bool(compile_result and compile_result.ok)
        lowering_audit = None
        if compile_ok:
            lowering_audit = build_lowering_audit(spec, current_plan, compile_result)
            artifacts["lowering_audit"] = str(
                write_lowering_audit(task_dir, lowering_audit)
            )
            evidence.record(
                "audit",
                "lowering_audit",
                lowering_audit["status"],
                facts={
                    "checks": [f"{c['id']}:{c['status']}" for c in lowering_audit["checks"]]
                },
                artifacts=[artifacts["lowering_audit"]],
            )
        if compile_ok and current_plan is not plan and mode == "template":
            # Documents must reflect the plan that actually compiled.
            design = build_design(spec, current_plan, knowledge_hits)
            (task_dir / "DESIGN.md").write_text(design, encoding="utf-8")
            replaced_plan = current_plan

        sim_result: SimResult | None = None
        if compile_ok and simulate and not can_test:
            self._write_debug(
                task_dir / "debug_log.md",
                f"## Simulation skipped — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                f"- classification: no_golden\n- golden_source: {golden_source}\n"
                "- reason: manifest metadata inputs need their `values` generators, "
                "which aiop does not implement",
            )
        if compile_ok and simulate and can_test:
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
            evidence.record(
                "simulate",
                "simulation",
                sim_result.status,
                category="accuracy" if sim_result.status == "failed" else "none",
                facts={
                    "backend": sim_result.backend,
                    "matched_ratio": sim_result.result.get("matched_ratio"),
                    "max_abs_error": sim_result.result.get("max_abs_error"),
                    "cases": len(sim_result.result.get("cases") or []),
                    "error": sim_result.error[-800:],
                },
                artifacts=[str(task_dir / "simulation_report.json")],
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
                        open_mode=mode == "open",
                        draft_path=draft,
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
                    golden_override=golden_override,
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
                    evidence.record(
                        "simulate",
                        "simulation_retry",
                        sim_result.status,
                        category="accuracy" if sim_result.status == "failed" else "none",
                        facts={
                            "backend": sim_result.backend,
                            "matched_ratio": sim_result.result.get("matched_ratio"),
                            "cases": len(sim_result.result.get("cases") or []),
                        },
                        artifacts=[str(task_dir / "simulation_report.json")],
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
            evidence.record(
                "board",
                "board_run",
                board_result.status,
                category="board" if board_result.status in ("failed", "blocked") else "none",
                facts={
                    "host": board_result.host,
                    "metrics": board_result.result,
                    "error": board_result.error[-800:],
                },
                artifacts=[str(task_dir / "board_report.json")],
            )
            if board_result.status == "passed":
                state.complete("board")
            else:
                state.stages["board"] = board_result.status
        elif board and not sim_ok_for_board:
            board_result = None
            evidence.record(
                "board",
                "board_run",
                "skipped",
                facts={"reason": "simulation not passed; board requires a passing or unavailable L3"},
            )
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
            status = "success" if mode == "template" else "success_static_only"
        else:
            status = "success_static_only"

        # Only a reference aiop owns counts as independent evidence; an LLM-written
        # golden agreeing with an LLM-written kernel is not proof.
        trusted_golden = mode == "template"
        verified = bool(compile_ok and sim_result and sim_result.passed and trusted_golden)

        # The delivered plan is what the contract must describe.
        final_plan = current_plan if compile_ok else plan
        if contract is not None:
            contract = build_contract(
                spec,
                final_plan,
                obligations=contract.obligations,
                open_mode=(mode == "open"),
            )
            artifacts["contract"] = str(write_contract(task_dir, contract))
        diagnosis = diagnose(evidence.records)
        artifacts["evidence"] = str(evidence.path)
        artifacts["diagnosis"] = str(write_diagnosis(task_dir, diagnosis))

        state.status = status
        state.save(task_dir / "state.json")

        result = {
            "task_id": task_id,
            "status": status,
            "verified": verified,
            "mode": mode,
            "golden_source": golden_source,
            "self_referenced": not trusted_golden,
            "simulation_requested": simulate,
            "generator": generator_name,
            "spec": spec.to_dict(),
            "plan": final_plan.to_dict(),
            "roofline": roofline(spec),
            "contract": contract.to_dict() if contract else None,
            "skills": skill_bundle.to_dict() if skill_bundle else None,
            "lowering_audit": lowering_audit,
            "diagnosis": diagnosis,
            "static": static_report.to_dict() if static_report else None,
            "compile": compile_result.to_dict() if compile_result else None,
            "simulation": sim_result.to_dict() if sim_result else None,
            "board": board_result.to_dict() if board_result else None,
            "attempts": attempts,
            "knowledge": knowledge_hits,
            "artifacts": artifacts,
            "environment": self._env_snapshot(),
            "elapsed_s": round(time.time() - started, 2),
            "notes": self._notes(sim_result, board_result, compile_result, mode, golden_source),
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
                    "mode": mode,
                    "golden_source": golden_source,
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

    def _run_unsupported(
        self,
        spec: TaskSpec,
        exc: UnsupportedOperator,
        *,
        task_id: str,
        task_dir: Path,
        state: TaskState,
        knowledge_hits: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any]:
        """Honest terminal state: record the gap, generate nothing, fake nothing."""
        reasons = list(exc.reasons)
        design = build_blocked_design(spec, reasons, knowledge_hits)
        (task_dir / "DESIGN.md").write_text(design, encoding="utf-8")
        state.stages["design"] = "blocked_unsupported"
        state.status = "blocked_unsupported"
        notes = [
            "Operator contract parsed from the TileOPs manifest, but no A5 template family "
            "covers it in aiop; no kernel was generated and no measurement is reported.",
            *[f"unsupported: {reason}" for reason in reasons],
            "To let the LLM attempt this operator, configure agent.llm "
            "(AIOP_LLM_BASE_URL / AIOP_LLM_MODEL / AIOP_LLM_API_KEY) and rerun; "
            "the open-ended path still enforces static + lowering + device-compile gates.",
        ]
        artifacts = {
            "design": str(task_dir / "DESIGN.md"),
            "spec": str(task_dir / "spec.yaml"),
        }
        result: dict[str, Any] = {
            "task_id": task_id,
            "status": "blocked_unsupported",
            "verified": False,
            "mode": "blocked",
            "golden_source": "none",
            "self_referenced": False,
            "simulation_requested": False,
            "generator": "none",
            "spec": spec.to_dict(),
            "plan": None,
            "roofline": roofline(spec),
            "contract": None,
            "skills": None,
            "lowering_audit": None,
            "diagnosis": None,
            "static": None,
            "compile": None,
            "simulation": None,
            "board": None,
            "attempts": [],
            "unsupported_reasons": reasons,
            "knowledge": knowledge_hits,
            "environment": self._env_snapshot(),
            "elapsed_s": round(time.time() - started, 2),
            "notes": notes,
        }
        evidence = EvidenceStore(task_dir, task_id)
        try:
            skill_bundle = SkillRegistry().select(spec, open_mode=False, board=False)
            contract = build_contract(
                spec, None, obligations=skill_bundle.obligations, open_mode=False
            )
            artifacts["contract"] = str(write_contract(task_dir, contract))
            artifacts["skill_lock"] = str(write_skill_lock(task_dir, skill_bundle))
            artifacts["context_manifest"] = str(
                write_context_manifest(
                    task_dir,
                    task_id=task_id,
                    spec=spec,
                    bundle=skill_bundle,
                    knowledge_hits=knowledge_hits,
                    open_mode=False,
                )
            )
            result["contract"] = contract.to_dict()
            result["skills"] = skill_bundle.to_dict()
            evidence.record(
                "design",
                "blocked_unsupported",
                "blocked",
                category="unsupported",
                facts={"reasons": reasons, "contract_hash": contract.contract_hash},
                artifacts=[artifacts["contract"]],
            )
        except Exception as exc:  # noqa: BLE001 - keep the terminal report honest
            evidence.record(
                "design",
                "blocked_unsupported",
                "blocked",
                category="unsupported",
                facts={"reasons": reasons, "error": f"{type(exc).__name__}: {exc}"},
            )
        diagnosis = diagnose(evidence.records)
        artifacts["evidence"] = str(evidence.path)
        artifacts["diagnosis"] = str(write_diagnosis(task_dir, diagnosis))
        result["diagnosis"] = diagnosis
        result["artifacts"] = artifacts
        state.artifacts = artifacts
        state.save(task_dir / "state.json")
        (task_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (task_dir / "report.md").write_text(self._report_markdown(result), encoding="utf-8")
        (task_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "spec": spec.to_dict(),
                    "plan": None,
                    "generator": "none",
                    "unsupported_reasons": reasons,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def _write_fix_request(
        self,
        task_dir: Path,
        spec: TaskSpec,
        plan: Plan,
        stage: str,
        error: str,
        static_report: StaticReport | None,
    ) -> Path:
        """Hand the failure back to the external writer (draft mode)."""
        findings = ""
        if static_report is not None:
            findings = "\n".join(
                f"- `{f.rule}` {f.severity}: {f.message}"
                for f in static_report.findings
                if f.severity == "error"
            )
            findings = f"\n## 静态检查 findings\n\n{findings}\n" if findings else ""
        path = task_dir / "FIX_REQUEST.md"
        path.write_text(
            f"""# 修复请求 — {spec.manifest.get('op', spec.name) if spec.manifest else spec.name}

> 阶段：**{stage}** · 布局：`llm_open` / blocks={plan.blocks}
> 修改 `draft.py` 后重跑同一条 `--strategy draft` 命令。

## 错误

```text
{error[-6000:]}
```
{findings}
## 约束回顾

- `T.Kernel` 一维、无 `threads=`；线程域只用 `T.SimtVF(threads=...)`。
- reducer 的 alloc/init/finalize 必须配对；op 只能是 sum/max/min。
- 禁止 `T.print` / `T.device_assert` / 动态循环边界。
- 张量签名必须与 `spec.yaml` 逐字一致；只声明一个输出。
- UB 208 KiB / L1 512 KiB / L0C 256 KiB 上限。

## 已有产物

- 生成物：`{spec.name}.py`
- 失败快照：`history/`
- CCE（若已 lowering）：`compile_result.cce.c`
""",
            encoding="utf-8",
        )
        self._write_debug(
            task_dir / "debug_log.md",
            f"## FIX_REQUEST — {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"- stage: {stage}\n- error: {error[-800:]}",
        )
        return path

    def _env_snapshot(self) -> dict[str, Any]:
        from ..runtime.tilelang_a5 import environment_probe

        try:
            return environment_probe(self.cfg)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _notes(
        self,
        sim: SimResult | None,
        board,
        compile_result: CompileResult | None,
        mode: str = "template",
        golden_source: str = "aiop-builtin",
    ) -> list[str]:
        notes: list[str] = []
        if mode == "open" and compile_result and compile_result.ok:
            notes.append(
                "open-ended mode: no A5 template family; the kernel was drafted by the LLM and "
                "still passed static + lowering + device compile gates. Tiling/UB assertions are "
                "not planned, only observed through lowering."
            )
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
            notes.append(f"L3 passed on backend={backend} (golden={golden_source}).")
            if golden_source != "aiop-builtin":
                notes.append(
                    "That L3 pass is self-referenced: the reference was also written by the LLM, "
                    "so agreement is not independent evidence. `verified` stays False."
                )
        if mode == "open" and sim is None:
            notes.append(
                "L3 skipped: no reference implementation is available for this operator "
                f"(golden_source={golden_source})."
            )
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
            f"- mode: `{result.get('mode', 'template')}`",
            f"- golden: `{result.get('golden_source', 'aiop-builtin')}`",
            f"- generator: `{result['generator']}`",
            f"- task: `{result['task_id']}`",
            "",
        ]
        contract = result.get("contract") or {}
        skills = result.get("skills") or {}
        audit = result.get("lowering_audit") or {}
        if contract:
            lines += [
                "## 契约（TaskContract）",
                "",
                f"- hash: `{contract.get('contract_hash', '')}`",
                f"- source: `{contract.get('source', '')}`",
                f"- determinism: `{contract.get('determinism')}` · open_mode: `{contract.get('open_mode')}`",
                f"- perf: `{json.dumps(contract.get('perf', {}), ensure_ascii=False)}`",
                f"- obligations: `{', '.join(contract.get('obligations', []))}`",
                "",
            ]
        if skills:
            names = ", ".join(item.get("name", "") for item in skills.get("skills", []))
            lines += ["## 技能", "", f"- selected: `{names}`", ""]
        lines += [
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
        if result.get("unsupported_reasons"):
            lines += ["## 不支持原因", ""]
            lines += [f"- {reason}" for reason in result["unsupported_reasons"]]
            lines += ["", "> 未生成 kernel，也未做任何编译/仿真/上板测量。", ""]
        if comp.get("error"):
            lines += ["```text", comp["error"][-2000:], "```", ""]
        if audit:
            lines += ["## Lowering 审计（计划 vs 实际）", "", f"- status: `{audit.get('status')}`"]
            lines += [
                f"- `{item['id']}` {item['status']}: {item['detail']}"
                for item in audit.get("checks", [])
            ]
            lines += [""]
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
        ]
        diagnosis = result.get("diagnosis") or {}
        if diagnosis:
            lines += ["## 诊断（失败聚类）", "", f"- verdict: `{diagnosis.get('verdict')}`"]
            for cluster in diagnosis.get("clusters", []):
                lines.append(
                    f"- `{cluster['loop']}` ×{cluster['count']}: "
                    f"{cluster['reason']} → {cluster['recommendation']}"
                )
            lines += [""]
        lines += [
            "## 事实与缺口",
            "",
        ]
        lines += [f"- {note}" for note in result.get("notes", [])] or ["- none"]
        lines += ["", "## 产物", ""]
        lines += [f"- {name}: `{path}`" for name, path in result.get("artifacts", {}).items()]
        return "\n".join(lines) + "\n"
