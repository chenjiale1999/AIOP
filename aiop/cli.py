"""Command line interface for the aiop A5 operator agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent.orchestrator import OperatorAgent, task_id_for
from .board.runner import run_board
from .checks.compiler import compile_kernel
from .checks.static import run_static
from .config import load_config
from .knowledge.index import default_index
from .runtime.hardware import Plan, build_plan
from .simulation.runner import run_simulation
from .specs import TaskSpec, spec_from_requirement


def _load_spec(args) -> TaskSpec:
    if getattr(args, "spec", None):
        path = Path(args.spec)
        if path.suffix in (".yaml", ".yml"):
            return TaskSpec.from_yaml(path)
        return TaskSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if getattr(args, "requirement", None):
        return spec_from_requirement(args.requirement)
    raise SystemExit("provide --spec <file> or --requirement <text>")


def _load_task(task_arg: str) -> tuple[Path, TaskSpec, Plan]:
    task_dir = Path(task_arg)
    if not task_dir.is_dir():
        task_dir = load_config().workspace / task_arg
    spec = TaskSpec.from_yaml(task_dir / "spec.yaml")
    result_path = task_dir / "result.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        plan_data = data.get("plan") or {}
        fields = {f for f in Plan.__dataclass_fields__}
        plan = Plan(**{k: v for k, v in plan_data.items() if k in fields})
    else:
        plan = build_plan(spec)
    return task_dir, spec, plan


def cmd_generate(args) -> int:
    cfg = load_config(args.config or None)
    spec = _load_spec(args)
    agent = OperatorAgent(cfg)
    result = agent.run(
        spec,
        task_id=args.task_id,
        simulate=args.simulate,
        board=args.board,
        force_board=args.force_board,
        strategy=args.strategy,
        reuse_workspace=args.reuse,
    )
    task_dir = cfg.workspace / result["task_id"]
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"status: {result['status']}  verified: {result['verified']}")
        print(f"task:   {task_dir}")
        print(f"kernel: {result['artifacts']['kernel']}")
        print(f"report: {task_dir / 'report.md'}")
        for note in result.get("notes", []):
            print(f"note:   {note}")
    return 0 if result["status"] in ("success", "success_static_only") else 1


def cmd_check(args) -> int:
    cfg = load_config(args.config or None)
    task_dir, spec, plan = _load_task(args.task)
    report = run_static(spec, plan, task_dir / f"{spec.name}.py", cfg)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.ok else 1


def cmd_compile(args) -> int:
    cfg = load_config(args.config or None)
    task_dir, spec, plan = _load_task(args.task)
    result = compile_kernel(task_dir / f"{spec.name}.py", task_dir, cfg)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


def cmd_simulate(args) -> int:
    cfg = load_config(args.config or None)
    task_dir, spec, plan = _load_task(args.task)
    result = run_simulation(task_dir, task_dir / f"test_{spec.name}.py", spec, cfg)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.passed or result.status == "unavailable" else 1


def cmd_board(args) -> int:
    cfg = load_config(args.config or None)
    if args.host:
        cfg.board.host = args.host
    task_dir, spec, plan = _load_task(args.task)
    result = run_board(task_dir, spec, cfg, force=args.force)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.status in ("passed", "skipped") else 1


def cmd_search(args) -> int:
    kb = default_index()
    hits = kb.search(args.query, k=args.k)
    for hit in hits:
        print(f"[{hit['score']:.2f}] {hit['doc']} › {hit['heading']}")
        snippet = hit["text"].strip().splitlines()[:3]
        for line in snippet:
            print(f"    {line[:160]}")
    return 0


def cmd_env(args) -> int:
    from .runtime.tilelang_a5 import compile_probe, environment_probe

    cfg = load_config(args.config or None)
    data = environment_probe(cfg)
    if args.probe_compile:
        data["compile_probe"] = compile_probe(cfg)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def cmd_eval(args) -> int:
    from .eval.run_benchmark import run_benchmark

    return run_benchmark(args)


def cmd_serve(args) -> int:
    from .service.server import serve

    return serve(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aiop", description="Ascend A5 TileLang operator generation agent")
    parser.add_argument("--config", default="")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="run the full generation pipeline")
    p.add_argument("--spec", help="TaskSpec YAML/JSON")
    p.add_argument("--requirement", help="natural-language requirement")
    p.add_argument("--task-id", default="")
    p.add_argument("--strategy", default="auto", choices=["auto", "template", "llm"])
    p.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--board", action="store_true")
    p.add_argument("--force-board", action="store_true")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("check", help="static checks for an existing task")
    p.add_argument("task")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("compile", help="lower + device compile an existing task")
    p.add_argument("task")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("simulate", help="run L3 simulation for an existing task")
    p.add_argument("task")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("board", help="run L4 board measurement for an existing task")
    p.add_argument("task")
    p.add_argument("--host", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("search", help="search the A5 knowledge base")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("env", help="probe the local A5 toolchain")
    p.add_argument("--probe-compile", action="store_true")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("eval", help="run the operator benchmark suite")
    p.add_argument("--cases", default="")
    p.add_argument("--out", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-simulate", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="start the HTTP service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
