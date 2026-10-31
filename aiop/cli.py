"""Command line interface for the aiop A5 operator agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent.campaign import run_negative_probe, write_campaign_report
from .agent.design import build_design, build_draft_prompt, build_open_design
from .agent.orchestrator import OperatorAgent, task_id_for
from .board.runner import run_board
from .checks.compiler import compile_kernel
from .checks.static import run_static
from .config import load_config
from .knowledge.index import default_index
from .manifest import bind_operator, describe_operator, list_operators
from .runtime.hardware import Plan, build_open_plan, build_plan
from .simulation.casematrix import input_key, plan_variants
from .simulation.runner import run_simulation
from .simulation.testgen import write_test
from .specs import TaskSpec, UnsupportedOperator, spec_from_requirement


def _load_spec(args) -> TaskSpec:
    manifest = getattr(args, "manifest", None)
    op = getattr(args, "op", None)
    if op and not manifest:
        manifest = load_config(getattr(args, "config", "") or None).paths.tileops_manifest
    if manifest and op:
        return bind_operator(
            manifest,
            op,
            workload=getattr(args, "workload", 0),
            dtype_case=getattr(args, "dtype_case", 0),
        )
    if manifest or op:
        raise SystemExit("--manifest and --op must be given together")
    if getattr(args, "spec", None):
        path = Path(args.spec)
        if path.suffix in (".yaml", ".yml"):
            return TaskSpec.from_yaml(path)
        return TaskSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if getattr(args, "requirement", None):
        return spec_from_requirement(args.requirement)
    raise SystemExit("provide --spec <file>, --requirement <text>, or --manifest <file> --op <name>")


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
        draft=args.draft or None,
        search=args.search,
        search_budget=args.search_budget,
        progress=not args.no_progress,
    )
    task_dir = cfg.workspace / result["task_id"]
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"status: {result['status']}  verified: {result['verified']}")
        print(f"task:   {task_dir}")
        if result["status"] == "blocked_unsupported":
            for reason in result.get("unsupported_reasons", []):
                print(f"unsupported: {reason}")
            print(f"design: {task_dir / 'DESIGN.md'}")
            return 2
        print(f"kernel: {result['artifacts']['kernel']}")
        print(f"report: {task_dir / 'report.md'}")
        for note in result.get("notes", []):
            print(f"note:   {note}")
    return 0 if result["status"] in ("success", "success_static_only") else 1


def cmd_campaign(args) -> int:
    """Run the full six-step campaign and archive one consolidated report."""
    cfg = load_config(args.config or None)
    spec = _load_spec(args)
    result = OperatorAgent(cfg).run(
        spec,
        task_id=args.task_id or None,
        simulate=args.simulate,
        board=args.board,
        strategy=args.strategy,
        search=args.search,
        search_budget=args.search_budget,
        progress=not args.no_progress,
    )
    task_dir = cfg.workspace / result["task_id"]
    negative: dict = {"status": "skipped", "reason": "no plan"}
    plan_data = result.get("plan") or {}
    if plan_data:
        fields = {field for field in Plan.__dataclass_fields__}
        plan = Plan(**{k: v for k, v in plan_data.items() if k in fields})
        negative = run_negative_probe(task_dir, spec, plan, cfg)
    report = write_campaign_report(task_dir, result, args.out or None, negative=negative)
    print(f"status: {result['status']}  verified: {result['verified']}")
    print(f"task:   {task_dir}")
    print(f"report: {report}")
    if negative.get("status") != "skipped":
        print(
            f"negative: {negative['status']} "
            f"(expected {negative.get('expected_rule')}, observed {negative.get('observed_rules')})"
        )
    for name in ("contract", "context_manifest", "frontier", "lowering_audit", "diagnosis"):
        path = (result.get("artifacts") or {}).get(name)
        if path:
            print(f"{name}: {path}")
    for note in result.get("notes", []):
        print(f"note:   {note}")
    ok = result["status"] in ("success", "success_static_only")
    return 0 if ok and negative.get("status") != "failed" else 1


def cmd_export(args) -> int:
    """Write a single-file, path-independent test with the kernel embedded."""
    cfg = load_config(args.config or None)
    task_dir, spec, plan = _load_task(args.task)
    kernel = task_dir / f"{spec.name}.py"
    if not kernel.is_file():
        print(f"kernel not found: {kernel}", file=sys.stderr)
        return 1
    data: dict = {}
    result_path = task_dir / "result.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text(encoding="utf-8"))
    golden_override = None
    if data.get("golden_source") in ("llm-self-referenced", "agent-self-referenced"):
        ref = task_dir / "golden_ref.py"
        if ref.is_file():
            text = ref.read_text(encoding="utf-8")
            prefix = "def golden_body():\n"
            if text.startswith(prefix):
                golden_override = text[len(prefix) :]
    kernel_text = kernel.read_text(encoding="utf-8")
    # Only deterministic-template kernels can be re-rendered for other
    # shape/dtype contracts; LLM/draft kernels keep their single bound case.
    enumerate_variants = (
        "deterministic template" in kernel_text[:1500]
        or data.get("mode") == "template"
        or data.get("generator") == "template"
    )
    variants = plan_variants(
        spec,
        plan,
        kernel_text,
        golden_override=golden_override,
        enumerate_variants=enumerate_variants,
    )
    out_dir = Path(args.out) if args.out else task_dir / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    test = write_test(
        spec,
        plan,
        out_dir,
        cann_home=cfg.cann.home,
        device_flags=cfg.compile.device_flags,
        warmup=cfg.simulation.warmup,
        repeats=cfg.simulation.repeats,
        tilelang_repo=cfg.paths.tilelang_repo,
        golden_override=golden_override,
        kernel_source=kernel_text,
        variants=variants,
    )
    available = sum(1 for variant in variants if variant.available)
    unique = len({input_key(variant.spec) for variant in variants})
    print(f"test: {test}")
    print(f"size: {test.stat().st_size} bytes")
    print(f"variants: {available}/{len(variants)} (shape/dtype combos: {unique})")
    for variant in variants:
        if not variant.available:
            print(
                f"warning: workload={variant.workload} dtype_case={variant.dtype_case} "
                f"unavailable: {variant.error}",
                file=sys.stderr,
            )
    print(f"run:  python {test.name} --device npu")
    print(f"copy: scp {test} <user>@<a5-host>:/tmp/")
    return 0


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
        try:
            data["compile_probe"] = compile_probe(cfg)
        except Exception as exc:  # noqa: BLE001 - report, never crash the probe
            data["compile_probe"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def cmd_prompt(args) -> int:
    """Prepare a drafting request so an IDE agent (Trae) can play the LLM role."""
    cfg = load_config(args.config or None)
    spec = _load_spec(args)
    hits = default_index().search(f"{spec.family} {spec.name} tiling 内存 精度 性能", k=5)
    try:
        plan = build_plan(spec)
        design = build_design(spec, plan, hits)
    except UnsupportedOperator as exc:
        plan = build_open_plan(spec)
        design = build_open_design(spec, exc.reasons, hits, golden_source="unavailable")
    task_id = task_id_for(spec)
    task_dir = cfg.workspace / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    spec.save(task_dir / "spec.yaml")
    (task_dir / "DESIGN.md").write_text(design, encoding="utf-8")
    prompt = build_draft_prompt(spec, design, hits)
    path = task_dir / "PROMPT.md"
    path.write_text(prompt, encoding="utf-8")
    print(f"task:   {task_dir}")
    print(f"prompt: {path}")
    if getattr(args, "op", None):
        cmd = (
            f"python -m aiop generate --op {args.op} --workload {args.workload} "
            f"--dtype-case {args.dtype_case} --strategy draft"
        )
    elif getattr(args, "spec", None):
        cmd = f"python -m aiop generate --spec {args.spec} --strategy draft"
    else:
        cmd = "python -m aiop generate --requirement '<text>' --strategy draft"
    print(f"next:   {cmd}")
    print()
    print("下一步：让 IDE Agent（Trae）按 PROMPT.md 写 draft.py，然后执行上面这条命令。")
    return 0


def cmd_manifest(args) -> int:
    source = args.manifest or load_config(args.config or None).paths.tileops_manifest
    if args.manifest_command == "list":
        rows = list_operators(source)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return 0
        for row in rows:
            family = row["aiop_family"] or "-"
            print(
                f"{row['op']:<32} {str(row['manifest_family']):<16} "
                f"rows={row['workloads']:<3} cases={row['cases']:<3} aiop={family}"
            )
        supported = [r for r in rows if r["aiop_family"]]
        print(f"\n{len(supported)}/{len(rows)} entries map onto an aiop A5 template family")
        return 0
    if args.manifest_command == "show":
        print(json.dumps(describe_operator(source, args.op), indent=2, ensure_ascii=False))
        return 0
    if args.manifest_command == "convert":
        spec = bind_operator(
            source, args.op, workload=args.workload, dtype_case=args.dtype_case
        )
        if args.out:
            spec.save(args.out)
            print(f"spec: {args.out}")
        else:
            print(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False))
        if not spec.supported:
            print("unsupported: " + "; ".join(spec.unsupported_reasons), file=sys.stderr)
            return 1
        return 0
    return 1


def cmd_eval(args) -> int:
    from .eval.run_benchmark import run_benchmark

    return run_benchmark(args)


def cmd_serve(args) -> int:
    from .service.server import serve

    return serve(args)


def _workload_selector(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aiop", description="Ascend A5 TileLang operator generation agent")
    parser.add_argument("--config", default="")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="run the full generation pipeline")
    p.add_argument("--spec", help="TaskSpec YAML/JSON")
    p.add_argument("--requirement", help="natural-language requirement")
    p.add_argument("--manifest", help="TileOPs manifest yaml (or its spec directory)")
    p.add_argument("--op", help="manifest entry name, e.g. RMSNormFwdOp")
    p.add_argument("--workload", default=0, type=_workload_selector, help="workload row index or label")
    p.add_argument("--dtype-case", type=int, default=0, help="dtype_cases index of that row")
    p.add_argument("--task-id", default="")
    p.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "template", "llm", "draft"],
        help="auto: verified template when the family has one, else LLM (open mode); "
        "template: templates only; llm: LLM first, template as fallback; "
        "draft: use an externally written draft.py (Trae/IDE agent), no API key needed",
    )
    p.add_argument("--draft", default="", help="draft kernel path (default <task>/draft.py)")
    p.add_argument("--search", action="store_true", help="bounded plan search before generation")
    p.add_argument("--search-budget", type=int, default=4, help="number of candidate plans to gate")
    p.add_argument("--no-progress", action="store_true", help="do not stream stage progress to stderr")
    p.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--board", action="store_true")
    p.add_argument("--force-board", action="store_true")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser(
        "campaign",
        help="run contract → skills → search → gates → simulation → board and write one report",
    )
    p.add_argument("--spec", help="TaskSpec YAML/JSON")
    p.add_argument("--requirement", help="natural-language requirement")
    p.add_argument("--manifest", help="TileOPs manifest yaml (or its spec directory)")
    p.add_argument("--op", help="manifest entry name")
    p.add_argument("--workload", default=0, type=_workload_selector)
    p.add_argument("--dtype-case", type=int, default=0)
    p.add_argument("--task-id", default="")
    p.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "template", "llm", "draft"],
    )
    p.add_argument("--search", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--search-budget", type=int, default=4)
    p.add_argument("--no-progress", action="store_true", help="do not stream stage progress to stderr")
    p.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--board", action="store_true")
    p.add_argument("--out", default="", help="optional campaign report path")
    p.set_defaults(func=cmd_campaign)

    p = sub.add_parser("prompt", help="write PROMPT.md so an IDE agent can draft the kernel")
    p.add_argument("--spec", help="TaskSpec YAML/JSON")
    p.add_argument("--requirement", help="natural-language requirement")
    p.add_argument("--manifest", help="TileOPs manifest yaml (or its spec directory)")
    p.add_argument("--op", help="manifest entry name")
    p.add_argument("--workload", default=0, type=_workload_selector)
    p.add_argument("--dtype-case", type=int, default=0)
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser(
        "export",
        help="write a standalone test_<op>.py with the kernel embedded",
    )
    p.add_argument("task", help="task id or workspace/<task>")
    p.add_argument("--out", default="", help="output directory (default <task>/export)")
    p.set_defaults(func=cmd_export)

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

    p = sub.add_parser("manifest", help="inspect / bind TileOPs manifest entries")
    p.add_argument("--manifest", default="", help="manifest yaml or spec directory")
    msub = p.add_subparsers(dest="manifest_command", required=True)

    m = msub.add_parser("list", help="list entries and their aiop family mapping")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_manifest)

    m = msub.add_parser("show", help="dump one entry's signature and workload rows")
    m.add_argument("op")
    m.set_defaults(func=cmd_manifest)

    m = msub.add_parser("convert", help="bind one workload row into a TaskSpec")
    m.add_argument("op")
    m.add_argument("--workload", default=0, type=_workload_selector)
    m.add_argument("--dtype-case", type=int, default=0)
    m.add_argument("--out", default="")
    m.set_defaults(func=cmd_manifest)

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
