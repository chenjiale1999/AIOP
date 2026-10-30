"""Benchmark runner: measures generation pass rates over the operator suite."""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from ..agent.orchestrator import OperatorAgent, task_id_for
from ..config import PROJECT_ROOT, load_config
from ..manifest import bind_operator
from ..specs import TaskSpec


def _is_manifest_case(item: object) -> bool:
    return isinstance(item, dict) and "op" in item and "family" not in item


def load_cases(path: str | Path | None = None) -> list[TaskSpec]:
    path = Path(path) if path else PROJECT_ROOT / "aiop" / "eval" / "cases.yaml"
    if path.is_dir():
        items = []
        for file in sorted(path.glob("*.y*ml")):
            items.extend(load_cases(file))
        return items
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = data.get("cases", data) if isinstance(data, dict) else data
    return [
        bind_operator(
            item.get("manifest"),
            item["op"],
            workload=item.get("workload", 0),
            dtype_case=item.get("dtype_case", 0),
        )
        if _is_manifest_case(item)
        else TaskSpec.from_dict(item)
        for item in raw
    ]


def run_benchmark(args) -> int:
    cfg = load_config(getattr(args, "config", "") or None)
    cases = load_cases(getattr(args, "cases", "") or "")
    limit = int(getattr(args, "limit", 0) or 0)
    if limit:
        cases = cases[:limit]
    simulate = not bool(getattr(args, "no_simulate", False))
    agent = OperatorAgent(cfg)
    started = time.time()
    results = []
    for index, spec in enumerate(cases, 1):
        task_id = task_id_for(spec)
        print(f"[{index}/{len(cases)}] {spec.name} {tuple(spec.input.shape)} {spec.input.dtype} ...", flush=True)
        try:
            result = agent.run(spec, task_id=task_id, simulate=simulate, board=False)
            results.append(
                {
                    "task_id": task_id,
                    "name": spec.name,
                    "family": spec.family,
                    "shape": list(spec.input.shape),
                    "dtype": spec.input.dtype,
                    "status": result["status"],
                    "verified": result["verified"],
                    "static_ok": (result.get("static") or {}).get("ok"),
                    "compile_ok": (result.get("compile") or {}).get("ok"),
                    "sim_status": (result.get("simulation") or {}).get("status"),
                    "elapsed_s": result["elapsed_s"],
                }
            )
            print(f"    -> {result['status']} (static={results[-1]['static_ok']} compile={results[-1]['compile_ok']} sim={results[-1]['sim_status']})")
        except Exception as exc:  # noqa: BLE001 - benchmark must continue
            results.append(
                {
                    "task_id": task_id,
                    "name": spec.name,
                    "family": spec.family,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"    -> error: {exc}")

    total = len(results)
    static_pass = sum(1 for r in results if r.get("static_ok"))
    compile_pass = sum(1 for r in results if r.get("compile_ok"))
    sim_avail = [r for r in results if r.get("sim_status") in ("passed", "failed")]
    sim_pass = sum(1 for r in sim_avail if r["sim_status"] == "passed")
    generation_pass = static_pass / total if total else 0.0
    compile_rate = compile_pass / total if total else 0.0
    summary = {
        "total": total,
        "static_pass": static_pass,
        "compile_pass": compile_pass,
        "sim_available": len(sim_avail),
        "sim_pass": sim_pass,
        "generation_pass_rate": round(generation_pass, 4),
        "compile_pass_rate": round(compile_rate, 4),
        "simulation_pass_rate": round(sim_pass / len(sim_avail), 4) if sim_avail else None,
        "simulate_requested": simulate,
        "elapsed_s": round(time.time() - started, 2),
        "cases": results,
    }
    out = Path(getattr(args, "out", "") or (PROJECT_ROOT / "reports" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    print(f"report: {out}")
    return 0 if generation_pass >= 0.9 and compile_rate >= 0.9 else 1


def _markdown(summary: dict) -> str:
    lines = [
        "# A5 算子生成基准报告",
        "",
        f"- 用例总数：{summary['total']}",
        f"- 静态检查通过：{summary['static_pass']}（{summary['generation_pass_rate']:.1%}）",
        f"- 设备编译通过：{summary['compile_pass']}（{summary['compile_pass_rate']:.1%}）",
        f"- 仿真可用/通过：{summary['sim_available']}/{summary['sim_pass']}"
        + (f"（{summary['simulation_pass_rate']:.1%}）" if summary["simulation_pass_rate"] is not None else "（本机无可用仿真后端）"),
        f"- 总耗时：{summary['elapsed_s']}s",
        "",
        "| 算子 | shape | dtype | status | static | compile | sim |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| {case['name']} | {case.get('shape', '')} | {case.get('dtype', '')} | "
            f"{case['status']} | {case.get('static_ok')} | {case.get('compile_ok')} | {case.get('sim_status')} |"
        )
    return "\n".join(lines) + "\n"
