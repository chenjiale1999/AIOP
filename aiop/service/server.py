"""Minimal HTTP service exposing the A5 operator agent.

Endpoints
---------
GET  /                          service index and command catalog
GET  /healthz
GET  /v1/env
POST /v1/search                 {"query": str, "k": int}
POST /v1/tasks                  {"spec"|"spec_yaml"|"requirement"|"op",
                                 "workload"?, "dtype_case"?, "simulate"?, "board"?,
                                 "strategy"?, "search"?, "search_budget"?, "task_id"?}
GET  /v1/tasks
GET  /v1/tasks/{id}
GET  /v1/tasks/{id}/report
GET  /v1/tasks/{id}/log
POST /v1/tasks/{id}/board       {"host"?, "force"?}
POST /v1/commands               {"command": str, "args"?: [str], "config"?: str}
GET  /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/log      ?tail=N

``/v1/commands`` runs a whitelisted CLI command as a child process, streams its
stdout/stderr into a job log, and exposes the same log over HTTP. Long jobs run
in a small thread pool; every response is JSON except reports/logs. If
``AIOP_SERVICE_TOKEN`` is set, Bearer auth is required.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
import yaml
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..agent.orchestrator import OperatorAgent, task_id_for
from ..agent.progress import ProgressReporter
from ..board.runner import run_board
from ..config import PROJECT_ROOT, load_config
from ..knowledge.index import default_index
from ..manifest import bind_operator
from ..runtime.command_log import logger_for
from ..specs import TaskSpec, spec_from_requirement

EXECUTOR = ThreadPoolExecutor(max_workers=2)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
CMD_JOBS: dict[str, dict] = {}
CMD_LOCK = threading.Lock()

# `serve` is deliberately excluded: a service must not recursively start itself.
ALLOWED_COMMANDS = {
    "generate",
    "campaign",
    "prompt",
    "export",
    "check",
    "compile",
    "simulate",
    "board",
    "search",
    "env",
    "manifest",
    "eval",
}


def _jobs_dir(cfg) -> Path:
    path = Path(cfg.workspace) / "_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_job_meta(cfg, job: dict) -> None:
    try:
        path = _jobs_dir(cfg) / f"{job['job_id']}.json"
        path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _parse_args(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, list) and all(isinstance(item, (str, int, float)) for item in raw):
        return [str(item) for item in raw]
    raise ValueError("args must be a string or a list of scalars")


def start_command(cfg, data: dict) -> dict:
    """Validate and queue one whitelisted CLI command."""
    command = str(data.get("command") or "").strip()
    if command not in ALLOWED_COMMANDS:
        raise ValueError(
            f"unsupported command {command!r}; allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
        )
    args = _parse_args(data.get("args"))
    config = str(data.get("config") or "")
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job = {
        "job_id": job_id,
        "command": command,
        "args": args,
        "status": "queued",
        "exit_code": None,
        "task_id": None,
        "log": str(_jobs_dir(cfg) / f"{job_id}.log"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with CMD_LOCK:
        CMD_JOBS[job_id] = job
    _write_job_meta(cfg, job)
    queued = dict(job)
    EXECUTOR.submit(_run_command_job, job_id, cfg, command, args, config)
    return queued


def _run_command_job(job_id: str, cfg, command: str, args: list[str], config: str) -> None:
    log_path = Path(_jobs_dir(cfg)) / f"{job_id}.log"
    job = CMD_JOBS[job_id]
    job.update(status="running", updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write_job_meta(cfg, job)
    argv = [sys.executable, "-m", "aiop"]
    if config:
        argv += ["--config", config]
    argv += [command, *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    exit_code = 1
    task_id = None
    if command in {"check", "compile", "simulate", "board"} and args:
        task_id = Path(args[0]).name
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(
                f"===== job {job_id} {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"{' '.join(shlex.quote(part) for part in argv)} =====\n"
            )
            log.flush()
            print(f"[{job_id}] $ {' '.join(argv)}", flush=True)
            proc = subprocess.Popen(
                argv,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            if proc.stdout is not None:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{job_id}] {line.rstrip()}", flush=True)
                    match = re.search(r"^task:\s+(\S+)", line)
                    if match and task_id is None:
                        task_id = Path(match.group(1)).name
            exit_code = proc.wait()
    except Exception as exc:  # noqa: BLE001 - reported in the job record
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"[job error] {type(exc).__name__}: {exc}\n")
        print(f"[{job_id}] job error: {type(exc).__name__}: {exc}", flush=True)
    finally:
        job.update(
            status="passed" if exit_code == 0 else "failed",
            exit_code=exit_code,
            task_id=task_id,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        _write_job_meta(cfg, job)


def list_jobs(cfg) -> list[dict]:
    records: dict[str, dict] = {}
    with CMD_LOCK:
        records.update(CMD_JOBS)
    for path in sorted((Path(cfg.workspace) / "_jobs").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records.setdefault(str(data.get("job_id")), data)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(
        records.values(), key=lambda item: str(item.get("created_at", "")), reverse=True
    )


def get_job(cfg, job_id: str) -> dict | None:
    with CMD_LOCK:
        if job_id in CMD_JOBS:
            return dict(CMD_JOBS[job_id])
    path = Path(cfg.workspace) / "_jobs" / f"{job_id}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "aiop/0.1"

    def _send(self, code: int, payload, content_type: str = "application/json") -> None:
        if isinstance(payload, (dict, list)):
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        else:
            body = str(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        token = os.environ.get("AIOP_SERVICE_TOKEN", "")
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {token}"

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def do_GET(self) -> None:  # noqa: N802
        if not self._auth_ok():
            self._send(401, {"error": "unauthorized"})
            return
        path = urlparse(self.path).path
        cfg = load_config()
        if path == "/":
            self._send(
                200,
                {
                    "service": "aiop",
                    "version": "0.1.0",
                    "endpoints": [
                        "GET /",
                        "GET /healthz",
                        "GET /v1/env",
                        "POST /v1/search",
                        "POST /v1/tasks",
                        "GET /v1/tasks",
                        "GET /v1/tasks/{id}",
                        "GET /v1/tasks/{id}/report",
                        "GET /v1/tasks/{id}/log",
                        "POST /v1/tasks/{id}/board",
                        "POST /v1/commands",
                        "GET /v1/jobs",
                        "GET /v1/jobs/{job_id}",
                        "GET /v1/jobs/{job_id}/log",
                    ],
                    "commands": sorted(ALLOWED_COMMANDS),
                    "examples": {
                        "submit_task": 'POST /v1/tasks {"requirement":"relu on [128, 1000] float32","task_id":"demo"}',
                        "run_terminal_command": 'POST /v1/commands {"command":"generate","args":["--spec","examples/softmax.yaml","--no-simulate"]}',
                        "command_log": "GET /v1/jobs/{job_id}/log?tail=100",
                    },
                },
            )
            return
        if path == "/healthz":
            self._send(200, {"status": "ok", "version": "0.1.0"})
            return
        if path == "/v1/jobs":
            self._send(200, {"jobs": list_jobs(cfg)})
            return
        if path.startswith("/v1/jobs/"):
            parts = path.strip("/").split("/")
            job_id = parts[2]
            if not re.fullmatch(r"job-[0-9a-f]{8}", job_id):
                self._send(404, {"error": "job not found"})
                return
            if len(parts) == 4 and parts[3] == "log":
                log = Path(cfg.workspace) / "_jobs" / f"{job_id}.log"
                if not log.is_file():
                    self._send(404, {"error": "log not found"})
                    return
                text = log.read_text(encoding="utf-8", errors="ignore")
                tail_n = parse_qs(urlparse(self.path).query).get("tail", [""])[0]
                if tail_n.isdigit():
                    text = "".join(text.splitlines(keepends=True)[-int(tail_n) :])
                self._send(200, text, "text/plain; charset=utf-8")
                return
            job = get_job(cfg, job_id)
            if job:
                self._send(200, job)
            else:
                self._send(404, {"error": "job not found"})
            return
        if path == "/v1/env":
            from ..runtime.tilelang_a5 import environment_probe

            self._send(200, environment_probe(cfg))
            return
        if path == "/v1/tasks":
            items = []
            for result in sorted(cfg.workspace.glob("*/result.json")):
                try:
                    data = json.loads(result.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                items.append(
                    {
                        "task_id": data.get("task_id"),
                        "status": data.get("status"),
                        "verified": data.get("verified"),
                        "spec": (data.get("spec") or {}).get("name"),
                    }
                )
            self._send(200, {"tasks": items})
            return
        if path.startswith("/v1/tasks/"):
            parts = path.strip("/").split("/")
            task_id = parts[2]
            if len(parts) == 4 and parts[3] == "report":
                report = cfg.workspace / task_id / "report.md"
                if report.is_file():
                    self._send(200, report.read_text(encoding="utf-8"), "text/markdown; charset=utf-8")
                else:
                    self._send(404, {"error": "report not found"})
                return
            if len(parts) == 4 and parts[3] == "log":
                log = cfg.workspace / task_id / "run.log"
                if log.is_file():
                    self._send(
                        200,
                        log.read_text(encoding="utf-8", errors="ignore"),
                        "text/plain; charset=utf-8",
                    )
                else:
                    self._send(404, {"error": "log not found"})
                return
            result = cfg.workspace / task_id / "result.json"
            if result.is_file():
                self._send(200, json.loads(result.read_text(encoding="utf-8")))
                return
            state = cfg.workspace / task_id / "state.json"
            if state.is_file():
                self._send(202, json.loads(state.read_text(encoding="utf-8")))
                return
            job = JOBS.get(task_id)
            if job:
                self._send(202, job)
                return
            self._send(404, {"error": "task not found"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._auth_ok():
            self._send(401, {"error": "unauthorized"})
            return
        path = urlparse(self.path).path
        try:
            data = self._body()
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        if path == "/v1/commands":
            try:
                job = start_command(load_config(), data)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(202, job)
            return
        if path == "/v1/search":
            query = str(data.get("query", "")).strip()
            if not query:
                self._send(400, {"error": "query is required"})
                return
            hits = default_index().search(query, k=int(data.get("k", 5)))
            self._send(200, {"query": query, "hits": hits})
            return
        if path == "/v1/tasks":
            try:
                spec = self._parse_spec(data)
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": f"invalid spec: {exc}"})
                return
            simulate = bool(data.get("simulate", True))
            board = bool(data.get("board", False))
            strategy = str(data.get("strategy", "auto"))
            search = bool(data.get("search", False))
            search_budget = int(data.get("search_budget", 4))
            draft = str(data.get("draft") or "") or None
            task_id = str(data.get("task_id") or "") or task_id_for(spec)
            cfg = load_config()
            agent = OperatorAgent(cfg)

            def run_job() -> None:
                try:
                    result = agent.run(
                        spec,
                        task_id=task_id,
                        simulate=simulate,
                        board=board,
                        strategy=strategy,
                        search=search,
                        search_budget=search_budget,
                        draft=draft,
                        progress=True,
                    )
                    with JOBS_LOCK:
                        JOBS[task_id] = {
                            "task_id": task_id,
                            "status": result.get("status", "unknown"),
                        }
                except Exception as exc:  # noqa: BLE001
                    with JOBS_LOCK:
                        JOBS[task_id] = {
                            "task_id": task_id,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-4000:],
                        }

            with JOBS_LOCK:
                JOBS[task_id] = {"task_id": task_id, "status": "queued"}
            EXECUTOR.submit(run_job)
            self._send(202, {"task_id": task_id, "status": "queued"})
            return
        if path.startswith("/v1/tasks/") and path.endswith("/board"):
            parts = path.strip("/").split("/")
            task_id = parts[2]
            cfg = load_config()
            if data.get("host"):
                cfg.board.host = str(data["host"])
            task_dir = cfg.workspace / task_id
            spec_path = task_dir / "spec.yaml"
            if not spec_path.is_file():
                self._send(404, {"error": "task not found"})
                return
            if not cfg.steps.npu:
                self._send(
                    200,
                    {
                        "task_id": task_id,
                        "status": "skipped",
                        "reason": "npu step disabled by configuration (steps.npu=false)",
                    },
                )
                return
            spec = TaskSpec.from_yaml(spec_path)
            command_log = logger_for(
                task_dir,
                level=cfg.logging.command_level,
                echo=cfg.logging.echo_commands,
            )

            def run_board_job() -> None:
                reporter = ProgressReporter(task_dir, task_id)
                reporter.emit("board", "start", host=cfg.board.host or "unset")
                try:
                    result = run_board(
                        task_dir,
                        spec,
                        cfg,
                        force=bool(data.get("force", False)),
                        command_log=command_log,
                    )
                    (task_dir / "board_report.json").write_text(
                        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
                    )
                    reporter.emit(
                        "board",
                        result.status,
                        host=result.host or "unset",
                        error=result.error,
                    )
                except Exception as exc:  # noqa: BLE001 - reported in the log and job
                    reporter.emit("board", "failed", error=f"{type(exc).__name__}: {exc}")
                    raise

            EXECUTOR.submit(run_board_job)
            self._send(202, {"task_id": task_id, "status": "board_queued"})
            return
        self._send(404, {"error": "not found"})

    @staticmethod
    def _parse_spec(data: dict) -> TaskSpec:
        if data.get("spec"):
            return TaskSpec.from_dict(data["spec"])
        if data.get("spec_yaml"):
            return TaskSpec.from_dict(yaml.safe_load(data["spec_yaml"]))
        if data.get("op"):
            cfg = load_config()
            manifest = data.get("manifest") or cfg.paths.tileops_manifest
            return bind_operator(
                manifest,
                str(data["op"]),
                workload=data.get("workload", 0),
                dtype_case=int(data.get("dtype_case", 0)),
            )
        if data.get("requirement"):
            return spec_from_requirement(str(data["requirement"]))
        raise ValueError("provide spec, spec_yaml, op(+workload/dtype_case), or requirement")

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[aiop] {self.address_string()} {fmt % args}")


def serve(args) -> int:
    cfg = load_config()
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"aiop service listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
