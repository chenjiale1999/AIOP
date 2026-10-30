"""Minimal HTTP service exposing the A5 operator agent.

Endpoints
---------
GET  /healthz
GET  /v1/env
POST /v1/search                 {"query": str, "k": int}
POST /v1/tasks                  {"spec"|"spec_yaml"|"requirement", "simulate"?, "board"?, "strategy"?}
GET  /v1/tasks
GET  /v1/tasks/{id}
GET  /v1/tasks/{id}/report
POST /v1/tasks/{id}/board       {"host"?, "force"?}

Long tasks run in a small thread pool; every response is JSON except the
markdown report. If AIOP_SERVICE_TOKEN is set, Bearer auth is required.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import yaml
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..agent.orchestrator import OperatorAgent
from ..board.runner import run_board
from ..config import load_config
from ..knowledge.index import default_index
from ..specs import TaskSpec, spec_from_requirement

EXECUTOR = ThreadPoolExecutor(max_workers=2)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


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
        if path == "/healthz":
            self._send(200, {"status": "ok", "version": "0.1.0"})
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
            task_id = str(data.get("task_id") or "")
            cfg = load_config()
            agent = OperatorAgent(cfg)

            def run_job() -> None:
                try:
                    agent.run(
                        spec,
                        task_id=task_id or None,
                        simulate=simulate,
                        board=board,
                        strategy=strategy,
                    )
                except Exception as exc:  # noqa: BLE001
                    with JOBS_LOCK:
                        JOBS[task_id] = {
                            "task_id": task_id,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-4000:],
                        }

            # Reserve the id synchronously when provided; otherwise let the agent derive it.
            if task_id:
                with JOBS_LOCK:
                    JOBS[task_id] = {"task_id": task_id, "status": "queued"}
            EXECUTOR.submit(run_job)
            self._send(202, {"task_id": task_id or "auto", "status": "queued"})
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
            spec = TaskSpec.from_yaml(spec_path)

            def run_board_job() -> None:
                result = run_board(task_dir, spec, cfg, force=bool(data.get("force", False)))
                (task_dir / "board_report.json").write_text(
                    json.dumps(result.to_dict(), indent=2), encoding="utf-8"
                )

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
        if data.get("requirement"):
            return spec_from_requirement(str(data["requirement"]))
        raise ValueError("provide spec, spec_yaml, or requirement")

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
