"""HTTP command submission, job logs and manifest task submission."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from aiop.service.server import Handler


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("AIOP_WORKSPACE", str(tmp_path / "workspace"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _request(method: str, url: str, body: dict | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8")
            return response.status, text
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _wait(url: str, timeout: float = 40.0) -> tuple[int, object]:
    deadline = time.time() + timeout
    code, body = 0, ""
    while time.time() < deadline:
        code, body = _request("GET", url)
        if code == 200:
            return code, body
        time.sleep(0.5)
    return code, body


def _wait_job(url: str, timeout: float = 40.0) -> tuple[int, object]:
    deadline = time.time() + timeout
    code, body = 0, ""
    while time.time() < deadline:
        code, body = _request("GET", url)
        if code == 200:
            try:
                status = json.loads(body).get("status")
            except json.JSONDecodeError:
                status = None
            if status in ("passed", "failed"):
                return code, body
        time.sleep(0.5)
    return code, body


def test_root_index_lists_commands(service):
    code, body = _request("GET", f"{service}/")
    assert code == 200
    data = json.loads(body)
    assert data["service"] == "aiop"
    assert "generate" in data["commands"]
    assert "serve" not in data["commands"]
    assert any("POST /v1/commands" in item for item in data["endpoints"])


def test_command_submission_writes_job_log(service):
    code, body = _request(
        "POST",
        f"{service}/v1/commands",
        {"command": "search", "args": ["UB 容量", "-k", "1"]},
    )
    assert code == 202, body
    job = json.loads(body)
    assert job["status"] == "queued"
    assert job["command"] == "search"

    code, body = _wait_job(f"{service}/v1/jobs/{job['job_id']}")
    assert code == 200, body
    finished = json.loads(body)
    assert finished["status"] == "passed", finished
    assert finished["exit_code"] == 0

    code, log = _request("GET", f"{service}/v1/jobs/{job['job_id']}/log?tail=50")
    assert code == 200
    assert "knowledge/" in log or "score" in log


def test_invalid_command_is_rejected(service):
    code, body = _request(
        "POST", f"{service}/v1/commands", {"command": "serve", "args": []}
    )
    assert code == 400
    assert "unsupported command" in body


def test_manifest_op_can_be_submitted_as_task(service):
    code, body = _request(
        "POST",
        f"{service}/v1/tasks",
        {
            "op": "GQADenseFwdOp",
            "workload": 0,
            "dtype_case": 0,
            "simulate": False,
            "task_id": "gqa-http",
        },
    )
    assert code == 202, body
    assert json.loads(body)["task_id"] == "gqa-http"

    code, body = _wait(f"{service}/v1/tasks/gqa-http", timeout=60.0)
    assert code == 200, body
    result = json.loads(body)
    assert result["status"] == "blocked_unsupported"
    assert result["task_id"] == "gqa-http"
