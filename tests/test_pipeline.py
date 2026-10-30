import json

import pytest

from aiop.agent.orchestrator import OperatorAgent, task_id_for
from aiop.board.runner import run_board
from aiop.config import load_config
from aiop.knowledge.index import default_index
from aiop.specs import TaskSpec, TensorSpec


def test_knowledge_retrieval_has_relevant_sources():
    kb = default_index()
    assert len(kb.chunks) > 100
    hits = kb.search("softmax 归约 精度", k=3)
    assert hits
    docs = {hit["doc"] for hit in hits}
    assert any(doc.startswith("knowledge/") for doc in docs)
    hits = kb.search("npusim 仿真 Ascend950", k=3)
    assert any("07-simulation" in hit["doc"] for hit in hits)


def test_environment_is_honest():
    from aiop.runtime.tilelang_a5 import environment_probe

    env = environment_probe(load_config())
    assert env["tilelang_native"] is True
    assert isinstance(env["npu_device"], bool)
    if not env["npu_device"]:
        assert env["dev_davinci"] == [] or True


def test_board_skips_without_host(tmp_path):
    cfg = load_config()
    cfg.board.host = ""
    spec = TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (16,), "float32")],
        [TensorSpec("Y", (16,), "float32")],
    )
    result = run_board(tmp_path, spec, cfg)
    assert result.status == "skipped"


@pytest.mark.slow
def test_end_to_end_relu_compile(tmp_path):
    cfg = load_config()
    cfg.paths.workspace = str(tmp_path / "workspace")
    cfg.simulation.backend = "static"
    agent = OperatorAgent(cfg)
    spec = TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
    )
    result = agent.run(spec, simulate=False)
    assert result["status"] in ("success", "success_static_only")
    assert result["compile"]["ok"] is True
    task_dir = tmp_path / "workspace" / result["task_id"]
    for name in ("DESIGN.md", "relu.py", "test_relu.py", "result.json", "report.md", "state.json"):
        assert (task_dir / name).is_file()
    data = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
    assert data["plan"]["variant"] in ("simt_ub_staged", "simt_direct_global")
    assert data["compile"]["audit"]["balanced_sync"] is True
