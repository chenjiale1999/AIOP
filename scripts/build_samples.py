"""Build the verified A5 sample library under samples/.

Each sample is produced by the same agent pipeline used in production and keeps
DESIGN.md, kernel, self-contained test, compile evidence and report.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiop.agent.orchestrator import OperatorAgent  # noqa: E402
from aiop.config import PROJECT_ROOT, load_config  # noqa: E402
from aiop.specs import TaskSpec, TensorSpec  # noqa: E402

SAMPLES = {
    "elementwise_add": TaskSpec(
        "add",
        "elementwise_binary",
        [TensorSpec("A", (1048576,), "float32"), TensorSpec("B", (1048576,), "float32")],
        [TensorSpec("C", (1048576,), "float32")],
        formula="C = A + B",
    ),
    "relu_bf16": TaskSpec(
        "relu",
        "elementwise_unary",
        [TensorSpec("X", (1024, 4096), "bfloat16")],
        [TensorSpec("Y", (1024, 4096), "bfloat16")],
        formula="Y = max(X, 0)",
    ),
    "reduce_sum": TaskSpec(
        "reduce_sum",
        "row_reduce",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128,), "float32")],
        formula="Y[i] = sum_j X[i, j]",
    ),
    "softmax": TaskSpec(
        "softmax",
        "softmax",
        [TensorSpec("X", (128, 1000), "float32")],
        [TensorSpec("Y", (128, 1000), "float32")],
        formula="softmax(x) = exp(x - max) / sum(exp(x - max))",
    ),
    "rmsnorm": TaskSpec(
        "rmsnorm",
        "rmsnorm",
        [TensorSpec("X", (128, 4096), "float32"), TensorSpec("W", (4096,), "float32")],
        [TensorSpec("Y", (128, 4096), "float32")],
        formula="y = x / sqrt(mean(x^2) + eps) * w",
        attrs={"eps": 1e-6},
    ),
    "layernorm": TaskSpec(
        "layernorm",
        "layernorm",
        [
            TensorSpec("X", (128, 1000), "float32"),
            TensorSpec("W", (1000,), "float32"),
            TensorSpec("B", (1000,), "float32"),
        ],
        [TensorSpec("Y", (128, 1000), "float32")],
        formula="y = (x - mean) / sqrt(var + eps) * w + b",
        attrs={"eps": 1e-6},
    ),
    "gemm_bf16": TaskSpec(
        "gemm",
        "gemm",
        [TensorSpec("A", (256, 256), "bfloat16"), TensorSpec("B", (448, 256), "bfloat16")],
        [TensorSpec("C", (256, 448), "float32")],
        formula="C = A @ B.T",
    ),
    "gemm_relu": TaskSpec(
        "gemm_relu",
        "gemm",
        [TensorSpec("A", (256, 128), "bfloat16"), TensorSpec("B", (224, 128), "bfloat16")],
        [TensorSpec("C", (256, 224), "float32")],
        formula="C = relu(A @ B.T)",
        attrs={"epilogue": "relu"},
    ),
}


def main() -> int:
    cfg = load_config()
    cfg.simulation.backend = "static"
    agent = OperatorAgent(cfg)
    out_root = PROJECT_ROOT / "samples"
    rows = []
    for name, spec in SAMPLES.items():
        task_id = f"sample-{name}"
        result = agent.run(spec, task_id=task_id, simulate=False)
        src = cfg.workspace / task_id
        dst = out_root / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for file in ("DESIGN.md", "spec.yaml", f"{spec.name}.py", f"test_{spec.name}.py", "result.json", "report.md", "manifest.json"):
            if (src / file).is_file():
                shutil.copy2(src / file, dst / file)
        if (src / "compile_result.cce.c").is_file():
            shutil.copy2(src / "compile_result.cce.c", dst / "kernel.cce.c")
        rows.append(
            {
                "name": name,
                "op": spec.name,
                "family": spec.family,
                "shape": list(spec.input.shape),
                "dtype": spec.input.dtype,
                "status": result["status"],
                "compile": bool(result["compile"] and result["compile"]["ok"]),
                "verified": result["verified"],
            }
        )
        print(f"{name}: {result['status']} compile={rows[-1]['compile']}")
    (out_root / "index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0 if all(row["compile"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
