"""TaskRegistry / TaskContract: the canonical operator contract.

The contract freezes everything the rest of the pipeline must agree on before
any code is written: semantics, ABI, precision, performance target, hard
resource limits and the verification obligations inherited from skills.

It is the aiop counterpart of the ``TaskRegistry`` module in the reference
architecture (``some_ref.md``): downstream stages consume the contract instead
of re-deriving facts from prose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..runtime.hardware import (
    AIC_CORES,
    AIV_CORES,
    ALIGN_BYTES,
    L0A_BUDGET_BYTES,
    L0B_BUDGET_BYTES,
    L0C_BUDGET_BYTES,
    L1_BUDGET_BYTES,
    SS_BUDGET_BYTES,
    UB_BUDGET_BYTES,
    VECTOR_BITS,
    Plan,
)
from ..specs import DTYPE_BYTES, TaskSpec, TensorSpec


@dataclass
class TensorContract:
    name: str
    direction: str  # input | output
    shape: list[int]
    dtype: str
    role: str = "compute"
    optional: bool = False
    mutated: bool = False
    contiguous: bool = False
    requires: list[str] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * DTYPE_BYTES.get(self.dtype, 0)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "direction": self.direction,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "role": self.role,
            "nbytes": self.nbytes,
        }
        if self.optional:
            data["optional"] = True
        if self.mutated:
            data["mutated"] = True
        if self.contiguous:
            data["contiguous"] = True
        if self.requires:
            data["requires"] = list(self.requires)
        return data


@dataclass
class TaskContract:
    op: str
    family: str
    semantics: str
    source: str
    inputs: list[TensorContract]
    outputs: list[TensorContract]
    precision: dict[str, float]
    perf: dict[str, Any]
    determinism: bool
    open_mode: bool
    obligations: list[str]
    limits: dict[str, int]
    plan_facts: dict[str, Any]
    headroom: dict[str, int]
    contract_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = {
            "op": self.op,
            "family": self.family,
            "semantics": self.semantics,
            "source": self.source,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "precision": self.precision,
            "perf": self.perf,
            "determinism": self.determinism,
            "open_mode": self.open_mode,
            "obligations": list(self.obligations),
            "limits": dict(self.limits),
            "plan_facts": dict(self.plan_facts),
            "headroom": dict(self.headroom),
            "contract_hash": self.contract_hash,
        }
        return data


def _tensor_contract(tensor: TensorSpec, direction: str) -> TensorContract:
    return TensorContract(
        name=tensor.name,
        direction=direction,
        shape=list(tensor.shape),
        dtype=tensor.dtype,
        role=tensor.role,
        optional=tensor.optional,
        mutated=tensor.mutated,
        contiguous=tensor.contiguous,
        requires=list(tensor.requires),
    )


def _plan_facts(spec: TaskSpec, plan: Plan | None) -> dict[str, Any]:
    if plan is None:
        return {"variant": "none", "declared": False}
    facts: dict[str, Any] = {
        "declared": True,
        "variant": plan.variant,
        "blocks": plan.blocks,
        "tile": plan.tile,
        "rows": plan.rows,
        "threads": plan.threads,
        "tile_extent": plan.tile_extent,
        "num_stages": plan.num_stages,
        "iterations": plan.iterations,
        "tail_mode": plan.tail_mode,
        "block_m": plan.block_m,
        "block_n": plan.block_n,
        "block_k": plan.block_k,
        "epilogue": plan.epilogue,
        "out_cast": plan.out_cast,
        "ub_used": plan.ub_used,
    }
    if spec.family == "gemm":
        elem = DTYPE_BYTES[spec.inputs[0].dtype]
        facts["l1_used"] = (plan.block_m + plan.block_n) * plan.block_k * elem * plan.num_stages
        facts["l0c_used"] = plan.block_m * plan.block_n * 4
    else:
        facts["l1_used"] = 0
        facts["l0c_used"] = 0
    return facts


def _headroom(facts: dict[str, Any]) -> dict[str, int]:
    return {
        "ub_bytes": UB_BUDGET_BYTES - int(facts.get("ub_used", 0)),
        "l1_bytes": L1_BUDGET_BYTES - int(facts.get("l1_used", 0)),
        "l0c_bytes": L0C_BUDGET_BYTES - int(facts.get("l0c_used", 0)),
    }


def build_contract(
    spec: TaskSpec,
    plan: Plan | None = None,
    *,
    obligations: list[str] | None = None,
    open_mode: bool = False,
    source: str | None = None,
) -> TaskContract:
    """Freeze the operator contract consumed by generate/search/evaluate."""
    if source is None:
        source = "tileops-manifest" if spec.manifest else "task-spec"
    plan_facts = _plan_facts(spec, plan)
    limits = {
        "ub_bytes": UB_BUDGET_BYTES,
        "l1_bytes": L1_BUDGET_BYTES,
        "l0c_bytes": L0C_BUDGET_BYTES,
        "l0a_bytes": L0A_BUDGET_BYTES,
        "l0b_bytes": L0B_BUDGET_BYTES,
        "ss_bytes": SS_BUDGET_BYTES,
        "aiv_cores": AIV_CORES,
        "aic_cores": AIC_CORES,
        "vector_bits": VECTOR_BITS,
        "align_bytes": ALIGN_BYTES,
    }
    perf = spec.perf.to_dict()
    perf["enforced"] = perf.get("target") is not None
    contract = TaskContract(
        op=spec.name,
        family=spec.family,
        semantics=spec.formula or f"{spec.family}:{spec.name}",
        source=source,
        inputs=[_tensor_contract(t, "input") for t in spec.inputs],
        outputs=[_tensor_contract(t, "output") for t in spec.outputs],
        precision=spec.precision.to_dict() if spec.precision else {},
        perf=perf,
        # Templates are deterministic; external drafts are not guaranteed to be.
        determinism=not open_mode,
        open_mode=open_mode,
        obligations=list(obligations or []),
        limits=limits,
        plan_facts=plan_facts,
        headroom=_headroom(plan_facts),
    )
    digest = hashlib.sha256(
        json.dumps(contract.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    contract.contract_hash = digest
    return contract


def write_contract(task_dir: str | Path, contract: TaskContract) -> Path:
    path = Path(task_dir) / "contract.json"
    path.write_text(json.dumps(contract.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
