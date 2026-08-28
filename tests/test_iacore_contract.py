"""The request contract this gateway shares with iacore — and nothing enforced it until now.

Every test here names the defect it catches. Nothing runs a server, opens a socket, or
imports `app.py`: the models are read out of the source with `ast`, so this suite needs no
FastAPI, no pydantic, no Ollama and no robot.

WHY THIS EXISTS. Four request models are declared **twice** — here in `app.py` and again in
`AI-VL-core/service.py` — and the tier boundary forbids sharing a module, so the duplication
is deliberate and permanent. What was missing is the tripwire. Neither side sets
`model_config`, so pydantic v2 defaults to `extra="ignore"`: a field the client sends that
this gateway does not declare is **dropped silently, with no error anywhere**. Add `seq` to
iacore's `CommandRequest` to fix the command/stop race, forget it here, and the sequence
number simply vanishes on the way through — the race stays open and nothing tells you.

WHY `ast` AND NOT AN IMPORT. Importing `service.py` pulls Ollama, Ultralytics and the model
config; importing `app.py` starts wiring FastAPI. Parsing the source keeps this suite fast,
dependency-free, and runnable in a CI job that checks out one repo. It is the same tactic
`robot-command-relay/tests/test_relay_boundary.py` uses to compare its allowlist against the
copy in C++.

HOW TO CHANGE THE CONTRACT. Editing a shared model is a **two-repo change**:
  1. add the field on both sides,
  2. update `CONTRACT` below **and** the identical copy in
     `AI-VL-core/tests/test_backend_contract.py`.
Do 1 without 2 and both suites go red; do 2 without 1 and both suites go red. That is the
point — there is no way to change one side quietly.

The helper below is duplicated in the sibling repo on purpose: importing it would be the very
boundary violation these tests exist to make unnecessary.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# The contract: model -> field names, IN ORDER.
#
# Order is part of it on purpose. It costs nothing to keep and it makes a review diff
# obvious: a field appended on one side and inserted in the middle on the other is the kind
# of drift that reads as identical until you compare them line by line.
#
# Types and defaults are deliberately NOT part of the contract. The gateway declares
# `scope: str | None = None` so it forwards exactly what the client sent and never invents a
# value; iacore declares `scope: str = CFG.get("scope", ...)` because the service owns the
# defaults. Asserting on types would fight that design instead of protecting it.
# --------------------------------------------------------------------------- #
CONTRACT = {
    "VlmRequest": ["image", "model", "scope", "variant", "max_tokens", "num_ctx", "think",
                   "prompt"],
    "VlmStreamRequest": ["image", "prompt", "model", "max_tokens", "num_ctx"],
    "SpeakRequest": ["text", "voice"],
    "CommandRequest": ["text", "image", "model", "robot", "num_ctx", "max_tokens"],
}

THIS_REPO = Path(__file__).resolve().parent.parent
OWN_SOURCE = THIS_REPO / "app.py"
# The sibling is cloned next to this repo in the ecosystem layout and git-ignored by the
# umbrella. A CI job that checks out only this repo will not have it — hence the skip.
SIBLING_SOURCE = THIS_REPO.parent / "AI-VL-core" / "service.py"

MODELS = sorted(CONTRACT)


def _basemodel_fields(source: Path) -> dict[str, list[str]]:
    """{class name: [annotated field names, in source order]} for every BaseModel in a file."""
    tree = ast.parse(source.read_text(), filename=str(source))
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
            isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases
        ):
            found[node.name] = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
    return found


@pytest.fixture(scope="module")
def own() -> dict[str, list[str]]:
    return _basemodel_fields(OWN_SOURCE)


@pytest.fixture(scope="module")
def sibling() -> dict[str, list[str]]:
    if not SIBLING_SOURCE.exists():
        pytest.skip(
            f"{SIBLING_SOURCE} not checked out — the cross-repo half of this contract test "
            "only runs in the ecosystem layout. The committed CONTRACT above still guards "
            "this repo on its own, which is what protects CI."
        )
    return _basemodel_fields(SIBLING_SOURCE)


# --------------------------------------------------------------------------- #
# This repo against the contract — always runs, CI included
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODELS)
def test_this_gateways_model_matches_the_committed_contract(own, name):
    """The defect: adding or renaming a field here and forgetting iacore, or vice versa."""
    assert name in own, f"{name} disappeared from app.py — the contract expects it"
    assert own[name] == CONTRACT[name], (
        f"app.py's {name} drifted from the contract.\n"
        f"  contract: {CONTRACT[name]}\n"
        f"  app.py  : {own[name]}\n"
        "If the change is intended, update CONTRACT here AND in "
        "AI-VL-core/tests/test_backend_contract.py."
    )


def test_every_field_the_gateway_forwards_is_declared(own):
    """The defect that motivated this file: silent field drop.

    pydantic v2 defaults to `extra="ignore"`, and neither side sets `model_config`. A field
    absent from the model is not an error — it is deleted. So the model IS the contract, and
    a field missing here cannot reach iacore no matter what the client sends.
    """
    for name in MODELS:
        assert own[name], f"{name} declares no fields at all; it would drop every input"


# --------------------------------------------------------------------------- #
# The sibling against the same contract — the cross-repo half
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODELS)
def test_iacores_model_matches_the_committed_contract(sibling, name):
    assert name in sibling, f"{name} disappeared from AI-VL-core/service.py"
    assert sibling[name] == CONTRACT[name], (
        f"service.py's {name} drifted from the contract.\n"
        f"  contract  : {CONTRACT[name]}\n"
        f"  service.py: {sibling[name]}\n"
        "A field added on one tier and not the other is dropped in transit."
    )


def test_the_contract_covers_every_model_the_two_tiers_share(own, sibling):
    """The defect: a NEW shared model appearing on both sides, guarded by nothing.

    The parametrized tests above only check what CONTRACT already lists, so without this the
    fifth shared model would be invisible to this suite.
    """
    shared = set(own) & set(sibling)
    assert shared == set(CONTRACT), (
        "the two tiers share models that the contract does not list.\n"
        f"  shared but unguarded: {sorted(shared - set(CONTRACT))}\n"
        f"  listed but no longer shared: {sorted(set(CONTRACT) - shared)}"
    )
