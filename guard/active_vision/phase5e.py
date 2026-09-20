"""Pure route-level proposer contracts for Phase 5E."""

import math
import hashlib

from guard.active_vision.phase5d import replay_equal
from guard.json_io import canonical_dumps

ROUTES = ("left_route", "right_route")
CHOICES = ROUTES + ("stop",)
PUBLIC_INPUT_KEYS = {"v0_rgb_sha256", "instruction", "public_route_metadata"}


def logits_to_proposal(logits):
    values = list(logits)
    if len(values) != 3 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("logits must be three finite numbers")
    # Fixed order makes ties deterministic: left, right, then stop.
    index = max(range(3), key=lambda i: (values[i], -i))
    return CHOICES[index]


def selector_decision(proposal, verifier_verdict, budget):
    if proposal not in CHOICES:
        raise ValueError("invalid proposal")
    if budget == 0:
        return {"proposal": proposal, "selected_candidate": proposal if proposal != "stop" else None,
                "stop_reason": "proposer_stop" if proposal == "stop" else None}
    if budget == 1:
        if proposal == "stop":
            return {"proposal": proposal, "selected_candidate": None, "stop_reason": "proposer_stop"}
        if verifier_verdict == "clear":
            return {"proposal": proposal, "selected_candidate": proposal, "stop_reason": None}
        return {"proposal": proposal, "selected_candidate": None,
                "stop_reason": "blocked" if verifier_verdict == "blocked" else "unobserved"}
    raise ValueError("only B=0 and B=1 are registered")


def validate_proposer_input(value):
    if set(value) != PUBLIC_INPUT_KEYS:
        raise ValueError("proposer input schema violation")
    forbidden = {"hidden_state", "collision", "oracle", "margin", "paid_view", "v_left", "v_right", "v_high"}
    serialized = canonical_dumps(value)
    if any(token in serialized for token in forbidden):
        raise ValueError("forbidden proposer input")
    return True


def contract_fixture_logits(value):
    """Non-trainable contract fixture; never an experimental proposer."""
    validate_proposer_input(value)
    instruction = value["instruction"]
    if instruction == "prefer left route":
        return [2.0, 0.0, -1.0]
    if instruction == "prefer right route":
        return [0.0, 2.0, -1.0]
    return [-1.0, -1.0, 2.0]


def decision_bytes(proposer_input, verifier_verdict="unobserved", budget=0):
    logits = contract_fixture_logits(proposer_input)
    proposal = logits_to_proposal(logits)
    decision = selector_decision(proposal, verifier_verdict, budget)
    return canonical_dumps({"logits": logits, "proposal": proposal, "decision": decision}).encode()


def decision_hash(proposer_input, verifier_verdict="unobserved", budget=0):
    return hashlib.sha256(decision_bytes(proposer_input, verifier_verdict, budget)).hexdigest()


def validate_five_tuple(record):
    keys = {"proposer_output", "mapped_candidate", "selector_decision", "executed_action", "physical_outcome"}
    if set(record) != keys:
        return False, "tuple_keys"
    if record["proposer_output"].get("proposal") not in CHOICES:
        return False, "proposal"
    return True, None


def derive_failures(record):
    valid, reason = validate_five_tuple(record)
    if not valid:
        raise ValueError(reason)
    proposer = record["proposer_output"]
    proposal = proposer["proposal"]
    selected = record["selector_decision"].get("selected_candidate")
    executed = record["executed_action"].get("route")
    outcome = record["physical_outcome"]
    return {
        "proposer_failure": not bool(proposer.get("valid", True)),
        "proposer_stop": proposal == "stop",
        "candidate_recall_failure": proposal not in CHOICES,
        "selector_failure": proposal in ROUTES and selected not in (proposal, None),
        "executed": executed in ROUTES,
        "collision": bool(outcome.get("collision", False)),
        "timeout": bool(outcome.get("timeout", False)),
    }


def replay_tuple(record, replay):
    return replay_equal(record, replay)
