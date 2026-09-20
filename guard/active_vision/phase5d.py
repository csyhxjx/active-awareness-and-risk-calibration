"""Phase 5D OpenVLA chunk projection, tuple schema, and replay contracts."""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from guard.active_vision.runtime import QueryBroker, state_hash
from guard.active_vision.scene import ROUTES, route_waypoints
from guard.json_io import canonical_dumps


ACTION_SHAPE = (8, 7)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
STOP_RADIUS_M = 0.12
POSITION_SCALE = np.array([0.20, 0.20, 0.05], dtype=np.float64)
POSITION_OFFSET = np.array([0.0, 0.0, 1.00], dtype=np.float64)


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def chunk_bytes(chunk):
    array = np.asarray(chunk, dtype=np.float64)
    return canonical_dumps(array.tolist()).encode("utf-8")


def chunk_hash(chunk):
    return _sha256(chunk_bytes(chunk))


def validate_chunk(chunk):
    array = np.asarray(chunk, dtype=np.float64)
    if array.shape != ACTION_SHAPE:
        return False, "shape"
    if not np.isfinite(array).all():
        return False, "non_finite"
    if np.any(array < ACTION_LOW) or np.any(array > ACTION_HIGH):
        return False, "out_of_bounds"
    return True, None


def _distance_to_route(layout, position, route):
    endpoint = np.asarray(route_waypoints(layout, route)[1], dtype=np.float64)
    return float(np.linalg.norm(position - endpoint))


def project_chunk(layout, chunk):
    """Map one OpenVLA `(8, 7)` chunk to the registered fixed candidates."""
    valid, reason = validate_chunk(chunk)
    if not valid:
        return {
            "valid": False,
            "mapped_candidate": None,
            "mapping_distance": None,
            "failure": reason,
            "chunk_hash": chunk_hash(chunk),
        }
    array = np.asarray(chunk, dtype=np.float64)
    terminal = POSITION_OFFSET + POSITION_SCALE * array[-1, :3]
    distances = {route: _distance_to_route(layout, terminal, route) for route in ROUTES}
    nearest = min(ROUTES, key=lambda route: (distances[route], ROUTES.index(route)))
    if min(distances.values()) > STOP_RADIUS_M:
        candidate = "stop"
        distance = min(distances.values())
    else:
        candidate = nearest
        distance = distances[nearest]
    return {
        "valid": True,
        "mapped_candidate": candidate,
        "mapping_distance": float(distance),
        "terminal_position": terminal.tolist(),
        "distances": distances,
        "chunk_hash": chunk_hash(chunk),
    }


class Phase5DQueryBroker(QueryBroker):
    """The validated 5B broker with an explicit Phase 5D public-input boundary."""

    def __init__(self, env, budget=1, public_input=None):
        self.public_input = public_input or {}
        if any(key in self.public_input for key in ("hidden_state", "oracle", "collision", "hidden_geometry")):
            raise ValueError("Phase 5D proposer/selector input contains forbidden hidden data")
        super().__init__(env, budget=budget)


TUPLE_KEYS = (
    "proposer_output",
    "mapped_candidate",
    "selector_decision",
    "executed_action",
    "physical_outcome",
)


def validate_trial_tuple(record):
    if set(record) != set(TUPLE_KEYS):
        return False, "tuple_keys"
    if not isinstance(record["proposer_output"], dict) or not isinstance(record["mapped_candidate"], dict):
        return False, "tuple_types"
    if not isinstance(record["selector_decision"], dict) or not isinstance(record["executed_action"], dict):
        return False, "tuple_types"
    if not isinstance(record["physical_outcome"], dict):
        return False, "tuple_types"
    candidate = record["mapped_candidate"].get("mapped_candidate")
    if candidate not in (*ROUTES, "stop", None):
        return False, "candidate"
    return True, None


def derive_failures(record):
    valid, reason = validate_trial_tuple(record)
    if not valid:
        raise ValueError(reason)
    proposer = record["proposer_output"]
    mapped = record["mapped_candidate"]
    selector = record["selector_decision"]
    outcome = record["physical_outcome"]
    proposer_valid = bool(mapped.get("valid"))
    candidate = mapped.get("mapped_candidate")
    route_valid = candidate in ROUTES
    executed = record["executed_action"].get("route") in ROUTES
    failures = {
        "proposer_failure": not proposer_valid,
        "proposer_stop": proposer_valid and candidate == "stop",
        "candidate_recall_failure": proposer_valid and not route_valid and candidate != "stop",
        "selector_failure": proposer_valid and route_valid and not bool(selector.get("correct")),
        "collision": bool(outcome.get("collision")),
        "timeout": bool(outcome.get("timeout")),
        "unresolved": bool(selector.get("unresolved")),
        "executed": executed,
    }
    return failures


def replay_equal(first, second):
    """Require byte-identical canonical tuple/replay payloads."""
    return canonical_dumps(first) == canonical_dumps(second)
