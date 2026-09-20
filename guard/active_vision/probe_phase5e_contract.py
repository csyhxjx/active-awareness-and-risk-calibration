"""Run the pre-training Phase 5E contract probe on a non-data fixture."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from guard.active_vision.phase5e import decision_bytes, decision_hash, replay_tuple, validate_five_tuple
from guard.active_vision.run_pilot import _fresh_env
from guard.active_vision.runtime import QueryBroker, physical_state
from guard.active_vision.scene import HIDDEN_STATES, Layout, route_waypoints
from guard.json_io import canonical_dumps, write_json


LAYOUT = Layout("phase5e_contract_fixture", (-0.103, 0.0, 1.01), (0.21, 0.0, 1.01), 0.175, 0.061, 0.295, route_ribbons=True)
SEED = 55000


def _equal_state(first, second):
    numeric = all(np.array_equal(first[key], second[key]) for key in ("qpos", "qvel", "act", "ctrl"))
    return numeric and first["time"] == second["time"] and canonical_dumps(first["python_rng"]) == canonical_dumps(second["python_rng"]) and canonical_dumps(first["numpy_rng"]) == canonical_dumps(second["numpy_rng"])


def public_metadata():
    return {route: [point.tolist() for point in route_waypoints(LAYOUT, route)] for route in ("left_route", "right_route")}


def run():
    rows = []
    for hidden in HIDDEN_STATES:
        env = _fresh_env(LAYOUT, hidden, SEED)
        try:
            broker = QueryBroker(env, budget=0)
            before = physical_state(env)
            v0 = broker.free_observation()
            after = physical_state(env)
            v0_hash = hashlib.sha256(np.asarray(v0).tobytes()).hexdigest()
            inputs = []
            for instruction in ("prefer left route", "prefer right route"):
                value = {"v0_rgb_sha256": v0_hash, "instruction": instruction, "public_route_metadata": public_metadata()}
                inputs.append({"instruction": instruction, "decision_hash": decision_hash(value), "input": value})
            rows.append({"hidden_state": hidden, "state_unchanged": _equal_state(before, after), "v0_sha256": v0_hash,
                         "paid_queries": sum(item["charged"] for item in broker.ledger), "decisions": inputs})
        finally:
            env.close()
    hidden_independent = all(len({row["decisions"][i]["decision_hash"] for row in rows}) == 1 for i in (0, 1))
    example = rows[0]["decisions"][0]["input"]
    code = "import json,sys; from guard.active_vision.phase5e import decision_bytes; print(decision_bytes(json.loads(sys.argv[1])).hex())"
    fresh = subprocess.check_output([sys.executable, "-c", code, json.dumps(example)], text=True).strip()
    local = decision_bytes(example).hex()
    tuple_record = {"proposer_output": {"proposal": "left_route", "valid": True}, "mapped_candidate": {"route": "left_route"},
                    "selector_decision": {"selected_candidate": "left_route"}, "executed_action": {"route": "left_route"},
                    "physical_outcome": {"collision": False, "timeout": False}}
    reconstructed = json.loads(canonical_dumps(tuple_record))
    tuple_ok = validate_five_tuple(reconstructed)[0] and replay_tuple(tuple_record, reconstructed)
    return {"schema_version": 1, "fixture_only": True, "gpu_training_started": False, "rows": rows,
            "all_v0_identical": len({row["v0_sha256"] for row in rows}) == 1,
            "hidden_state_independent": hidden_independent,
            "all_queries_state_preserving": all(row["state_unchanged"] for row in rows),
            "no_paid_queries": all(row["paid_queries"] == 0 for row in rows),
            "fresh_process_bytes_equal": fresh == local,
            "tuple_roundtrip_exact": tuple_ok,
            "all_pass": len({row["v0_sha256"] for row in rows}) == 1 and hidden_independent and
                        all(row["state_unchanged"] for row in rows) and all(row["paid_queries"] == 0 for row in rows) and
                        fresh == local and tuple_ok}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run()
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
