"""Audit the frozen route proposer, then run the frozen Phase 5E policy ladder."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.phase5e import logits_to_proposal
from guard.active_vision.run_formal_v2 import layout_from_row
from guard.active_vision.train_proposer_v4 import RouteProposer, feature
from guard.active_vision.evaluate_formal_v2 import bootstrap_difference
from guard.json_io import write_json


def load_models(paths):
    import torch
    models = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        model = RouteProposer(payload["input_size"]); model.load_state_dict(payload["state_dict"]); model.eval(); models.append(model)
    return models


def audit_proposer(corpus_root, summary, model_paths):
    import torch
    models = load_models(model_paths); rows = []
    for layout_row in summary["layouts"]:
        layout = layout_from_row(layout_row["layout"]); layout_id = layout.layout_id
        for state in layout_row["states"]:
            image = np.asarray(Image.open(corpus_root / layout_id / state["hidden_state"] / "images" / "v0.png"))
            for preference in ("left_route", "right_route"):
                instruction = f"prefer {preference.replace('_route', '')} route"
                value = torch.tensor(feature("v0_instruction_geometry", instruction, image, layout)).unsqueeze(0)
                with torch.no_grad(): logits = torch.stack([model(value)[0] for model in models]).mean(0).numpy()
                proposal = logits_to_proposal(logits.tolist())
                rows.append({"layout_id": layout_id, "hidden_state": state["hidden_state"], "preference": preference,
                             "logits": logits.tolist(), "proposal": proposal})
    routes = sum(row["proposal"] in ("left_route", "right_route") for row in rows)
    correct = sum(row["proposal"] == row["preference"] for row in rows)
    return {"trials": len(rows), "valid_routes": routes, "coverage": routes / len(rows), "preference_accuracy": correct / len(rows),
            "stops": sum(row["proposal"] == "stop" for row in rows), "failures": sum(row["proposal"] not in ("left_route", "right_route", "stop") for row in rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("corpus_root", type=Path); parser.add_argument("freeze", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args(); freeze = json.loads(args.freeze.read_text()); summary = json.loads((args.corpus_root / "summary.json").read_text())
    audit = audit_proposer(args.corpus_root, summary, [Path(path) for path in freeze["proposer_checkpoints"]])
    if audit["valid_routes"] != audit["trials"] or audit["preference_accuracy"] != 1.0:
        write_json(args.output, {"schema_version": 4, "split": "test", "proposer_audit": audit, "policy_evaluation": None,
                                 "interpretation": "proposer_failure_prevents_registered_comparison"})
        raise SystemExit(1)
    temporary = args.output.with_suffix(".v2.json")
    subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_formal_v2.py")), str(args.corpus_root), str(args.freeze), str(temporary)], check=True)
    result = json.loads(temporary.read_text()); temporary.unlink()
    result["schema_version"] = 4; result["proposer_audit"] = audit
    generator = np.random.Generator(np.random.PCG64(20260917)); indices = generator.integers(0, 40, size=(10_000, 40))
    mechanism = {"completion": bootstrap_difference(result["policies"]["B6"], result["policies"]["B1"], "collision_free_completion", indices),
                 "collision": bootstrap_difference(result["policies"]["B6"], result["policies"]["B1"], "collision_rate", indices)}
    result["comparisons"]["B6_vs_B1_mechanism"] = mechanism
    result["mechanism_gain"] = mechanism["completion"]["two_sided_95_ci"][0] > 0 and mechanism["collision"]["one_sided_95_upper"] <= 0
    if result["method_contribution"]: result["interpretation"] = "learned_method_contribution"
    elif result["mechanism_gain"]: result["interpretation"] = "active_vision_mechanism_geometric_sufficient"
    else: result["interpretation"] = "no_observed_active_vision_gain"
    write_json(args.output, result)


if __name__ == "__main__": main()
