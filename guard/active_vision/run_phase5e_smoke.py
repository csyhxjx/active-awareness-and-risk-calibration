"""Run the frozen Phase 5E route-proposer train-only smoke."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from guard.active_vision.phase5e import CHOICES, logits_to_proposal
from guard.active_vision.run_pilot import _execute_fresh, _fresh_env
from guard.active_vision.runtime import physical_state, state_hash
from guard.active_vision.scene import HIDDEN_STATES, Layout
from guard.active_vision.train_proposer_v4 import RouteProposer, feature
from guard.json_io import canonical_dumps, write_json


def load_ensemble(root, variant):
    models = []
    for path in sorted(root.glob(f"{variant}_seed*.pt")):
        payload = torch.load(path, map_location="cpu")
        model = RouteProposer(payload["input_size"]); model.load_state_dict(payload["state_dict"]); model.eval()
        models.append(model)
    if len(models) != 3:
        raise ValueError(f"expected three {variant} checkpoints")
    return models


def infer(models, vector):
    value = torch.tensor(vector).unsqueeze(0)
    with torch.no_grad():
        logits = torch.stack([model(value)[0] for model in models]).mean(0).numpy()
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path); parser.add_argument("checkpoints", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text()); args.output.mkdir(parents=True)
    ensembles = {variant: load_ensemble(args.checkpoints, variant) for variant in ("instruction_only", "v0_instruction", "v0_instruction_geometry")}
    trials = []
    for row in manifest["layouts"]:
        layout = Layout(**{key: value for key, value in row.items() if key in Layout.__dataclass_fields__})
        for hidden in HIDDEN_STATES:
            env = _fresh_env(layout, hidden, manifest["scene_seed"])
            try:
                image = env.capture_camera("v0"); branch_hash = state_hash(physical_state(env))
            finally: env.close()
            for route in ("left_route", "right_route"):
                instruction = f"prefer {route.replace('_route','')} route"
                outputs = {}
                for variant, models in ensembles.items():
                    logits = infer(models, feature(variant, instruction, image, layout))
                    outputs[variant] = {"logits": logits.tolist(), "proposal": logits_to_proposal(logits.tolist())}
                proposal = outputs["v0_instruction_geometry"]["proposal"]
                outcome = _execute_fresh(layout, hidden, proposal, {"seed": manifest["scene_seed"], "max_steps": 100, "hold_steps": 5}, branch_hash) if proposal in CHOICES[:2] else {"route": None, "collision": False, "collision_free_success": False}
                record = {"proposer_output": {"valid": True, "proposal": proposal, "instruction": instruction, "ablations": outputs,
                                                "v0_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest()},
                          "mapped_candidate": {"route": proposal},
                          "selector_decision": {"selected_candidate": proposal if proposal != "stop" else None, "not_trained": True},
                          "executed_action": {"route": None, "smoke_transfer_only": True},
                          "physical_outcome": outcome}
                trials.append({"layout_id": layout.layout_id, "hidden_state": hidden, "preference": route,
                               "record": record, "replay": json.loads(canonical_dumps(record))})
    write_json(args.output / "smoke.json", {"schema_version": 1, "protocol": "active_vision_v4", "trials": trials})


if __name__ == "__main__": main()
