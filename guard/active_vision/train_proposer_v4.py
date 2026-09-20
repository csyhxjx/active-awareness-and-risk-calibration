"""Train the three frozen Phase 5E route-proposer ablations."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from guard.active_vision.run_pilot import _fresh_env
from guard.active_vision.scene import Layout, route_waypoints
from guard.json_io import write_json

INSTRUCTIONS = (("prefer left route", 0), ("prefer right route", 1), ("remain stopped", 2))
VARIANTS = ("instruction_only", "v0_instruction", "v0_instruction_geometry")
SEEDS = (5801, 5802, 5803)


class RouteProposer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, value):
        return self.network(value)


def instruction_feature(text):
    values = np.zeros(3, dtype=np.float32)
    values[[item[0] for item in INSTRUCTIONS].index(text)] = 1
    return values


def image_feature(image):
    value = np.asarray(image, dtype=np.float32) / 255.0
    return np.concatenate((value.mean((0, 1)), value.std((0, 1))))


def geometry_feature(layout):
    return np.concatenate([np.concatenate(route_waypoints(layout, route)) for route in ("left_route", "right_route")]).astype(np.float32)


def feature(variant, instruction, image, layout):
    parts = [instruction_feature(instruction)]
    if variant != "instruction_only":
        parts.append(image_feature(image))
    if variant == "v0_instruction_geometry":
        parts.append(geometry_feature(layout))
    return np.concatenate(parts).astype(np.float32)


def collect(manifest):
    examples = []
    for row in manifest["layouts"]:
        layout = Layout(**{key: value for key, value in row.items() if key in Layout.__dataclass_fields__})
        env = _fresh_env(layout, "00", manifest["scene_seed"])
        try:
            image = env.capture_camera("v0")
        finally:
            env.close()
        for instruction, target in INSTRUCTIONS:
            examples.append((layout, image, instruction, target))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    examples = collect(manifest)
    args.output.mkdir(parents=True)
    records = []
    for variant in VARIANTS:
        x = torch.tensor(np.stack([feature(variant, i, image, layout) for layout, image, i, _ in examples]))
        y = torch.tensor([target for _, _, _, target in examples], dtype=torch.long)
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = RouteProposer(x.shape[1]).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
            x_gpu, y_gpu = x.cuda(), y.cuda()
            for _ in range(300):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x_gpu), y_gpu)
                loss.backward(); optimizer.step()
            path = args.output / f"{variant}_seed{seed}.pt"
            torch.save({"variant": variant, "seed": seed, "input_size": x.shape[1], "state_dict": model.cpu().state_dict()}, path)
            with torch.no_grad():
                accuracy = float((model(x).argmax(1) == y).float().mean())
            records.append({"variant": variant, "seed": seed, "loss": float(loss.cpu()), "train_accuracy": accuracy,
                            "checkpoint": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(args.output / "training.json", {"schema_version": 1, "nominal_hidden_state": "00", "examples": len(examples), "records": records})


if __name__ == "__main__":
    main()
