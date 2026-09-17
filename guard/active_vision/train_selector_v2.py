"""Train the frozen Phase 5B learned selectors using train and validation only."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from guard.active_vision.check_formal_v2 import verifier
from guard.active_vision.run_formal_v2 import layout_from_row
from guard.active_vision.scene import ROUTES, _look_at_quat, camera_specs, route_waypoints
from guard.json_io import write_json


CAMERAS = ("v_left", "v_right", "v_high")
SEEDS = (1701, 1702, 1703)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def route_result(rows, route):
    return next(row for row in rows if row["route"] == route)


def geometry_features(layout_row, route, camera):
    layout = layout_from_row(layout_row)
    points = np.stack(route_waypoints(layout, route)).astype(np.float32)
    position, target, fovy = camera_specs(layout)[camera]
    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    direction = target - position
    direction /= np.linalg.norm(direction)
    quaternion = _look_at_quat(position, target).astype(np.float32)
    distances = np.linalg.norm(points - position, axis=1)
    route_onehot = [1.0, 0.0] if route == "left_route" else [0.0, 1.0]
    projection = [float(distances.mean()), float(distances.min()), float(distances.max()), float(np.dot(points.mean(0) - position, direction))]
    occluder = [layout.occluder_x, 0.0, layout.occluder_yaw, 0.025, 0.34, 0.20]
    return np.asarray(
        [*points.reshape(-1), *route_onehot, *route_onehot, *projection, *position, *quaternion, float(fovy), *occluder],
        dtype=np.float32,
    )


def build_examples(root, summary):
    examples = []
    for layout in summary["layouts"]:
        row = layout["layout"]
        states = {item["hidden_state"]: item for item in layout["states"]}
        v0_path = root / row["layout_id"] / "00" / "images" / "v0.png"
        for route in ROUTES:
            for camera in CAMERAS:
                outcomes = []
                for hidden_state, state in states.items():
                    image = Image.open(root / row["layout_id"] / hidden_state / "images" / f"{camera}.png")
                    verdict = verifier(image, route)["verdict"]
                    result = route_result(state["routes"], route)
                    truth = "clear" if result["collision_free_success"] else "blocked"
                    outcomes.append(1.0 if verdict == truth else (-1.0 if verdict != "unobserved" else 0.0))
                gain = float(np.mean(outcomes))
                examples.append(
                    {
                        "layout_id": row["layout_id"],
                        "route": route,
                        "camera": camera,
                        "v0_path": str(v0_path),
                        "geometry": geometry_features(row, route, camera),
                        "utility": gain - 0.10,
                    }
                )
    return examples


class SelectorDataset(Dataset):
    def __init__(self, examples, mean, scale, candidate_aware=True):
        self.examples = examples
        self.mean = mean
        self.scale = scale
        self.candidate_aware = candidate_aware

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        image = np.asarray(Image.open(row["v0_path"]), dtype=np.float32) / 255.0
        geometry = row["geometry"].copy()
        if not self.candidate_aware:
            geometry[:17] = 0.0
        geometry = (geometry - self.mean) / self.scale
        return (
            torch.from_numpy(image.transpose(2, 0, 1)),
            torch.from_numpy(geometry),
            torch.tensor(row["utility"], dtype=torch.float32),
            row["layout_id"],
            row["route"],
        )


class Selector(nn.Module):
    def __init__(self, geometry_dim):
        super().__init__()
        layers = []
        channels = (3, 16, 32, 64)
        for source, target in zip(channels[:-1], channels[1:]):
            layers.extend((nn.Conv2d(source, target, 3, stride=2, padding=1), nn.ReLU()))
        self.image = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, image, geometry):
        return self.head(torch.cat((self.image(image), self.geometry(geometry)), dim=1)).squeeze(1)


def loss_value(prediction, target, layout_ids, routes):
    mse = nn.functional.mse_loss(prediction, target)
    ranking = []
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            if layout_ids[left] != layout_ids[right] or routes[left] != routes[right]:
                continue
            difference = target[left] - target[right]
            if torch.abs(difference) < 1e-12:
                continue
            sign = torch.sign(difference)
            ranking.append(torch.relu(0.10 - sign * (prediction[left] - prediction[right])))
    hinge = torch.stack(ranking).mean() if ranking else prediction.new_tensor(0.0)
    return mse + 0.5 * hinge, mse, hinge


def train_one(train_examples, validation_examples, mean, scale, seed, candidate_aware, output, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = Selector(len(mean)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        SelectorDataset(train_examples, mean, scale, candidate_aware),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        SelectorDataset(validation_examples, mean, scale, candidate_aware), batch_size=64, shuffle=False
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, 201):
        model.train()
        train_losses = []
        for images, geometry, target, layout_ids, routes in train_loader:
            images = images.to(device)
            geometry = geometry.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            prediction = model(images, geometry)
            loss, mse, hinge = loss_value(prediction, target, layout_ids, routes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        squared = []
        with torch.no_grad():
            for images, geometry, target, _, _ in validation_loader:
                images = images.to(device)
                geometry = geometry.to(device)
                target = target.to(device)
                prediction = model(images, geometry)
                squared.extend(((prediction - target) ** 2).tolist())
        validation_mse = float(np.mean(squared))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_mse": validation_mse})
        if validation_mse < best_loss:
            best_loss = validation_mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    torch.save(best_state, output)
    return {"seed": seed, "best_epoch": best_epoch, "validation_mse": best_loss, "epochs_run": len(history), "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_root", type=Path)
    parser.add_argument("validation_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite training output: {args.output_root}")
    args.output_root.mkdir(parents=True)
    train_summary = json.loads((args.train_root / "summary.json").read_text())
    validation_summary = json.loads((args.validation_root / "summary.json").read_text())
    if train_summary["split"] != "train" or validation_summary["split"] != "validation":
        raise ValueError("training requires train and validation summaries")
    train_examples = build_examples(args.train_root, train_summary)
    validation_examples = build_examples(args.validation_root, validation_summary)
    geometry = np.stack([row["geometry"] for row in train_examples])
    mean = geometry.mean(axis=0).astype(np.float32)
    scale = geometry.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    write_json(args.output_root / "normalization.json", {"mean": mean, "scale": scale})
    runs = []
    for candidate_aware, name in ((True, "candidate_aware"), (False, "candidate_agnostic")):
        for seed in SEEDS:
            checkpoint = args.output_root / f"{name}_seed{seed}.pt"
            result = train_one(
                train_examples,
                validation_examples,
                mean,
                scale,
                seed,
                candidate_aware,
                checkpoint,
                device,
            )
            result.update({"model": name, "checkpoint": checkpoint.name, "sha256": sha256_file(checkpoint)})
            runs.append(result)
    # B3 uses the same validation verifier/decision contract and ties left, right, high.
    camera_scores = {}
    for camera in CAMERAS:
        rows = [row for row in validation_examples if row["camera"] == camera]
        camera_scores[camera] = float(np.mean([row["utility"] for row in rows]))
    best_fixed = max(CAMERAS, key=lambda camera: (camera_scores[camera], -CAMERAS.index(camera)))
    write_json(
        args.output_root / "training_summary.json",
        {
            "schema_version": 2,
            "train_layouts": len(train_summary["layouts"]),
            "validation_layouts": len(validation_summary["layouts"]),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "execution_device": str(device),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "best_fixed_camera": best_fixed,
            "fixed_camera_validation_utility": camera_scores,
            "runs": runs,
        },
    )


if __name__ == "__main__":
    main()
