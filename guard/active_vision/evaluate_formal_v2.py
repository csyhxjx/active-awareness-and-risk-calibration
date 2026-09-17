"""Evaluate the frozen Phase 5B policies on one sealed formal corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from guard.active_vision.check_formal_v2 import verifier
from guard.active_vision.run_formal_v2 import layout_from_row
from guard.active_vision.scene import ROUTES, camera_specs, route_waypoints
from guard.active_vision.train_selector_v2 import CAMERAS, Selector, geometry_features, prepare_geometry
from guard.json_io import write_json


RANDOM_SEEDS = (7001, 7002, 7003, 7004, 7005)
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_SAMPLES = 10_000


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def route_result(state, route):
    return next(row for row in state["routes"] if row["route"] == route)


def project_points(points, position, target, fovy):
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(points, dtype=np.float64) - position
    depth = relative @ forward
    scale = np.tan(np.deg2rad(float(fovy)) / 2.0)
    xy = np.column_stack(((relative @ right) / (depth * scale), (relative @ up) / (depth * scale)))
    return xy, depth


def geometric_coverage(layout_row, route, camera):
    layout = layout_from_row(layout_row)
    path = (np.asarray(layout.start),) + route_waypoints(layout, route)
    samples = []
    for start, end in zip(path[:-1], path[1:]):
        fractions = np.linspace(0.0, 1.0, 64, endpoint=False)[:, None]
        samples.append(np.asarray(start) + fractions * (np.asarray(end) - np.asarray(start)))
    samples.append(np.asarray(path[-1])[None, :])
    samples = np.concatenate(samples)
    position, target, fovy = camera_specs(layout)[camera]
    projected, depth = project_points(samples, position, target, fovy)

    half = np.array([0.025, 0.34, 0.20])
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    vertices = signs * half
    yaw = layout.occluder_yaw
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    vertices = vertices @ rotation.T + np.array([layout.occluder_x, 0.0, 1.02])
    box, box_depth = project_points(vertices, position, target, fovy)
    box = box[box_depth > 0]
    visible = (depth > 0) & (np.abs(projected[:, 0]) <= 1) & (np.abs(projected[:, 1]) <= 1)
    if len(box):
        lower = box.min(axis=0)
        upper = box.max(axis=0)
        occluded = np.all((projected >= lower) & (projected <= upper), axis=1)
        visible &= ~occluded
    return float(visible.mean())


class LearnedChooser:
    def __init__(self, model_paths, normalization, candidate_aware):
        self.models = []
        geometry_dim = len(normalization["mean"])
        for path in model_paths:
            model = Selector(geometry_dim)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            self.models.append(model)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.scale = np.asarray(normalization["scale"], dtype=np.float32)
        self.candidate_aware = candidate_aware

    def __call__(self, trial):
        image = np.asarray(Image.open(trial["v0_path"]), dtype=np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        scores = []
        with torch.no_grad():
            for camera in CAMERAS:
                geometry = geometry_features(trial["layout"], trial["preference"], camera)
                geometry = prepare_geometry(geometry, self.mean, self.scale, self.candidate_aware)
                tensor = torch.from_numpy(geometry).unsqueeze(0)
                scores.append(float(np.mean([model(image, tensor).item() for model in self.models])))
        return CAMERAS[int(np.argmax(scores))]


def build_trials(root, summary):
    trials = []
    for layout in summary["layouts"]:
        layout_id = layout["layout"]["layout_id"]
        for state in layout["states"]:
            for preference in ROUTES:
                trials.append(
                    {
                        "layout_id": layout_id,
                        "layout": layout["layout"],
                        "hidden_state": state["hidden_state"],
                        "state": state,
                        "preference": preference,
                        "v0_path": root / layout_id / state["hidden_state"] / "images" / "v0.png",
                        "image_root": root / layout_id / state["hidden_state"] / "images",
                    }
                )
    return trials


def evaluate(name, trials, camera_chooser=None, execute_nominal=False):
    rows = []
    for trial in trials:
        outcome = route_result(trial["state"], trial["preference"])
        successful = bool(outcome["collision_free_success"])
        double_blocked = trial["hidden_state"] == "11"
        camera = camera_chooser(trial) if camera_chooser else None
        verdict = None
        if camera:
            verdict = verifier(Image.open(trial["image_root"] / f"{camera}.png"), trial["preference"])["verdict"]
        execute = execute_nominal or verdict == "clear"
        unresolved = verdict == "unobserved"
        rows.append(
            {
                "layout_id": trial["layout_id"],
                "completed": execute and successful,
                "collision": execute and bool(outcome["collision"]),
                "executed": execute,
                "stopped": not execute,
                "false_rejection": not execute and successful,
                "correct_double_blocked_stop": not execute and double_blocked,
                "double_blocked": double_blocked,
                "unresolved": unresolved,
                "query_count": int(camera is not None),
                "decision_flip": not execute,
                "camera": camera,
            }
        )
    return summarize(name, rows)


def summarize(name, rows):
    count = len(rows)
    total = lambda key: sum(int(row[key]) for row in rows)
    executed = total("executed")
    successful = total("completed") + sum(
        int(row["false_rejection"]) for row in rows
    )
    double_blocked = total("double_blocked")
    correct_double_blocked = total("correct_double_blocked_stop")
    metrics = {
        "collision_free_completion": total("completed") / count,
        "collision_rate": total("collision") / count,
        "execution_coverage": executed / count,
        "collision_given_execution": total("collision") / executed if executed else 0.0,
        "erroneous_rejection_rate": total("false_rejection") / count,
        "erroneous_rejection_given_successful_candidate": (
            total("false_rejection") / successful if successful else 0.0
        ),
        "stop_rate": total("stopped") / count,
        "correct_double_blocked_stop_rate": correct_double_blocked / double_blocked,
        "unresolved_rate": total("unresolved") / count,
        "mean_query_count": sum(row["query_count"] for row in rows) / count,
        "mean_observation_latency_ms": 0.0,
        "decision_flip_rate": total("decision_flip") / count,
        "raw_counts": {
            "trials": count,
            "completed": total("completed"),
            "collisions": total("collision"),
            "executed": executed,
            "false_rejections": total("false_rejection"),
            "successful_candidates": successful,
            "double_blocked_trials": double_blocked,
            "correct_double_blocked_stops": correct_double_blocked,
        },
    }
    layout_metrics = {}
    for layout_id in sorted({row["layout_id"] for row in rows}):
        group = [row for row in rows if row["layout_id"] == layout_id]
        layout_metrics[layout_id] = {
            "collision_free_completion": sum(row["completed"] for row in group) / len(group),
            "collision_rate": sum(row["collision"] for row in group) / len(group),
        }
    return {"policy": name, "trials": count, "metrics": metrics, "layout_metrics": layout_metrics, "rows": rows}


def bootstrap_difference(left, right, metric, indices):
    layout_ids = sorted(left["layout_metrics"])
    differences = np.asarray(
        [left["layout_metrics"][key][metric] - right["layout_metrics"][key][metric] for key in layout_ids]
    )
    samples = differences[indices].mean(axis=1)
    return {
        "estimate": float(differences.mean()),
        "two_sided_95_ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "one_sided_95_upper": float(np.quantile(samples, 0.95)),
        "two_sided_p": float(min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))),
    }


def holm_adjust(p_values):
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    summary = json.loads((args.corpus_root / "summary.json").read_text())
    if summary["split"] != "test" or len(summary["layouts"]) != 40:
        raise ValueError("evaluation requires the sealed 40-layout test corpus")
    normalization = json.loads(Path(freeze["normalization"]).read_text())
    aware = LearnedChooser([Path(path) for path in freeze["aware_checkpoints"]], normalization, True)
    agnostic = LearnedChooser([Path(path) for path in freeze["agnostic_checkpoints"]], normalization, False)
    trials = build_trials(args.corpus_root, summary)
    policies = {}
    policies["B0"] = evaluate("B0 always stop", trials)
    policies["B1"] = evaluate("B1 nominal execute", trials, execute_nominal=True)
    policies["B2"] = evaluate("B2 conservative stop", trials)
    policies["B3"] = evaluate("B3 fixed view", trials, lambda trial: freeze["best_fixed_camera"])
    random_runs = {}
    for seed in RANDOM_SEEDS:
        generator = np.random.Generator(np.random.PCG64(seed))
        choices = generator.integers(0, len(CAMERAS), size=len(trials))
        mapping = {id(trial): CAMERAS[int(choice)] for trial, choice in zip(trials, choices)}
        random_runs[str(seed)] = evaluate(f"B4 random {seed}", trials, lambda trial, m=mapping: m[id(trial)])
    first_random = random_runs[str(RANDOM_SEEDS[0])]
    policies["B4"] = {
        "policy": "B4 random mean over five registered seeds",
        "trials": first_random["trials"],
        "metrics": dict(first_random["metrics"]),
        "layout_metrics": json.loads(json.dumps(first_random["layout_metrics"])),
        "rows": [],
    }
    policies["B5"] = evaluate("B5 candidate agnostic", trials, agnostic)
    policies["B6"] = evaluate(
        "B6 geometric",
        trials,
        lambda trial: max(CAMERAS, key=lambda camera: (geometric_coverage(trial["layout"], trial["preference"], camera), -CAMERAS.index(camera))),
    )
    policies["M"] = evaluate("M candidate aware", trials, aware)

    # B4's registered result is the mean of five complete seeded policies.
    for key in policies["B4"]["metrics"]:
        if key == "raw_counts":
            policies["B4"]["metrics"][key] = {
                count: float(np.mean([row["metrics"][key][count] for row in random_runs.values()]))
                for count in policies["B4"]["metrics"][key]
            }
        else:
            policies["B4"]["metrics"][key] = float(
                np.mean([row["metrics"][key] for row in random_runs.values()])
            )
    for layout_id in policies["B4"]["layout_metrics"]:
        for metric in policies["B4"]["layout_metrics"][layout_id]:
            policies["B4"]["layout_metrics"][layout_id][metric] = float(
                np.mean([row["layout_metrics"][layout_id][metric] for row in random_runs.values()])
            )

    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(0, 40, size=(BOOTSTRAP_SAMPLES, 40))
    comparisons = {}
    for baseline in ("B0", "B1", "B2", "B3", "B4", "B5", "B6"):
        comparisons[f"M_vs_{baseline}"] = {
            "completion": bootstrap_difference(policies["M"], policies[baseline], "collision_free_completion", indices),
            "collision": bootstrap_difference(policies["M"], policies[baseline], "collision_rate", indices),
        }
    family = {name: comparisons[name]["completion"]["two_sided_p"] for name in comparisons if name != "M_vs_B6"}
    holm = holm_adjust(family)
    for name, adjusted in holm.items():
        comparisons[name]["completion"]["holm_adjusted_p"] = adjusted
    primary = comparisons["M_vs_B6"]
    method_contribution = (
        primary["completion"]["two_sided_95_ci"][0] > 0
        and primary["collision"]["one_sided_95_upper"] <= 0.025
    )
    result = {
        "schema_version": 2,
        "split": "test",
        "corpus_summary_sha256": sha256_file(args.corpus_root / "summary.json"),
        "freeze_sha256": sha256_file(args.freeze),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES, "unit": "layout"},
        "observation_latency_semantics": "cached offline retrieval is defined as zero; query count is the observation-cost endpoint",
        "policies": policies,
        "random_runs": random_runs,
        "comparisons": comparisons,
        "method_contribution": method_contribution,
        "interpretation": "learned_method_contribution" if method_contribution else "mechanism_validation_geometric_rule_sufficient",
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
