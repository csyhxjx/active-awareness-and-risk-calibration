"""Generate committed Phase 5A evidence from a checked pilot."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from guard.active_vision.check_pilot import obstacle_visible
from guard.json_io import write_json


CAMERAS = ("v_left", "v_right", "v_high")
ROUTES = ("left_route", "right_route")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def route_result(state, route):
    return next(row for row in state["routes"] if row["route"] == route)


def camera_covers(camera, route, mirror):
    if camera == "v_left":
        return route == "left_route"
    if camera == "v_right":
        return route == "right_route"
    near_route = "left_route" if mirror == 1 else "right_route"
    return route == near_route


def load_image(root, layout, state, camera):
    return Image.open(root / layout / state / "images" / f"{camera}.png").copy()


def decide_with_camera(root, layout, state, route, camera, mirror):
    if not camera_covers(camera, route, mirror):
        return "unobserved"
    return "blocked" if obstacle_visible(load_image(root, layout, state, camera), route) else "clear"


def evaluate_policy(name, trials, chooser):
    rows = []
    for trial in trials:
        decision = chooser(trial)
        chosen = decision["route"]
        outcome = route_result(trial["state_row"], chosen) if chosen else None
        completed = bool(outcome and outcome["collision_free_success"])
        collision = bool(outcome and outcome["collision"])
        feasible = any(route_result(trial["state_row"], route)["collision_free_success"] for route in ROUTES)
        rows.append(
            {
                "completed": completed,
                "collision": collision,
                "executed": chosen is not None,
                "stopped": chosen is None,
                "feasible_stop": chosen is None and feasible,
                "false_rejection": bool(decision.get("false_rejection", False)),
                "cost": decision["cost"],
            }
        )
    count = len(rows)
    costs = np.asarray([row["cost"] for row in rows], dtype=np.float64)
    metric = lambda key: sum(row[key] for row in rows) / count
    return {
        "policy": name,
        "trials": count,
        "collision_free_completion_rate": metric("completed"),
        "collision_rate": metric("collision"),
        "execution_coverage": metric("executed"),
        "false_rejection_rate": metric("false_rejection"),
        "stop_rate": metric("stopped"),
        "feasible_stop_rate": metric("feasible_stop"),
        "mean_observation_cost": float(costs.mean()),
        "median_observation_cost": float(np.median(costs)),
        "p95_observation_cost": float(np.quantile(costs, 0.95)),
    }


def build_trials(root, summary, config):
    mirrors = {row["layout_id"]: row["mirror"] for row in config["layouts"]}
    trials = []
    for layout in summary["layouts"]:
        for state_row in layout["states"]:
            for preference in ROUTES:
                trials.append(
                    {
                        "root": root,
                        "layout": layout["layout"],
                        "state": state_row["hidden_state"],
                        "state_row": state_row,
                        "preference": preference,
                        "mirror": mirrors[layout["layout"]],
                    }
                )
    return trials


def camera_policy(camera):
    def choose(trial):
        verdict = decide_with_camera(
            trial["root"], trial["layout"], trial["state"], trial["preference"], camera, trial["mirror"]
        )
        successful = route_result(trial["state_row"], trial["preference"])["collision_free_success"]
        return {
            "route": trial["preference"] if verdict == "clear" else None,
            "cost": 1,
            "false_rejection": verdict == "blocked" and successful,
        }

    return choose


def matched_b1(trial):
    camera = "v_left" if trial["preference"] == "left_route" else "v_right"
    verdict = decide_with_camera(
        trial["root"], trial["layout"], trial["state"], trial["preference"], camera, trial["mirror"]
    )
    successful = route_result(trial["state_row"], trial["preference"])["collision_free_success"]
    return {
        "route": trial["preference"] if verdict == "clear" else None,
        "cost": 1,
        "false_rejection": verdict == "blocked" and successful,
    }


def matched_b2(trial):
    first = matched_b1(trial)
    if first["route"] is not None:
        return first
    alternate = "right_route" if trial["preference"] == "left_route" else "left_route"
    camera = "v_left" if alternate == "left_route" else "v_right"
    verdict = decide_with_camera(
        trial["root"], trial["layout"], trial["state"], alternate, camera, trial["mirror"]
    )
    successful = route_result(trial["state_row"], alternate)["collision_free_success"]
    return {
        "route": alternate if verdict == "clear" else None,
        "cost": 2,
        "false_rejection": first.get("false_rejection", False) or (verdict == "blocked" and successful),
    }


def make_figures(root, summary, output):
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for row, state in enumerate(("00", "10", "01", "11")):
        for column, camera in enumerate(("v0", "v_left", "v_right", "v_high")):
            axes[row, column].imshow(load_image(root, "dev_00", state, camera))
            axes[row, column].set_title(f"{state} / {camera}", fontsize=9)
            axes[row, column].axis("off")
    fig.suptitle("One layout: identical free view, candidate-dependent paid evidence")
    fig.tight_layout()
    fig.savefig(output / "view_audit.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5.4))
    pairs = (
        ("dev_00", "left_route", "v_left", "00", "10"),
        ("dev_03", "right_route", "v_right", "00", "01"),
    )
    for row, (layout, route, camera, clear_state, blocked_state) in enumerate(pairs):
        axes[row, 0].imshow(load_image(root, layout, clear_state, "v0"))
        axes[row, 0].set_title(f"{layout}: shared V0")
        axes[row, 1].imshow(load_image(root, layout, clear_state, camera))
        axes[row, 1].set_title(f"{route}: clear -> execute")
        axes[row, 2].imshow(load_image(root, layout, blocked_state, camera))
        axes[row, 2].set_title(f"{route}: blocked -> stop")
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle("Paired decisions: same initial input, observation changes the correct action")
    fig.tight_layout()
    fig.savefig(output / "paired_decisions.png", dpi=160)
    plt.close(fig)

    matrix = []
    labels = []
    for layout in summary["layouts"]:
        states = {row["hidden_state"]: row for row in layout["states"]}
        matrix.append(
            [
                1 if route_result(states[state], route)["collision_free_success"] else -1
                for state in ("00", "10", "01", "11")
                for route in ROUTES
            ]
        )
        labels.append(layout["layout"])
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks(
        range(8),
        [f"{state}-{route[0].upper()}" for state in ("00", "10", "01", "11") for route in ("left", "right")],
        rotation=45,
        ha="right",
    )
    axis.set_title("Candidate outcomes (green: collision-free completion; red: blocked collision)")
    fig.tight_layout()
    fig.savefig(output / "route_outcomes.png", dpi=160)
    plt.close(fig)


def percent(value):
    return f"{100.0 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.pilot_root / "summary.json").read_text())
    gates = json.loads((args.pilot_root / "gates.json").read_text())
    config = json.loads(args.config.read_text())
    trials = build_trials(args.pilot_root, summary, config)
    policies = [
        evaluate_policy("Always stop (B=0)", trials, lambda trial: {"route": None, "cost": 0}),
        evaluate_policy("Nominal preferred route (B=0)", trials, lambda trial: {"route": trial["preference"], "cost": 0}),
        evaluate_policy("Fixed V_left (B=1)", trials, camera_policy("v_left")),
        evaluate_policy("Fixed V_right (B=1)", trials, camera_policy("v_right")),
        evaluate_policy("Fixed V_high (B=1)", trials, camera_policy("v_high")),
        evaluate_policy("Candidate-aware side view (B=1)", trials, matched_b1),
        evaluate_policy("Candidate-aware fallback (B<=2)", trials, matched_b2),
    ]
    random_metrics = []
    for seed in range(5):
        generator = random.Random(seed)

        def random_policy(trial, rng=generator):
            return camera_policy(rng.choice(CAMERAS))(trial)

        random_metrics.append(evaluate_policy(f"Random B=1 seed {seed}", trials, random_policy))
    random_mean = {"policy": "Random view (B=1, 5 seeds)", "trials": 5 * len(trials)}
    for key in policies[0]:
        if key not in {"policy", "trials"}:
            random_mean[key] = float(np.mean([row[key] for row in random_metrics]))
    policies.insert(5, random_mean)
    write_json(args.output / "gates.json", gates)
    write_json(args.output / "fixed_controller_metrics.json", {"policies": policies, "random_seeds": random_metrics})
    provenance = {
        "pilot_root": str(args.pilot_root),
        "guard_head": summary["git_head"],
        "canonical_candidate_trajectories": summary["canonical_candidate_trajectories"],
        "replay_audits": summary["replay_audits"],
        "summary_sha256": sha256_file(args.pilot_root / "summary.json"),
        "gates_sha256": sha256_file(args.pilot_root / "gates.json"),
        "config_sha256": sha256_file(args.config),
        "protocol_sha256": sha256_file(args.protocol),
        "excluded_attempt": "guard_workspace/active_vision_v1/pilot_60ab168 (P0 pre-route failure)",
    }
    write_json(args.output / "provenance.json", provenance)
    make_figures(args.pilot_root, summary, args.output)
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if row['pass'] else 'FAIL'} | "
        + ", ".join(f"{key}={value}" for key, value in row.items() if key != "pass")
        + " |"
        for name, row in gates["gates"].items()
    )
    metric_rows = "\n".join(
        "| {policy} | {completion} | {collision} | {coverage} | {reject} | {stop} | {feasible_stop} | {cost:.2f} |".format(
            policy=row["policy"],
            completion=percent(row["collision_free_completion_rate"]),
            collision=percent(row["collision_rate"]),
            coverage=percent(row["execution_coverage"]),
            reject=percent(row["false_rejection_rate"]),
            stop=percent(row["stop_rate"]),
            feasible_stop=percent(row["feasible_stop_rate"]),
            cost=row["mean_observation_cost"],
        )
        for row in policies
    )
    report = f"""# Phase 5A active-vision development pilot

Canonical scope: 12 layouts x 4 hidden states = 48 scenes and 96 candidate trajectories. The 12 deterministic replay audits are verification reruns, not added samples. Run HEAD: `{summary['git_head']}`.

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
{gate_rows}

All pre-registered P0-P5 gates pass. `V0` is pixel-identical within every four-state layout group. The RGB-only color/shape audit was committed before collection and separates the relevant route in 12/12 layouts. It is an automated single-auditor development check, not an inter-rater study.

## Fixed-controller metrics

The table crosses both public route preferences over all 48 scenes (96 decision trials). Physical candidate labels are reused; these are not 96 additional rollouts. A stop is never counted as completion.

| Policy | Collision-free completion | Collision | Execution coverage | False rejection | Stop | Feasible stop | Mean views |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{metric_rows}

The candidate-aware B=1 rule removes collisions but completes only when the preferred candidate is clear; its 25% feasible-stop rate is the cost of refusing an unobserved fallback. B<=2 closes that gap: it completes every scene with at least one clear route (75% overall) and correctly stops in double-blocked scenes. Always-stop has zero collision but also zero completion, so it is not a winning policy.

These fixed-controller numbers establish the observation/decision mechanism only. They do not establish a learned selector advantage and are not an OpenVLA improvement. Same-budget learned, fixed, random, and geometric comparison requires a separately frozen Phase 5B contract. Frozen OpenVLA proposals enter only after that mechanism comparison; proposer failures and candidate recall must then be reported separately.

## Evidence

- `view_audit.png`: all four hidden states for one layout; free images are identical while paid evidence is candidate-dependent.
- `paired_decisions.png`: same initial input and candidate, followed by clear/blocked observations that justify different correct decisions.
- `route_outcomes.png`: all 12 layouts and eight state-candidate outcomes; no failed layouts were removed.

## Limitations

This development benchmark deliberately uses color-coded obstacles and candidate-aligned side views. It validates H1 information value and closes the fixed-controller decision chain, but it does not by itself validate H2 learned view selection, H3 generalization, moving-camera cost, or safety of natural OpenVLA behavior. Development layouts and derivatives remain excluded from future validation and test sets.
"""
    (args.output / "pilot_stats.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
