"""Static, preflight, and formal hard-gate checker for Phase 6A7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from guard.active_vision.belief_branching import ROUTES, STATES
from guard.active_vision.phase6a7 import (
    ALL_STATES,
    CHECKER_VERSION,
    CLEARANCE_THRESHOLD_M,
    FIXED_SEQUENCES,
    PAID_CAMERAS,
    PROTOCOL_ID,
    ROI,
    SEED,
    allowed_roi_geom,
    sha256_bytes,
    trajectory_sha256,
)
from guard.json_io import write_json


def check_manifest(manifest: dict, ledger: dict | None, protocol_sha256: str) -> dict:
    errors = []
    layouts = manifest.get("layouts", [])
    ids = [row.get("layout_id") for row in layouts]
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("protocol_sha256") != protocol_sha256:
        errors.append("protocol_provenance")
    if manifest.get("seed") != SEED:
        errors.append("seed")
    if len(layouts) != 12 or len(ids) != len(set(ids)):
        errors.append("layout_scope")
    if manifest.get("scene_count") != 96 or manifest.get("route_count") != 288:
        errors.append("declared_scope")
    if manifest.get("color_semantics") != "global_fixed_phase6a6":
        errors.append("color_semantics")
    if manifest.get("roi") != list(ROI):
        errors.append("roi")
    sequences = {tuple(row) for row in manifest.get("fixed_sequences", [])}
    if sequences != set(FIXED_SEQUENCES) or len(sequences) != 17:
        errors.append("fixed_sequences")
    for layout in layouts:
        if layout.get("color_permutation") != [0, 1, 2]:
            errors.append(f"{layout.get('layout_id')}:color")
        if set(layout.get("states", [])) != set(ALL_STATES):
            errors.append(f"{layout.get('layout_id')}:states")
        table = layout.get("observation_table", {})
        if set(table) != set(ALL_STATES) or any(set(table[state]) != set(PAID_CAMERAS) for state in table):
            errors.append(f"{layout.get('layout_id')}:observation_table")
    if ledger is None:
        errors.append("preflight_ledger_missing")
    else:
        if ledger.get("policy_result_filtering") is not False:
            errors.append("policy_filtering")
        admitted = {row["candidate_index"] for row in layouts}
        passed = {row["candidate_index"] for row in ledger.get("entries", []) if row.get("accepted")}
        if admitted != passed:
            errors.append("admission_ledger")
    return {
        "schema_version": 1,
        "checker_version": CHECKER_VERSION,
        "stage": "manifest",
        "layouts": len(layouts),
        "scenes": len(layouts) * len(ALL_STATES),
        "routes": len(layouts) * len(ALL_STATES) * len(ROUTES),
        "all_pass": not errors,
        "errors": errors,
    }


def check_candidate_plan(plan: dict, protocol_sha256: str) -> dict:
    errors = []
    candidates = plan.get("candidates", [])
    ids = [row.get("layout_id") for row in candidates]
    indices = [row.get("candidate_index") for row in candidates]
    if plan.get("protocol_id") != PROTOCOL_ID or plan.get("protocol_sha256") != protocol_sha256:
        errors.append("protocol_provenance")
    if plan.get("seed") != SEED or plan.get("requested_layouts") != 12:
        errors.append("scope_seed")
    if len(candidates) < 12 or len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
        errors.append("candidate_uniqueness")
    if plan.get("policy_result_filtering") is not False:
        errors.append("policy_filtering")
    if plan.get("color_semantics") != "global_fixed_phase6a6" or plan.get("roi") != list(ROI):
        errors.append("observation_contract")
    if len({tuple(row) for row in plan.get("fixed_sequences", [])}) != 17:
        errors.append("fixed_sequences")
    table = plan.get("observation_table", {})
    if set(table) != set(ALL_STATES) or any(set(table[state]) != set(PAID_CAMERAS) for state in table):
        errors.append("observation_table")
    if any(row.get("color_permutation") != [0, 1, 2] for row in candidates):
        errors.append("candidate_color")
    return {
        "schema_version": 1,
        "checker_version": CHECKER_VERSION,
        "stage": "candidate",
        "candidate_count": len(candidates),
        "requested_layouts": plan.get("requested_layouts"),
        "all_pass": not errors,
        "errors": errors,
    }


def check_preflight(manifest: dict, results_dir: Path) -> dict:
    errors = []
    checked_routes = 0
    for layout in manifest["layouts"]:
        path = results_dir / f"candidate_{layout['candidate_index']:03d}.json"
        if not path.exists() or sha256_bytes(path.read_bytes()) != layout["preflight_result_sha256"]:
            errors.append(f"{layout['layout_id']}:preflight_hash")
            continue
        result = json.loads(path.read_text())
        if not result.get("accepted") or result.get("errors"):
            errors.append(f"{layout['layout_id']}:rejected")
        for state_row in result.get("states", []):
            for route in state_row.get("routes", []):
                checked_routes += 1
                if route.get("errors"):
                    errors.append(f"{layout['layout_id']}:{state_row['state']}:{route['route']}")
    if checked_routes != 288:
        errors.append("route_scope")
    return {
        "schema_version": 1,
        "checker_version": CHECKER_VERSION,
        "stage": "preflight",
        "routes": checked_routes,
        "all_pass": not errors,
        "errors": errors,
    }


def check_formal(
    summary: dict,
    manifest: dict,
    manifest_sha256: str,
    protocol_sha256: str,
    preflight_results: Path,
) -> dict:
    errors = []
    layout_results = []
    refs = summary.get("layouts", [])
    if len(refs) != 12 or len({row["layout_id"] for row in refs}) != 12:
        errors.append("layout_scope")
    expected_ids = {row["layout_id"] for row in manifest["layouts"]}
    if {row["layout_id"] for row in refs} != expected_ids:
        errors.append("layout_membership")
    for ref in refs:
        row = json.loads(Path(ref["summary"]).read_text())
        summary_dir = Path(ref["summary"]).parent
        layout_errors = []
        if row.get("manifest_sha256") != manifest_sha256 or row.get("protocol_sha256") != protocol_sha256:
            layout_errors.append("provenance_hash")
        if row.get("checker_version") != CHECKER_VERSION or row.get("seed") != SEED:
            layout_errors.append("provenance_version")
        scenes = {scene["state"]: scene for scene in row.get("scenes", [])}
        if set(scenes) != set(ALL_STATES):
            layout_errors.append("scene_scope")
        heads = {scene.get("git_head") for scene in scenes.values()}
        if len(heads) != 1 or not next(iter(heads), ""):
            layout_errors.append("git_head")
        v0_hashes = {
            next((item.get("image_sha256") for item in scene["fingerprint"]["ledger"] if item["camera"] == "v0"), None)
            for scene in scenes.values()
        }
        if len(v0_hashes) != 1 or None in v0_hashes:
            layout_errors.append("hidden_public_isolation")
        preflight_path = preflight_results / f"candidate_{row['spec']['candidate_index']:03d}.json"
        if not preflight_path.exists() or sha256_bytes(preflight_path.read_bytes()) != row["spec"]["preflight_result_sha256"]:
            layout_errors.append("preflight_provenance")
            preflight_routes = {}
        else:
            preflight = json.loads(preflight_path.read_text())
            preflight_routes = {
                (state_row["state"], route["route"]): route
                for state_row in preflight["states"]
                for route in state_row["routes"]
            }
        route_count = 0
        for state, scene in scenes.items():
            required = {
                "protocol_id", "git_head", "manifest_sha256", "protocol_sha256",
                "checker_version", "seed", "shard_id", "layout_id", "state",
                "fingerprint", "fresh_process_fingerprint_sha256",
                "fresh_process_replay_exact", "routes",
            }
            if not required <= set(scene):
                layout_errors.append(f"{state}:provenance_fields")
            if not scene.get("fresh_process_replay_exact"):
                layout_errors.append(f"{state}:replay")
            ledger = {item["camera"]: item for item in scene["fingerprint"]["ledger"]}
            if set(ledger) != {"v0", *PAID_CAMERAS}:
                layout_errors.append(f"{state}:camera_scope")
            for camera in PAID_CAMERAS:
                item = ledger.get(camera, {})
                if item.get("detector_output") != item.get("expected_outcome"):
                    layout_errors.append(f"{state}/{camera}:roi_semantics")
                if item.get("roi_geom_names") != [allowed_roi_geom(camera)]:
                    layout_errors.append(f"{state}/{camera}:roi_isolation")
                for key in ("image_sha256", "roi_sha256", "roi_bounds", "state_hash"):
                    if not item.get(key):
                        layout_errors.append(f"{state}/{camera}:{key}")
            state_hashes = {item.get("state_hash") for item in ledger.values()}
            if len(state_hashes) != 1:
                layout_errors.append(f"{state}:query_state")
            for route in scene.get("routes", []):
                route_count += 1
                index = ROUTES.index(route["route"])
                clear = state[index] == "0"
                if clear and not route.get("collision_free_success"):
                    layout_errors.append(f"{state}/{route['route']}:clear")
                if clear and route.get("minimum_clearance_m") is not None and route["minimum_clearance_m"] < CLEARANCE_THRESHOLD_M:
                    layout_errors.append(f"{state}/{route['route']}:clearance")
                if clear and any(route.get("collision_classes", {}).values()):
                    layout_errors.append(f"{state}/{route['route']}:collision_class")
                if not clear and not route.get("collision_classes", {}).get("route_obstacle"):
                    layout_errors.append(f"{state}/{route['route']}:blocked")
                if route.get("collision_classes", {}).get("static_environment"):
                    layout_errors.append(f"{state}/{route['route']}:static_collision")
                if route.get("collision_classes", {}).get("self_collision"):
                    layout_errors.append(f"{state}/{route['route']}:self_collision")
                for key in ("trajectory_sha256", "trajectory_length_m", "nearest_active_obstacle", "collision_classes"):
                    if key not in route:
                        layout_errors.append(f"{state}/{route['route']}:{key}")
                trajectory_path = summary_dir / state / route.get("trajectory_file", "")
                if (
                    not trajectory_path.is_file()
                    or trajectory_sha256(json.loads(trajectory_path.read_text())) != route.get("trajectory_sha256")
                ):
                    layout_errors.append(f"{state}/{route['route']}:trajectory_file")
                registered = preflight_routes.get((state, route["route"]))
                if registered is None or any(
                    route.get(key) != registered.get(key)
                    for key in (
                        "trajectory_sha256", "collision", "collision_free_success",
                        "collision_classes", "minimum_clearance_m",
                    )
                ):
                    layout_errors.append(f"{state}/{route['route']}:preflight_parity")
        if route_count != 24:
            layout_errors.append("route_scope")
        sequences = {tuple(item["sequence"]) for item in row.get("fixed", [])}
        if sequences != set(FIXED_SEQUENCES) or len(sequences) != 17:
            layout_errors.append("fixed_sequences")
        adaptive = row.get("adaptive", [])
        if {item["state"] for item in adaptive} != set(STATES):
            layout_errors.append("adaptive_scope")
        best = max(row.get("fixed", []), key=lambda item: (item["mean_utility"], item["completion"]))
        adaptive_metrics = _adaptive_metrics(adaptive)
        result = {
            "layout_id": row["layout_id"],
            "hard_pass": not layout_errors,
            "errors": layout_errors,
            "adaptive": adaptive_metrics,
            "best_fixed": {
                "sequence": best["sequence"],
                "completion": best["completion"],
                "collision": best["collision"],
                "stop": best["stop"],
                "mean_queries": best["mean_queries"],
                "mean_utility": best["mean_utility"],
            },
        }
        layout_results.append(result)
        errors.extend(f"{row['layout_id']}:{error}" for error in layout_errors)
    hard_pass = not errors
    categories = {
        "roi_contract": not any("roi_" in error for error in errors),
        "hidden_public_isolation": not any("hidden_public_isolation" in error for error in errors),
        "route_physics": not any(
            any(token in error for token in (":clear", ":blocked", ":static_collision", ":self_collision", ":preflight_parity"))
            for error in errors
        ),
        "clearance": not any(error.endswith(":clearance") for error in errors),
        "fresh_process_replay": not any(error.endswith(":replay") for error in errors),
        "fixed_sequences": not any(error.endswith(":fixed_sequences") for error in errors),
        "provenance": not any(
            any(token in error for token in ("provenance", "git_head", "trajectory_file"))
            for error in errors
        ),
    }
    return {
        "schema_version": 1,
        "checker_version": CHECKER_VERSION,
        "stage": "formal",
        "hard_gates": categories,
        "all_hard_gates_pass": hard_pass,
        "scientific_results_authorized": hard_pass,
        "layouts": layout_results,
        "errors": errors,
    }


def _adaptive_metrics(rows: list[dict]) -> dict:
    count = len(rows)
    completion = sum(item["physical_outcome"]["success"] for item in rows)
    collision = sum(item["physical_outcome"]["collision"] for item in rows)
    stop = sum(item["terminal_action"] == "stop" for item in rows)
    queries = [sum(step["decision"]["kind"] == "query" for step in item["trace"]) for item in rows]
    utilities = [
        (1.0 if item["physical_outcome"]["success"] else -4.0 if item["physical_outcome"]["collision"] else -0.25)
        - 0.05 * query
        for item, query in zip(rows, queries)
    ]
    return {
        "completion": completion,
        "collision": collision,
        "stop": stop,
        "mean_queries": sum(queries) / count,
        "mean_utility": sum(utilities) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("candidate", "manifest", "preflight", "formal"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--preflight-results", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    protocol_sha = sha256_bytes(args.protocol.read_bytes())
    manifest = json.loads(manifest_bytes)
    if args.stage == "candidate":
        result = check_candidate_plan(manifest, protocol_sha)
    elif args.stage == "manifest":
        ledger = json.loads(args.ledger.read_text()) if args.ledger else None
        result = check_manifest(manifest, ledger, protocol_sha)
    elif args.stage == "preflight":
        if args.preflight_results is None:
            parser.error("preflight stage requires --preflight-results")
        result = check_preflight(manifest, args.preflight_results)
    else:
        if args.summary is None or args.preflight_results is None:
            parser.error("formal stage requires --summary and --preflight-results")
        result = check_formal(
            json.loads(args.summary.read_text()), manifest, sha256_bytes(manifest_bytes), protocol_sha,
            args.preflight_results,
        )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("all_pass", result.get("all_hard_gates_pass", False)) else 1)


if __name__ == "__main__":
    main()
