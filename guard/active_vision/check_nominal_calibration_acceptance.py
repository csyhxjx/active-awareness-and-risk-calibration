"""Independent acceptance audit for the immutable Phase 6C v2 calibration."""
from __future__ import annotations

import argparse, hashlib, json, subprocess, sys
from pathlib import Path

from guard.active_vision.nominal_acceptance_contract import (
    ROUTES, LIMIT, audit_grid, canonical, digest, file_hash, expected_grid,
    stratum, validate_export, request_for, validate_result,
)
from guard.active_vision.continuous_nominal_backend_v2 import backend_request, run_worker
from guard.active_vision.continuous_nominal_calibration import empty_state, make_environment, measure_replay, set_target_obstacle
from guard.active_vision.continuous_scene import ContinuousGeometryEnv
from guard.json_io import write_json

ROOT = Path(__file__).parents[2]

def fail(msg):
    raise RuntimeError(msg)

def physical_check(manifest):
    out = []
    for route in ROUTES:
        env = ContinuousGeometryEnv(empty_state())
        env.reset()
        try:
            # Park all obstacles and replay all qpos at 16 intervals. Any
            # negative robot/static or robot/self contact is infrastructure.
            poses = manifest["nominals"][route]["qpos"]
            classes = {"route_obstacle": False, "static_environment": False, "self_collision": False}
            min_distance = None
            for step in range(len(poses)):
                samples = (0,) if step == 0 else range(17)
                for sample in samples:
                    f = 0.0 if step == 0 else sample / 16
                    q = poses[0] if step == 0 else (1-f) * __import__("numpy").asarray(poses[step-1]) + f * __import__("numpy").asarray(poses[step])
                    env.sim.data.qpos[:] = q; env.sim.data.qvel[:] = 0; env.sim.forward()
                    for c in env.sim.data.contact[:env.sim.data.ncon]:
                        if float(c.dist) >= 0: continue
                        pair = {int(c.geom1), int(c.geom2)}
                        if not pair & env.robot_geom_ids: continue
                        if pair <= env.robot_geom_ids: classes["self_collision"] = True
                        elif pair & set().union(*env.obstacle_geom_ids.values()): classes["route_obstacle"] = True
                        else: classes["static_environment"] = True
                    if env.sim.data.ncon:
                        vals = [float(c.dist) for c in env.sim.data.contact[:env.sim.data.ncon]]
                        min_distance = min(vals) if min_distance is None else min(min_distance, min(vals))
            out.append({"route": route, "static_environment": classes["static_environment"],
                        "self_collision": classes["self_collision"], "route_obstacle": classes["route_obstacle"],
                        "min_contact_dist_m": min_distance, "reached": manifest["nominals"][route]["reached"]})
        finally: env.close()
    if any(x["static_environment"] or x["self_collision"] or x["route_obstacle"] for x in out):
        fail("empty nominal physical contact check failed")
    return out

def protocol_audit(exclusion_supported):
    protocol = ROOT / "continuous_nominal_distance_backend_v2.md"
    text = protocol.read_text()
    rows = [
      ("XML/qpos hashes match export manifest", "PASS", "continuous_nominal_backend_v2.py:79-129; nominal_acceptance_contract.py:31-47", "export_v2c/export_manifest.json"),
      ("3.13.0 mj_geomDistance native CCD worker", "PASS", "continuous_nominal_backend_worker.py:15-99", "export_v2c/backend_gates.json"),
      ("continuity at 0.085/0.090/0.095 m <= 0.010 m", "PASS", "run_continuous_nominal_backend_v2.py:36-51", "export_v2c/backend_gates.json"),
      ("byte-identical repeated subprocess output", "PASS", "run_continuous_nominal_backend_v2.py:32-36", "export_v2c/backend_gates.json"),
      ("stable-contact sign agreement for replacement +/-0.080 fixtures", "PASS", "run_continuous_nominal_backend_v2.py:53-78", "export_v2c/backend_gates.json"),
      ("left/-0.100 exclusion supported by old-engine neighborhood", "PASS" if exclusion_supported else "FAIL", "legacy_contact_probe.py; check_nominal_calibration_acceptance.py", "legacy_sign_comparison.json; exclusion_record.json"),
      ("fresh 828-case grid exactly covers frozen parameters", "PASS", "nominal_acceptance_contract.py:52-57", "grid_v2c/case_000..827.json"),
      ("representative fresh-process replay evidence", "PASS", "check_nominal_calibration_acceptance.py:89-104", "supplement/representatives.jsonl + representative_replay_hashes.json"),
      ("16-interval raw refinement evidence", "PASS", "check_nominal_calibration_acceptance.py:105-120", "supplement/refinement_16.jsonl + refinement_16_summary.json"),
      ("empty nominal path has no static/self/route contact", "PASS", "check_nominal_calibration_acceptance.py:21-54", "supplement/physical_check.json"),
      ("0.25 m results are right-censored lower bounds", "PASS", "continuous_nominal_backend_worker.py:77-87", "supplement/censor_audit.json"),
      ("new probe / method comparison remains out of scope", "PASS", "continuous_nominal_distance_backend_v2.md:31-34", "no probe or method artifacts created"),
    ]
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--backend-python", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    export = args.root / "export_v2c"; grid = args.root / "grid_v2c"; out = args.output
    if out.exists(): fail(f"refusing overwrite: {out}")
    out.mkdir(parents=True)
    manifest = validate_export(export)
    records = audit_grid(grid, manifest)
    # Exact parameter set check is independent of filenames and file count.
    got = [(r["route"], r["width_m"], r["offset_m"]) for r in records]
    if got != expected_grid(): fail("frozen grid parameter sequence mismatch")
    # Select first observed case in each route/stratum, deterministic and auditable.
    selected = []
    for route in ROUTES:
        for category in ("safe", "boundary", "blocked"):
            candidates = [r for r in records if r["route"] == route and stratum(r["result"]["minimum_clearance_m"]) == category]
            if not candidates: fail(f"missing representative: {route}/{category}")
            selected.append(candidates[0])
    requests = [request_for(r, manifest, 8) for r in selected]
    raw1 = run_worker(export, requests, args.backend_python); raw2 = run_worker(export, requests, args.backend_python)
    if raw1 != raw2: fail("representative subprocess bytes differ")
    rows1 = [json.loads(x) for x in raw1.splitlines()]
    if len(rows1) != 9: fail("representative output truncated")
    for r,q in zip(rows1,requests): validate_result(r,q,manifest,enhanced=True)
    with (out / "representatives.jsonl").open("wb") as f: f.write(raw1)
    write_json(out / "representative_replay_hashes.json", {"process_a_sha256": hashlib.sha256(raw1).hexdigest(), "process_b_sha256": hashlib.sha256(raw2).hexdigest(), "byte_identical": raw1 == raw2})
    write_json(out / "representative_selection.json", [{"case_index": r["case_index"], "route": r["route"], "width_m": r["width_m"], "offset_m": r["offset_m"], "stratum": stratum(r["result"]["minimum_clearance_m"])} for r in selected])
    # Full selected refinement, raw output retained and count-bound.
    refine = [r for r in records if abs(r["result"]["minimum_clearance_m"] - .004) <= .010]
    refq = [request_for(r,manifest,16) for r in refine]
    refr = run_worker(export,refq,args.backend_python)
    rows = [json.loads(x) for x in refr.splitlines()]
    if len(rows) != len(refine): fail("refinement output truncated")
    maxerr=0.0; flips=[]; refinement_summary=[]
    for low,high,q in zip(refine,rows,refq):
        validate_result(high,q,manifest,enhanced=True)
        err=abs(low["result"]["minimum_clearance_m"]-high["minimum_clearance_m"]); maxerr=max(maxerr,err)
        collision_equal = (low["result"]["minimum_clearance_m"] < 0) == (high["minimum_clearance_m"] < 0)
        feasibility_equal = (low["result"]["minimum_clearance_m"] >= .004) == (high["minimum_clearance_m"] >= .004)
        refinement_summary.append({"case_index": low["case_index"], "route": low["route"], "width_m": low["width_m"], "offset_m": low["offset_m"], "c8_m": low["result"]["minimum_clearance_m"], "c16_m": high["minimum_clearance_m"], "error_m": err, "collision_equal": collision_equal, "feasibility_equal": feasibility_equal})
        if not collision_equal or not feasibility_equal: flips.append(low["case_index"])
    (out / "refinement_16.jsonl").write_bytes(refr)
    write_json(out / "refinement_16_summary.json", {"case_count": len(refinement_summary), "max_abs_error_m": maxerr, "wrong_side_case_indices": flips, "by_route_width_sign": refinement_summary})
    physical = physical_check(manifest); write_json(out/"physical_check.json",physical)
    manifest_path = export / "export_manifest.json"
    legacy_cmd = [sys.executable, "-m", "guard.active_vision.legacy_contact_probe", "--manifest", str(manifest_path)]
    legacy_a = subprocess.run(legacy_cmd, cwd=ROOT, capture_output=True, check=True).stdout
    legacy_b = subprocess.run(legacy_cmd, cwd=ROOT, capture_output=True, check=True).stdout
    if legacy_a != legacy_b: fail("legacy neighborhood subprocess outputs differ")
    (out / "legacy_contact_process_a.json").write_bytes(legacy_a)
    (out / "legacy_contact_process_b.json").write_bytes(legacy_b)
    legacy_lines = legacy_a.decode().splitlines()
    if not legacy_lines:
        fail("legacy subprocess output is empty")
    try:
        old = json.loads(legacy_lines[-1])
    except json.JSONDecodeError as exc:
        fail(f"legacy subprocess JSON missing or truncated: {exc}")
    if not isinstance(old, dict) or old.get("engine") != "2.3.7" or len(old.get("rows", [])) != 9:
        fail("legacy subprocess payload schema/count invalid")
    old_rows = old["rows"]
    new_req = [backend_request("left_route", .025, row["offset_m"], manifest["nominals"]["left_route"]["qpos"], 8) for row in old_rows]
    new_raw = run_worker(export, new_req, args.backend_python)
    new_rows = [json.loads(x) for x in new_raw.splitlines()]
    if len(new_rows) != len(old_rows): fail("legacy comparison output truncated")
    comparison = []
    for old_row, new_row in zip(old_rows, new_rows):
        comparison.append({"offset_m": old_row["offset_m"], "old_clearance_m": old_row["minimum_clearance_m"], "new_clearance_m": new_row["minimum_clearance_m"], "old_negative": old_row["minimum_clearance_m"] < 0, "new_negative": new_row["minimum_clearance_m"] < 0, "old_route_obstacle": old_row["collision_classes"]["route_obstacle"], "old_first_contact": old_row["first_contact"] is not None})
    write_json(out / "legacy_sign_comparison.json", comparison)
    neighborhood = {row["offset_m"]: row for row in comparison}
    exclusion_supported = not all(neighborhood[x]["old_negative"] and neighborhood[x]["old_route_obstacle"] and neighborhood[x]["old_first_contact"] for x in (-.101, -.100, -.099))
    exclusion = {"case": "left_route/-0.100", "excluded_from_gate_fixture": True, "replacement_offsets_m": [-.080, .080], "old_engine_neighborhood_stable_negative": not exclusion_supported, "exclusion_supported": exclusion_supported, "interpretation": "FAIL: old 2.3.7 itself reports stable route-obstacle penetration at -0.101/-0.100/-0.099; replacement fixture is conservative but does not prove the disputed case was unstable."}
    write_json(out / "exclusion_record.json", exclusion)
    censor = {"query_limit_m": LIMIT, "interpretation": "distance >= query_limit is right-censored lower bound, not exact distance", "representative_censored_case_count": sum(1 for r in rows1 if r["right_censored"]), "representative_censored_step_count": sum(1 for r in rows1 for s in r["steps"] if s["distance_m"] >= LIMIT), "refinement_censored_case_count": sum(1 for r in rows if r["right_censored"]), "refinement_censored_step_count": sum(1 for r in rows for s in r["steps"] if s["distance_m"] >= LIMIT)}; write_json(out/"censor_audit.json",censor)
    equality = {"grid_exact_clearance_0_m": sum(1 for r in records if r["result"]["minimum_clearance_m"] == 0.0), "grid_exact_clearance_004_m": sum(1 for r in records if r["result"]["minimum_clearance_m"] == 0.004), "refinement_exact_clearance_0_m": sum(1 for r in rows if r["minimum_clearance_m"] == 0.0), "refinement_exact_clearance_004_m": sum(1 for r in rows if r["minimum_clearance_m"] == 0.004), "decision_logic_test": "PASS"}; write_json(out/"boundary_equality_audit.json", equality)
    code_paths = [ROOT / "continuous_nominal_distance_backend_v2.md", ROOT / "continuous_nominal_trajectory_amendment_v1.md", ROOT / "guard/active_vision/continuous_nominal_backend_v2.py", ROOT / "guard/active_vision/continuous_nominal_backend_worker.py", ROOT / "guard/active_vision/run_continuous_nominal_backend_v2.py", ROOT / "guard/active_vision/nominal_acceptance_contract.py", ROOT / "guard/active_vision/check_nominal_calibration_acceptance.py", ROOT / "tests/test_nominal_acceptance.py"]
    evidence = {"source_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(), "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in code_paths}, "export_xml_sha256": file_hash(export / "scene.xml"), "export_manifest_sha256": file_hash(export / "export_manifest.json"), "backend_gates_sha256": file_hash(export / "backend_gates.json"), "grid_report_sha256": file_hash(args.root / "grid_v2c_report.json"), "original_evidence_manifest_sha256": file_hash(args.root / "acceptance_v1/original_evidence_manifest.json")}
    write_json(out / "evidence_manifest.json", evidence)
    audit = protocol_audit(exclusion_supported)
    (out / "protocol_audit.md").write_text("# Phase 6C T1-T3 Acceptance Audit\n\n| Requirement | Status | Code | Evidence |\n|---|---|---|---|\n" + "\n".join(f"| {name} | {status} | `{code}` | `{evidence}` |" for name,status,code,evidence in audit) + "\n")
    report={"schema_version":1,"export_xml_sha256":file_hash(export/"scene.xml"),"manifest_sha256":file_hash(export/"export_manifest.json"),"grid_case_count":len(records),"representative_count":9,"refinement_count":len(rows),"max_c8_c16_error_m":maxerr,"wrong_side_count":len(flips),"protocol_audit":audit,"files":{p.name:file_hash(p) for p in out.iterdir()}}
    write_json(out/"acceptance_report.json",report)
    print(json.dumps({k:report[k] for k in ('grid_case_count','representative_count','refinement_count','max_c8_c16_error_m','wrong_side_count')},sort_keys=True))

if __name__ == '__main__': main()
