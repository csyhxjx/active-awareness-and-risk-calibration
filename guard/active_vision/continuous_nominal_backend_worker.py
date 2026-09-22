"""JSON-lines MuJoCo 3.13.0 worker; run only from the pinned backend env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def load_export(path: Path):
    manifest = json.loads((path / "export_manifest.json").read_text())
    if manifest.get("source_mujoco_version") != "2.3.7":
        raise RuntimeError("unexpected source engine in export")
    if manifest.get("xml_sha256") != __import__("hashlib").sha256((path / "scene.xml").read_bytes()).hexdigest():
        raise RuntimeError("export XML hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    # mjDSBL_NATIVECCD is a disable bit; clearing it explicitly makes the
    # provenance independent of the XML defaults.
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    data = mujoco.MjData(model)
    robot_ids = [model.geom(name).id for name in manifest["robot_geom_names"]]
    obstacle_ids = {
        route: [model.geom(name).id for name in names]
        for route, names in manifest["obstacle_geom_names"].items()
    }
    body_ids = {
        route: [model.body(name).id for name in names]
        for route, names in manifest["obstacle_body_names"].items()
    }
    return manifest, model, data, robot_ids, obstacle_ids, body_ids


def measure(model, data, robot_ids, obstacle_ids, body_ids, manifest, request):
    route = request["route"]
    if route not in obstacle_ids:
        raise ValueError(route)
    width = float(request["width_m"])
    offset = float(request["offset_m"])
    if not (0.0 < width < 0.2) or not np.isfinite(offset):
        raise ValueError("invalid obstacle parameters")
    # All route obstacle geoms are boxes with y half-size equal to width.
    for geom_id in obstacle_ids[route]:
        model.geom_size[geom_id, 1] = width
    lane = float(manifest["route_lanes"][route])
    for body_id in body_ids[route]:
        model.body_pos[body_id] = [0.070, lane + offset, 0.960]
    for other, ids in obstacle_ids.items():
        if other == route:
            continue
        for body_id in body_ids[other]:
            model.body_pos[body_id] = [2.0, 0.0, 0.960]
    qposes = np.asarray(request["qpos"], dtype=np.float64)
    intervals = int(request.get("intervals", 8))
    if qposes.ndim != 2 or qposes.shape[1] != model.nq or intervals < 1:
        raise ValueError("invalid qpos/interpolation request")
    minimum = None
    steps = []
    fromto = np.empty(6, dtype=np.float64)
    for step in range(len(qposes)):
        samples = (0,) if step == 0 else range(intervals + 1)
        step_min = None
        for sample in samples:
            fraction = 0.0 if step == 0 else sample / intervals
            pose = qposes[0] if step == 0 else (1.0 - fraction) * qposes[step - 1] + fraction * qposes[step]
            data.qpos[:] = pose
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            for robot_id in robot_ids:
                for obstacle_id in obstacle_ids[route]:
                    distance = float(mujoco.mj_geomDistance(model, data, robot_id, obstacle_id, 0.25, fromto))
                    if not np.isfinite(distance):
                        raise RuntimeError("non-finite mj_geomDistance")
                    candidate = {
                        "step": step, "sample": sample, "fraction": fraction,
                        "distance_m": distance,
                        "robot_geom": model.geom(robot_id).name,
                        "obstacle_geom": model.geom(obstacle_id).name,
                        "fromto": fromto.tolist(),
                    }
                    if step_min is None or distance < step_min["distance_m"]:
                        step_min = candidate
                    if minimum is None or distance < minimum["distance_m"]:
                        minimum = candidate
        steps.append(step_min)
    return {
        "schema_version": 2, "backend": "mujoco_native_ccd", "mujoco_version": mujoco.__version__,
        "native_ccd": not bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
        "route": route, "width_m": width, "offset_m": offset, "intervals": intervals,
        "minimum": minimum, "minimum_clearance_m": None if minimum is None else minimum["distance_m"],
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    manifest, model, data, robot_ids, obstacle_ids, body_ids = load_export(args.export)
    for line in __import__("sys").stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        print(json.dumps(measure(model, data, robot_ids, obstacle_ids, body_ids, manifest, request),
                         sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
