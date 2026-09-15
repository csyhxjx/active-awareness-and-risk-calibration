"""Export one LIBERO task's MuJoCo scene names and free joints."""

import contextlib
import io
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUARD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUARD_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "openvla-oft"))

from guard.json_io import canonical_dumps, write_json

task_id = int(sys.argv[1])
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
with contextlib.redirect_stdout(io.StringIO()):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(task_id)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
    env.seed(0)
    description = task.language
    env.reset()
    sim = env.sim
    model = sim.model

payload = {
    "task_id": task_id,
    "task_name": task.name,
    "task_description": description,
    "body_names": [model.body_id2name(i) for i in range(model.nbody)],
    "site_names": [model.site_id2name(i) for i in range(model.nsite)],
    "free_joints": [
        model.joint_id2name(j) for j in range(model.njnt) if model.jnt_type[j] == 0
    ],
}
if output_path:
    write_json(output_path, payload)
else:
    print(canonical_dumps(payload, indent=2))
