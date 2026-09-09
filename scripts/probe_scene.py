"""Export one LIBERO task's MuJoCo scene names and free joints."""

import json
import contextlib
import io
import sys

sys.path.insert(0, "/root/gpufree-data/third_party/openvla-oft")

task_id = int(sys.argv[1])
with contextlib.redirect_stdout(io.StringIO()):
    from experiments.robot.libero.libero_utils import get_libero_env
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(task_id)
    env, description = get_libero_env(task, "openvla")
    env.reset()
    sim = env.sim
    model = sim.model

print(
    json.dumps(
        {
            "task_id": task_id,
            "task_name": task.name,
            "task_description": description,
            "body_names": [model.body_id2name(i) for i in range(model.nbody)],
            "site_names": [model.site_id2name(i) for i in range(model.nsite)],
            "free_joints": [
                model.joint_id2name(j) for j in range(model.njnt) if model.jnt_type[j] == 0
            ],
        },
        indent=2,
    )
)
