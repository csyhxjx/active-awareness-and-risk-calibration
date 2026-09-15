# Guard

Small experimental guard package for OpenVLA-OFT LIBERO evaluation.

This project keeps the upstream checkouts under `../code/` read-only. The
guard runner imports OpenVLA-OFT helpers from `../code/openvla-oft` and records parity data under
`../guard_workspace/runs`.

Example:

```bash
cd /internsdata/yewenhao
guard/run_guard_eval.sh \
  --pretrained_checkpoint /internsdata/yewenhao/models/models/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 2 \
  --center_crop True \
  --use_wandb False \
  --guard off \
  --parity_out /internsdata/yewenhao/guard_workspace/runs/official_actions_srv2.jsonl \
  --monitor_out /internsdata/yewenhao/guard_workspace/runs/constraints_off_srv2.jsonl
```

The wrapper sources `../env.sh`, which activates `project`, selects physical
GPU 3 by default, and keeps model caches and outputs on the data disk.

Probe the ten LIBERO spatial scenes and freeze the pilot initial-state split:

```bash
cd /internsdata/yewenhao
source ./env.sh
mkdir -p guard/data/scene_probe/company_server_20260915
for tid in 0 1 2 3 4 5 6 7 8 9; do
  python guard/scripts/probe_scene.py "$tid" \
    "guard/data/scene_probe/company_server_20260915/task_${tid}.json"
done
python guard/scripts/build_initial_states.py
```

The monitor only reads the current MuJoCo state. It records margins for
workspace, gripper/static contact, self-collision, object drop, and non-finite
state constraints, with only false-to-true transitions recorded as onsets.

Before freezing the sensor configuration, calibrate `gripper_env` on the
training split. If its onsets are concentrated at normal table-contact
grasping moments, exclude the table body from `static_geoms` or widen
`gripper_penetration`, record the chosen rule and threshold, and rerun the
monitored off/on parity check.

Compare two recordings:

```bash
cd /internsdata/yewenhao
source ./env.sh
PYTHONPATH=/internsdata/yewenhao/guard:$PYTHONPATH \
  python -m guard.parity.check_parity \
  /internsdata/yewenhao/guard_workspace/runs/official_actions_srv2.jsonl \
  /internsdata/yewenhao/guard_workspace/runs/guard_actions_srv2.jsonl
```
