# Guard

Small experimental guard package for OpenVLA-OFT LIBERO evaluation.

This project intentionally keeps third-party checkouts under `../third_party/`
read-only. The guard runner imports OpenVLA-OFT helpers from
`../third_party/openvla-oft` and records parity data under
`../guard_workspace/runs`.

Example:

```bash
cd /root/gpufree-data
guard/run_guard_eval.sh \
  --pretrained_checkpoint /root/gpufree-data/hf-cache/openvla-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 2 \
  --center_crop True \
  --use_wandb False \
  --guard off \
  --parity_out /root/gpufree-data/guard_workspace/runs/official_actions.jsonl \
  --monitor_out /root/gpufree-data/guard_workspace/runs/constraints_off.jsonl
```

The wrapper points both `HF_HOME` and `HF_HUB_CACHE` at the local model cache,
which is required when the cache uses the legacy `models--...` layout.

Probe the ten LIBERO spatial scenes and freeze the pilot initial-state split:

```bash
mkdir -p /root/gpufree-data/guard_workspace/scene_probe
for tid in 0 1 2 3 4 5 6 7 8 9; do
  conda run --no-capture-output -n project python guard/scripts/probe_scene.py "$tid" \
    > "/root/gpufree-data/guard_workspace/scene_probe/task_${tid}.json"
done
conda run --no-capture-output -n project python guard/scripts/build_initial_states.py
```

The monitor only reads the current MuJoCo state. It records margins for
workspace, gripper/static contact, self-collision, object drop, and non-finite
state constraints, with only false-to-true transitions recorded as onsets.

Compare two recordings:

```bash
cd /root/gpufree-data
PYTHONPATH=/root/gpufree-data/guard conda run -n project \
  python -m guard.parity.check_parity \
  /root/gpufree-data/guard_workspace/runs/official_actions.jsonl \
  /root/gpufree-data/guard_workspace/runs/guard_actions.jsonl
```
