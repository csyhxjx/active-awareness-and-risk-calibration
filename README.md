# Guard

Small experimental guard package for OpenVLA-OFT LIBERO evaluation.

This project intentionally keeps third-party checkouts under `../third_party/`
read-only. The guard runner imports OpenVLA-OFT helpers from
`../third_party/openvla-oft` and records parity data under
`../guard_workspace/runs`.

Example:

```bash
cd /root/gpufree-data
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl HF_HOME=/root/gpufree-data/hf-cache \
  conda run -n project python guard/run_libero_eval_guard.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 2 \
  --center_crop True \
  --use_wandb False \
  --guard off \
  --parity_out /root/gpufree-data/guard_workspace/runs/official_actions.jsonl
```

Compare two recordings:

```bash
cd /root/gpufree-data
PYTHONPATH=/root/gpufree-data/guard conda run -n project \
  python -m guard.parity.check_parity \
  /root/gpufree-data/guard_workspace/runs/official_actions.jsonl \
  /root/gpufree-data/guard_workspace/runs/guard_actions.jsonl
```
