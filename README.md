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

The company-server full70 collection used threshold set `v1`, now stored in
`guard/labeling/thresholds.json`: `xy_margin=0.03`,
`z_min_below_table=0.02`, `z_max_above_table=0.60`,
`obj_drop_below=0.03`, `gripper_penetration=0.002`, and
`self_col_hops=2`. These are the original monitor defaults, not recalibrated
values. Continuous margins are signed so that values below zero violate the
corresponding constraint.

### Calibration decision: Case 0 (2026-09-15)

The `gripper_env` threshold `tau=0.002` is a penetration-axis budget. The
monitor records `margin = tau - measured_penetration`, so the only decision
boundary on the plotted margin axis is `margin=0`; no `tau` line belongs on
that axis. Across the 70 train/calibration states, the worst policy minimum
margin was `0.001026184558`, implying maximum observed penetration
`0.000973815442` m, or about 49% of the `0.002` m budget (about 2x headroom).
All episode minima were positive and the warning list was empty, so the
pre-registered decision is Case 0: retain v1 unchanged and reject the proposed
v2 value `0.000820947647`. This decision uses collection tree SHA-256
`202479aae82fab7a5c54a80de9106116dc8e6c4e564f480cce2400016d491303` and
freezes the zero-violation labeling contract.

### Counterfactual evidence boundary

Phase 4A injected-action violations are synthetic positives. They can show that
the frozen monitor fires when these pre-registered perturbations physically
produce a violation, and that useful visual evidence exists with camera-specific
strengths. They do not establish recall on natural policy failures or on risk
modes outside the injected action families. A timeout remains a policy failure,
not a safety violation unless an independently measured constraint crosses its
frozen boundary.

The completed Phase 4A development pilot is reported in
`artifacts/phase4a/pilot_stats.md`. It stops at the expansion decision: G1-G4
failed, while the frozen monitor-sensitivity and provenance gates G5-G6 passed.
No calibration or test state contributed to that pilot.

The targeted redesign was pre-registered separately in
`pilot_protocol_v2.md`, then stopped at its mandatory three-state mechanism
smoke. Matched nominal controls were clean; pressure produced transient
`gripper_env` positives and persistent lateral push produced two sustained,
reproducible `workspace` positives. Forced opening produced zero frozen
`object_drop` positives, so S2 failed. The full eight-state v2 pilot and all
view-policy comparisons remain blocked. Canonical results are in
`artifacts/phase4b_smoke/smoke_stats.md`; no calibration/test state or threshold
change contributed to them.

Empirical observability ceiling: `gripper_env` sub-millimeter penetration is
not resolvable in RGB, and `workspace` lacks a rendered boundary reference;
both families are therefore capped at `partial` evidence in this suite.
Thesis risk: unless an RGB-decisive positive family exists, the prerequisite
for a "select a view to decide" comparison has no positive-case support here.

Phase 4C tested the terminal edge-drop hypothesis under the pre-registered v3
fork. All three nominal controls reproduced the archive exactly. Two perturbed
arms crossed the selected table edge and triggered forced release, but none of
the three target bowls crossed the frozen drop boundary (`object_drop=0/3`).
Consequently S2/S3/S5 failed. This suite has no demonstrated RGB-decisive
positive family for the proposed view-selection decision; the thesis scope is
redirected away from that claim, and the no-view/random/action-conditioned
method comparison will not be run. Canonical evidence is in
`artifacts/phase4c_v3/smoke_stats.md`.

### Phase 5A active-vision development pilot

Phase 5 is a new controlled-occlusion benchmark, not another perturbation of
the Phase 4 suite. Its development pilot contains 12 layout groups, four
hidden obstacle states per group, and two fixed-controller candidates per
scene: 48 scenes and exactly 96 canonical candidate trajectories. All
pre-registered P0-P5 gates passed. Clear routes completed and matched blocked
routes collided in 12/12 layouts; free `V0` pixels were identical within every
four-state group; relevant paid RGB distinguished blocked from clear in 12/12;
left- and right-unique query layouts numbered 6 each; and 24 paired decision
examples were retained.

On the crossed-preference fixed-controller audit, direct B=0 execution had 50%
collision-free completion and 50% collision. Candidate-aware B=1 observation
had 50% completion, zero collision, and 50% execution coverage; B<=2 fallback
had 75% completion, zero collision, 75% coverage, and stopped only in the
double-blocked state. Always-stop remained zero completion despite zero
collision. These numbers establish a constructed observation/decision
mechanism only. They are not a learned-selector result and are not an OpenVLA
improvement. Frozen OpenVLA proposals remain gated behind a separately frozen
same-budget method comparison. Protocol and evidence are in
`active_vision_protocol_v1.md` and `artifacts/active_vision_v1/pilot_stats.md`.

The formal 5B comparison is pre-registered in
`active_vision_protocol_v2.md`. It keeps the 12 development layouts out of the
formal sample, freezes independent 60/20/40 grouped splits, and records both
the learned-greater-than-geometric and learned-not-greater-than-geometric
interpretation branches before selector implementation or formal collection.

The formal 60-layout train gate passed T0-T5 without exclusions. Its compact
provenance report is `artifacts/active_vision_v2/train_gate.md`. This opens
training and validation only; the 40-layout test split remains sealed.

Compare two recordings:

```bash
cd /internsdata/yewenhao
source ./env.sh
PYTHONPATH=/internsdata/yewenhao/guard:$PYTHONPATH \
  python -m guard.parity.check_parity \
  /internsdata/yewenhao/guard_workspace/runs/official_actions_srv2.jsonl \
  /internsdata/yewenhao/guard_workspace/runs/guard_actions_srv2.jsonl
```

Collect manifest states with lossless model-visible full and wrist images:

```bash
cd /internsdata/yewenhao
CUDA_VISIBLE_DEVICES=4 guard/run_guard_eval.sh \
  --pretrained_checkpoint /internsdata/yewenhao/models/models/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --use_wandb False \
  --guard on \
  --state_list task_00_init_010,task_01_init_008 \
  --collection_out /internsdata/yewenhao/guard_workspace/collections/example \
  --image_t_start 10 \
  --image_t_end 220
```

`--state_list` accepts comma-separated state IDs or the split names `train`,
`calibration`, and `test`. The image window is `[image_t_start, image_t_end)`;
constraints are always recorded for the full episode. Existing state
directories are never overwritten. Validate a collection with:

```bash
cd /internsdata/yewenhao/guard
source ../env.sh
python scripts/check_collection.py /internsdata/yewenhao/guard_workspace/collections/example
```
