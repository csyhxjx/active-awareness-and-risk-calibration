# Phase 4C Edge-Drop Counterfactual Pilot Protocol v3

Status: pre-registration before any Phase 4C implementation or rollout
Date: 2026-09-16
Predecessor evidence: `artifacts/phase4b_smoke/` at commit `657884e`
Manifest: `data/pilot_v0/manifest.json`
Manifest SHA-256: `2e3ef3bc7723fb59f823778b6f7201baff94959bb362f392815f0d45c66a7ecc`
Threshold contract: frozen `v1`; hard continuous violations have `margin < 0`

Phase 4B falsified release-over-table as an `object_drop` mechanism but
established reproducible pressure and lateral-push mechanisms. Phase 4C tests
only the remaining edge-escape hypothesis. It does not run a view-policy
comparison, change any frozen threshold or constraint, or reinterpret an
earlier result.

## V3.1 Scope and smoke states

Only Phase 4A's eight registered train states are eligible. Calibration and
test are excluded from branch discovery, predecessor replay, tuning, and
evaluation. The mechanism smoke contains exactly these states, in this order:

1. `task_07_init_023`: the fragile state where the Phase 4A C arm violated
   `gripper_env`;
2. `task_04_init_035`: a protected state where Phase 4A C did not violate
   (`min margin=0.0013488061997494528`) and the archived Phase 4B trace already
   demonstrates an elevated closed-gripper branch with 60 transitions left;
3. `task_07_init_021`: the random stratum result.

The random pool is the sorted Phase 4A state list after removing
`task_07_init_023` and all three C-negative protected candidates
(`task_04_init_032`, `task_04_init_035`, `task_08_init_003`). Rank each remaining
state by `SHA256("phase4c-v3-random-v1:" + state_id)` and select the smallest
digest; this yields `task_07_init_021`. The selection is frozen here and is not
repeated after feasibility observations.

Every arm uses a fresh environment and reproduces full70's per-task train-state
history before the target. JSON writes use `guard/json_io.py`; full and wrist
RGB are stored losslessly after every transition. State/RNG hashes, parent
hashes, protocol hash, Guard HEAD, GPU, and parity errors are persisted.

## V3.2 Truth-only branch and edge selection

Simulator truth in this section is allowed only for branch selection, state
screening, action-direction construction, and oracle labels. Persist it under
branch provenance. It is never included in model-visible RGB or supplied as an
input to a later visual method.

The payload is exactly `env.obj_of_interest[0]`, which must equal
`akita_black_bowl_1`. During exact archived baseline replay, record after every
transition its body-center `(x,y,z)`, the archived gripper command, and the
frozen monitor's table bounds `(x0,x1,y0,y1)` and `table_z`. Let `z0` be payload
z after stabilization step 9.

A carry candidate must satisfy all of the following:

- policy step `t >= 10`, at or after the first archived open-to-closed command;
- archived gripper command `> 0`;
- payload elevation `z_t-z0 >= 0.02 m`;
- all 60 archived actions `t+1..t+60` exist.

Select the candidate with maximum `z_t`; ties select the earliest step. This is
the frozen elevated-carry observation, and the branch begins from its post-action
state at archived action step `t+1`. If no candidate exists, the state is
ineligible and the smoke fails rather than shortening the horizon or changing
states.

At the selected carry point, the target center must lie inside or on the table
projection. Compute signed inward distances in fixed order:

`x-x0`, `x1-x`, `y-y0`, `y1-y`.

All must be finite and nonnegative. Select the minimum; ties use the order
above. This freezes the nearest edge, action index (`x=0`, `y=1`), outward
direction (`lower=-1`, `upper=+1`), and preflight carry-point-to-edge distance.
No workspace `xy_margin` is added: this experiment targets the physical table
projection, not the workspace oracle boundary.

Edge crossing is defined prospectively on the payload body center after a
transition: lower-edge crossing is coordinate `< lower_bound`; upper-edge
crossing is coordinate `> upper_bound`. Equality is not a crossing.

## V3.3 Arms and two-stage perturbation

The arm horizon is exactly `H=60` transitions beginning at the selected
carry-observation step plus one. All actions are seven-dimensional
`OSC_POSE` environment actions clipped to `[-1,+1]`. No policy model is loaded.

| Arm | Rule | Purpose |
| --- | --- | --- |
| `A_edge` | Replay all 60 archived actions unchanged. | Branch-matched parity and clean control. |
| `F_edge` | Execute the two stages below. | Edge escape followed by forced release. |

For `F_edge`, copy the archived action at every transition and overwrite only
the selected lateral component and gripper command:

1. **Push/hold stage:** set the selected x/y component to the frozen outward
   direction (`-1` or `+1`) and set gripper command to `+1`. This stage begins at
   arm step 0 and runs until a crossing is observed, with a maximum of 60
   transitions.
2. **Push/release stage:** beginning on the transition immediately after the
   first observed crossing, keep the same lateral override and set gripper
   command to `-1` through arm step 59. Its maximum is 59 transitions and both
   stages together always total 60.

If no crossing is observed by the final transition, stage two never starts;
record `edge_crossed=false`, `release_triggered=false`, and count no mechanism
hit. The crossing step, release-start step, target coordinates, edge distance,
and stage for every action are persisted. Visible edge crossing or release is
not an `object_drop` label: the frozen oracle still requires at least one free
object below `table_z - 0.03 m`.

Phase 4A pressure arms retain `H=30` semantics. They are absent from this smoke.
Only if all v3 smoke gates pass may a later, separately executed second-stage
dataset rerun `A`, `C04`, `C06`, and `C08` on all eight Phase 4A states at the
original contact branches and `H=30`, reporting the state-level dose curve.
That dataset is not authorized by this smoke execution and begins only after
the expansion decision is reported.

## V3.4 Staged execution

The code revision must be committed, pushed, and clean before execution.

1. Run `A_edge` alone for all three smoke states into a dedicated A-gate root.
   Checker success, exact branch state/RNG agreement, archived decision/flag
   equality, maximum absolute margin error `<=1e-7`, and zero A hard violation
   are required before any `F_edge` action runs.
2. Run `F_edge` for the same states into a separate smoke root. Stop launching
   further states as soon as the conjunction `object_drop positive` and
   `RGB-decisive evidence` is established; otherwise run exactly all three.
3. Rerun the first positive state in manifest order from a fresh environment.
   State/RNG hashes and `constraints.jsonl` bytes must match exactly.

No-view, random-view, action-conditioned-view, calibration, test, and the
eight-state pressure extension are forbidden during this execution.

## V3.5 Smoke gates

All five gates are mandatory.

| Gate | Pass condition |
| --- | --- |
| S1 integrity and control | All three `A_edge` arms pass the checker and archived parity, have zero hard violation and no `non_finite`, and share branch state/RNG hashes with their matched `F_edge` arm. |
| S2 mechanism switch | `F_edge` records both `edge_crossed` and `release_triggered`, causes a frozen `object_drop` hard violation, and places the registered target bowl itself below the same `table_z-0.03 m` line in at least 1/3 states. This prevents another free object from being counted as the edge-drop mechanism. |
| S3 observation switch | At least one true-positive `F_edge/object_drop` case is manually rated `decisive` in full or wrist RGB using onset-3/onset/onset+3 frames. Crossing/release without oracle violation is ineligible. |
| S4 oracle and isolation | Signed continuous constraints use exactly `margin < 0`; frozen boolean `self_collision.violated` remains authoritative because its margin is diagnostic zero. `k=3` is reported separately, no non-finite value occurs, simulator branch truth is absent from visual inputs, and only registered train states are present. |
| S5 reproducibility | The first positive state in manifest order reruns with identical branch state/RNG hashes and byte-identical `constraints.jsonl`. If no positive exists, S5 fails as not testable. |

View ratings are `{decisive, partial, none}` with a written reason per camera.
An object visibly leaving the supporting table and falling below its surface is
decisive; release alone, edge proximity, occlusion, or an off-frame object is
at most partial. Simulator margin remains the oracle and never substitutes for
the visual rating.

## V3.6 Terminal thesis fork

There is no v4 perturbation retry under this thesis question.

- **All S1-S5 pass:** record that the suite supplies its first reproducible,
  RGB-decisive positive family and therefore admits a future matched-budget
  view-policy comparison. Stop at the expansion decision; do not start that
  comparison in Phase 4C.
- **Any gate fails:** add the terminal conclusion to README that the current
  suite supplies no demonstrated RGB-decisive positive family for the proposed
  view-selection decision, redirect the thesis scope away from that claim, and
  do not run the method comparison.

Either branch closes mechanism search. Results must include the S table,
arm-by-constraint matrix, margin/onset plots, onset-adjacent dual-camera frames,
view ratings, A parity, branch/state/RNG provenance, checker output, and hashes.
