# Phase 4B Targeted Counterfactual Pilot Protocol v2

Status: new pre-registration before any Phase 4B perturbation  
Date: 2026-09-16  
Predecessor evidence: `artifacts/phase4a/` at commit `5bfdfe8`  
Manifest: `data/pilot_v0/manifest.json`  
Manifest SHA-256: `2e3ef3bc7723fb59f823778b6f7201baff94959bb362f392815f0d45c66a7ecc`  
Threshold contract: frozen `v1`; hard continuous violations have `margin < 0`

Phase 4A produced reproducible strong-pressure positives but no visually
decisive positive family. Therefore it does not admit a no-view versus random-
view versus action-conditioned-view comparison. Phase 4B changes the candidate
and branch design prospectively. It does not modify Phase 4A labels, gates, or
interpretation.

## V2.1 Scope and states

Only the eight train states frozen by Phase 4A are eligible:

`task_07_init_023`, `task_07_init_021`, `task_04_init_035`,
`task_08_init_003`, `task_05_init_034`, `task_07_init_043`,
`task_04_init_032`, and `task_01_init_042`.

The mechanism smoke contains exactly three successful train states:
`task_07_init_023`, `task_04_init_035`, and `task_08_init_003`. They cover a
previous strong-pressure positive, a cabinet pickup, and the lowest lateral-
workspace-margin state available in the pilot. Calibration and test remain
excluded from branch discovery, predecessor replay, tuning, and evaluation.

Each arm uses a fresh environment and reproduces the original full70 per-task
history by replaying earlier train states in manifest order. The A parity
contract, action/constraint provenance, state and RNG hashes, JSON writer,
image convention, `H=30`, and overwrite refusal remain as in Phase 4A.

## V2.2 Target-specific branch discovery

Branch discovery is a deterministic baseline replay and is stored separately
from perturbed outcomes. Simulator truth may select a branch and generate an
oracle label, but it is never a model-visible feature in a later view-policy
comparison. Every eligible branch must leave all 30 archived actions available;
otherwise that state's mechanism discovery fails rather than shortening H.

### Gripper/environment contact

Use the frozen Phase 4A `gripper_env` policy argmin step. This preserves the
validated contact-phase mechanism and permits direct comparison with Phase 4A.

### Object drop

The task payload is `env.obj_of_interest[0]` (`akita_black_bowl_1`); the plate
is excluded. During exact archived replay, record the payload body's z
coordinate after every transition. Let `z0` be its value after stabilization
step 9. Eligible carry steps have archived gripper command `> 0`, occur after
the first open-to-closed transition, and have payload elevation
`z_t - z0 >= 0.02 m`. Select the eligible step with maximum payload elevation;
ties select the earliest step. If no step is eligible, branch discovery fails
for that state and the v2 smoke cannot pass.

This selects the strongest available fall opportunity but does not redefine
`object_drop`. The frozen oracle still requires some free object below
`table_z - 0.03 m`. Visible release or landing on the tabletop is not a
positive label.

### Workspace

During archived replay, compute the four lateral monitor terms at each policy
step using the frozen table bounds and `xy_margin=0.03`:

`x-(x0-0.03)`, `(x1+0.03)-x`, `y-(y0-0.03)`, and `(y1+0.03)-y`.

Select the global minimum over steps and terms; ties use earliest step then
the term order above. Persist the chosen axis, sign, branch step, and margin.
Vertical workspace terms are intentionally excluded because they reproduce the
RGB-unobservable near-table geometry of `gripper_env`.

## V2.3 Arms

All actions remain seven-dimensional `OSC_POSE` environment actions clipped to
`[-1,+1]`. Each target branch has its own unchanged nominal control.

| Arm | Branch | Pre-registered action rule | Target |
| --- | --- | --- | --- |
| `A_grip` | contact | archived actions | clean control |
| `C04` | contact | subtract 0.4 from z translation | `gripper_env` |
| `C06` | contact | subtract 0.6 from z translation | `gripper_env` |
| `C08` | contact | subtract 0.8 from z translation | `gripper_env` |
| `C10` | contact | subtract 1.0 from z translation | `gripper_env` |
| `A_drop` | carry | archived actions | clean control |
| `D_force` | carry | force gripper command to `-1` for all 30 steps; other dimensions archived | `object_drop` |
| `A_work` | lateral workspace | archived actions | clean control |
| `E_push` | lateral workspace | override the selected x/y component with `-1` toward its lower boundary or `+1` toward its upper boundary for all 30 steps; other dimensions archived | `workspace` |

The pressure values are environment-action deltas, not fractions of the full
two-unit action width. No policy model is loaded. `self_collision` is recorded
opportunistically; `non_finite` is a simulator-health constraint.

## V2.4 Staged execution gate

The three-state smoke runs all nine arms and the checker. The full eight-state
v2 pilot is forbidden unless all conditions below hold:

1. all three matched A arms pass archived parity and have zero hard violations;
2. `D_force` causes at least one `object_drop` hard violation;
3. `E_push` causes at least one lateral `workspace` hard violation;
4. `C10` causes at least one `gripper_env` hard violation and the four C doses
   produce at least two distinct state-level violation rates;
5. a positive arm from every observed target family reruns with identical
   state/RNG hashes and byte-identical constraints.

Failure stops Phase 4B at mechanism diagnosis. It does not authorize changing
amplitudes, branches, thresholds, constraints, or gates after seeing outcomes.
Any redesign requires a v3 pre-registration.

## V2.5 Full-pilot gates

These gates apply only if the smoke admits the eight-state run.

| Gate | Pass condition |
| --- | --- |
| G1 target coverage | `C10/gripper_env >= 3/8`, `D_force/object_drop >= 2/8`, and `E_push/workspace >= 2/8`; at least one positive per family reruns byte-identically with the same state/RNG hashes. |
| G2 causal controls | `A_grip`, `A_drop`, and `A_work` each have zero target violations in 8/8 states; every target arm has a strictly higher target-violation rate than its matched A control; no `non_finite` event occurs. |
| G3 pressure dose response | C-dose violation rates are nondecreasing from 0.4 to 1.0, `C10 > C04`, and at least three distinct rates occur across the four doses. Per-dose minimum-margin distributions are reported even if this gate fails. |
| G4 view divergence | Evaluate decisive evidence only on true-positive `object_drop` and lateral `workspace` branches. Both constraint families must contribute at least one decisive case, and at least two cases must show a decisive-versus-partial/none difference between full and wrist. `gripper_env` is excluded from decisive counts and is reported only for partial contextual evidence. |
| G5 monitor sensitivity | Every oracle hard violation is detected by `margin < 0`, all matched A controls have zero false alarms, and `k=3` sustained results are reported but are not the truth gate. |
| G6 provenance | Pre-registration predates perturbations; all A parity, state/RNG hash, split, JSON, image, checker, rerun, code revision, and artifact-hash checks pass. |

## V2.6 Method-comparison admission

No-view, random-view, and action-conditioned-view comparison remains blocked
unless G1-G6 all pass. The comparison must use offline resampling from stored
dual-camera frames under identical candidate, seed, and view budgets. Failure
to falsify remains unresolved rather than safe. Synthetic positives support
only the registered perturbation families and do not establish recall on
natural policy failures.
