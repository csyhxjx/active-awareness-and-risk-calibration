# Phase 5A active-vision development pilot protocol

Date: 2026-09-17. Version: `active_vision_v1`. Scope: development pilot only.

## Frozen design

The pilot contains 12 layout groups from `configs/active_vision_v1/pilot.json`. Each group has hidden obstacle states `00`, `10`, `01`, and `11`, giving 48 scenes. Both fixed candidates (`left_route`, `right_route`) are reconstructed in independent fresh environments from the same registered scene and seed; their complete pre-action physical/controller/RNG hashes must match before execution. Each candidate is executed once per scene, giving exactly 96 canonical candidate trajectories. Development layouts and their derivatives are excluded from future validation and test splits.

The free view is `V0`. Paid views are `V_left`, `V_right`, and `V_high`; RGB is 224 x 224. Query budget is measured in released images. Data generation uses an explicitly marked cache budget of three to persist all counterfactual views; every evaluated policy receives them through a fresh broker with B=1. A broker rejects free-camera and over-budget paid requests, records latency and pre-query state hash, and raises if qpos, qvel, act, ctrl, simulator time, controller numeric state, Python RNG, or NumPy RNG changes during rendering. Hidden coordinates, contacts, segmentation, future frames, and unpurchased RGB are forbidden policy inputs.

The fixed controller uses Panda `OSC_POSE` at 20 Hz, at most 100 steps, a 2 cm target radius, and five consecutive in-radius steps. Collision truth is robot contact with the registered route obstacle. A result is a collision-free completion only when it reaches the target and has no registered collision. Clear and blocked outcomes are both retained; gates are not edited after observing the 12-layout run.

Shadows are disabled. Within a layout, the four hidden states differ only by moving the two static route obstacles between their registered location and an off-scene location. Camera poses are functions of public layout geometry, never hidden state. `V_left` and `V_right` are narrow path-centered views. `V_high` looks across a public visual-only central divider, so it exposes only the physically near lane without changing candidate dynamics; mirroring makes that lane correspond to logical left in some layouts and logical right in others. Its utility is therefore geometry-dependent rather than fixed to a candidate or forced to zero. The main front occluder and hidden route obstacles retain collision geometry.

## Gates

- P0: every query is state invariant; broker access tests pass; public routes are identical across hidden states; one canonical representative per layout is deterministically re-executed from the same snapshot and its outcome record must match.
- P1: in at least 10 of 12 layouts, each route completes without collision when its bit is clear and collides when its bit is blocked. Every exception remains in the report.
- P2: all `V0` pixels and all public non-visual inputs are identical across the four hidden states of every layout.
- P3: an RGB-only audit correctly separates blocked from clear for each relevant side view in at least 10 of 12 layouts. Oracle outcomes are checked separately.
- P4: at least four layouts require `V_left` for the left candidate and at least four require `V_right` for the right candidate; neither the opposite side view nor `V_high` may universally solve both. The best fixed view remains a later baseline.
- P5: at least six paired examples share identical `V0` and candidate but, after the same relevant query, correctly allow and execute the clear member while rejecting or stopping the blocked member.

Failure of any mandatory gate stops expansion and triggers scene/control diagnosis. It is not rewritten as a negative result about active vision.

## Metrics and staging

The primary system metric is collision-free task completion over all trials. Collision rate, execution coverage, erroneous rejection of a successful candidate, stop rate, and mean/quantile observation cost are reported together. A zero-collision all-stop policy has zero completion and cannot rank first. Metrics are paired by layout; multiple hidden states do not count as independent layout samples.

Phase 5A uses only the fixed controller to isolate the observation mechanism. Passing P0-P5 permits a separately frozen same-budget comparison; it does not itself establish a learned-method advantage. Frozen OpenVLA proposals are integrated only after the fixed-controller mechanism and evaluation contract pass. Any fixed-controller gain must not be described as an OpenVLA improvement.
