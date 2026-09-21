# Phase 6B: RGB Observation Validation Protocol v1

Date: 2026-09-21. Status: prospective protocol only. This phase validates
whether the Phase 6A7 controlled cue-ROI mechanism survives an actual RGB
observation model. It does not modify, relabel, or reuse the Phase 6A7 formal
result, manifest, checker, raw root, or complete-history bundle.

## 1. Scope and claim boundary

Phase 6A7 is frozen as a controlled exact-belief ROI color-cue mechanism. Its
adaptive gain uses the same query budget, route candidates, stopping rule, and
physical controller as the exhaustive public-layout fixed baseline. The fixed
baseline is the public-layout envelope over all 17 ordered B=2 sequences, not a
per-hidden-state choice. States `000/111` are physical controls and are outside
the primary adaptive-gain denominator. None of that result is evidence that an
RGB model understands natural obstacles.

Phase 6B asks a narrower question:

> How much of the frozen exact-belief gain remains when the registered ROI
> observation is produced by a held-out-layout RGB detector?

The mechanism still receives only the registered cue ROI and detector output.
Full RGB frames are archived for audit. Pixels outside the ROI, simulator
state, hidden state, collision result, and route outcome are forbidden inputs to
the detector and planner.

## 2. New data and split freeze

The 12 Phase 6A7 layouts are permanently excluded from every Phase 6B split.
The new namespace is `belief_active_vision_rgb_v1`; generation seed is `66301`.
The grouped split is fixed before any detector training:

- 24 train layout groups;
- 12 validation layout groups;
- 12 sealed test layout groups;
- eight hidden states per group, with `000/111` controls reported separately;
- every layout group owns all of its mirrored and derived scenes;
- no image, crop, state, route, or geometry from one group may cross splits.

Each group uses the same public layout randomization family as the mechanism
task but a new deterministic draw. The manifest records the split, seed,
protocol hash, layout-group id, observation table, cue ROI, public geometry, and
hidden-state isolation fields before images are generated. The sealed test
manifest is created once after the detector and planner freeze and is never
opened for diagnosis or tuning.

## 3. Frozen observation methods

All methods use the same belief updater, adaptive planner, B=2 budget, stopping
rule, candidate routes, tie-breaks, and official controller. Only the source of
the observation differs:

1. **Oracle ROI:** registered symbolic observation from the frozen table; this
   is the Phase 6A7 upper-bound reference, not an RGB result.
2. **RGB detector:** a detector trained only on train layouts and calibrated
   only on validation layouts. Its input is the purchased ROI crop plus the
   camera id; it cannot receive hidden state, route outcome, simulator truth,
   unpurchased views, or full-frame pixels.
3. **Best fixed B=2:** the public-layout exhaustive 17-sequence envelope, using
   the same RGB detector outputs only when a fixed sequence queries a view. Its
   sequence-selection rule is frozen on validation and cannot inspect test
   outcomes.
4. **Noisy observation baseline (optional, pre-registered):** take oracle
   symbolic outputs and apply the validation-estimated confusion matrix. This
   isolates planner sensitivity to observation noise from visual feature
   learning. It is reported only if the confusion-matrix construction passes
   its validation-only audit; it cannot replace the RGB detector comparison.

No RGB method may use the Phase 6A7 images as training or validation examples.
No particle filter, learned particle proposal, continuous hidden geometry, or
new route planner is authorized in this phase.

## 4. Detector training and calibration

The detector architecture, augmentation family, optimizer, maximum epochs,
early-stopping rule, and random seeds are selected on train/validation only and
written to a freeze record before sealed test generation. Calibration is fit on
validation predictions only. The detector output schema is fixed as:

```text
camera_id, predicted_outcome, confidence, calibrated_probabilities,
model_hash, detector_version
```

An `unobserved` output is allowed and is not silently converted to `clear`.
The planner must treat low confidence or contradictory outputs as unresolved
evidence and may stop. A detector checkpoint is eligible for test only if it
has finite outputs, a complete confusion matrix, and no split/provenance
violations.

## 5. Primary and diagnostic metrics

Metrics are reported separately for six main states and controls `000/111`, at
layout-group level and pooled:

- ROI recognition accuracy and macro-F1;
- hidden-state confusion matrix, including `unobserved` and contradiction
  outcomes;
- adaptive decision error rate relative to the oracle decision;
- collision-free completion rate;
- collision rate, stop rate, mean query count, and mean utility;
- route-switch success when the initial preferred route is blocked;
- detector latency and end-to-end query latency.

The primary RGB endpoint is the difference between RGB-detector adaptive and
the validation-frozen best-fixed B=2 policy on collision-free completion,
co-reported with collision rate. Oracle ROI is an upper bound and is not a
learned-method comparator. The noisy baseline is diagnostic.

Layout group is the bootstrap unit. The bootstrap seed, replicate count,
two-sided interval rule, and any multiplicity adjustment are frozen in the
formal split addendum before test. Test results are generated once after model
and planner freeze; no test observation may change thresholds, calibration,
the fixed sequence, or the narrative.

## 6. Hard gates and terminal interpretations

Before opening sealed test, train/validation must pass:

- exact split disjointness and exclusion of every Phase 6A7 layout;
- hidden-state/public-input isolation;
- complete ROI/state/view table and no unregistered ROI object;
- detector finite-output, calibration, and fresh-process replay checks;
- broker budget/state-preservation checks;
- candidate recall and route-controller checks on validation.

If RGB recognition fails while oracle adaptive gain remains, report perception
loss separately; do not call it a planner failure. If RGB recognition passes
but adaptive decisions fail, report belief/planner sensitivity separately. If
both pass but completion does not improve over fixed, report that the mechanism
does not transfer at the tested visual quality. No branch authorizes particle
methods.

Only after this RGB phase is stable may a new protocol introduce continuous
hidden geometry. That later ladder is fixed as MAP scene estimate, ordinary
particle filter, oracle belief upper bound, and only then a learned particle
proposal/update if ordinary PF has a stable, reproducible bottleneck.

## 7. Run order

1. Commit and push this protocol and the split addendum before detector code or
   image generation.
2. Generate the new grouped train/validation/sealed-test manifests and audit
   hashes, counts, and exclusions.
3. Train/calibrate the RGB detector on train/validation only.
4. Freeze detector, calibration, planner, fixed sequence, evaluator, and
   provenance hashes.
5. Run one sealed test and report oracle, RGB, fixed, and optional noisy rows.
6. Stop at the interpretation branch above. Do not start continuous geometry
   or particle methods in this protocol.
