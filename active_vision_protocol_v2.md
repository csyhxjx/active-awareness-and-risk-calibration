# Phase 5B formal active-vision comparison protocol

Date: 2026-09-17. Version: `active_vision_v2`. Status: pre-registration; no formal generator implementation, formal collection, selector training, or test evaluation exists at this commit.

## 1. Scope and evidence boundary

Phase 5A is development evidence only. Its 12 layout groups, 48 scenes, 96 candidate trajectories, camera tuning, gate decisions, and failed attempts are excluded from every Phase 5B train/validation/test split and from the Phase 5B confirmatory sample size. Passing P0-P5 establishes that the constructed observation mechanism is feasible; it is not a formal method comparison.

Phase 5B will create 120 new, independent layout groups: 60 train, 20 validation, and 40 sealed test. Each group contains hidden states `00`, `10`, `01`, and `11`; each scene has both fixed candidates. The resulting formal corpus contains 480 scenes and exactly 960 canonical candidate trajectories. Crossing the two public route preferences reuses these physical labels and does not create additional independent samples.

All layout derivatives stay in their parent group: four hidden states, route preferences, all cameras, appearance variants, replays, and candidate outcomes never cross a split. Development layouts and any deterministic transform of them are forbidden. The test split remains sealed until generator integrity, training, verifier calibration, and all validation-only choices are frozen and pushed.

Phase 5B/C uses the fixed controller and the independent `route_collision_v1` oracle. It makes no OpenVLA claim. Phase 5D remains unchanged: a short action-chunk label is not whole-task safety, proposer failures and candidate-recall failures are reported separately, and all methods share the same frozen proposer and candidate set.

## 2. Formal generator and grouped split

The implementation must materialize a canonical manifest before any formal rollout. Manifest IDs are `av2_train_000..059`, `av2_val_000..019`, and `av2_test_000..039`. Namespace seeds are train `51060`, validation `52020`, and test `53040`; scene dynamics seed is `20260917`. Layout sampling uses NumPy `PCG64` and records every sampled scalar rather than depending on later library defaults.

Each split is balanced over mirror sign and six public occlusion families. Train has 10 layouts per family. Validation uses counts `[4,4,3,3,3,3]`; test uses `[6,6,7,7,7,7]`. Family order is fixed in the manifest and test allocation is assigned before rollout. A family changes public occluder placement/orientation and camera coverage, not the hidden obstacle prior. Hidden states remain the full four-state factorial in every layout.

| Family | yaw center (rad) | divider y center (m) | side-camera azimuth center (deg) |
| --- | ---: | ---: | ---: |
| F0 | -0.10 | -0.035 | -8 |
| F1 | -0.10 | +0.035 | +8 |
| F2 | 0 | -0.035 | 0 |
| F3 | 0 | +0.035 | 0 |
| F4 | +0.10 | -0.035 | +8 |
| F5 | +0.10 | +0.035 | -8 |

Train assigns five layouts of each mirror sign inside every family. Validation mirror counts for F0-F5 are `[(2,2),(2,2),(2,1),(1,2),(2,1),(1,2)]`; test counts are `[(3,3),(3,3),(4,3),(3,4),(4,3),(3,4)]`, written as `(mirror +1, mirror -1)`. Thus each split is exactly balanced overall even where a family count is odd.

Continuous ranges are fixed prospectively:

| Parameter | Distribution |
| --- | --- |
| target x | uniform `[0.18, 0.22]` m |
| absolute route lane y | uniform `[0.16, 0.20]` m |
| hidden-obstacle x | uniform `[0.045, 0.085]` m |
| front-occluder x | uniform `[0.28, 0.32]` m |
| front-occluder yaw | family center in `{-0.10, 0, +0.10}` rad plus uniform `[-0.02,+0.02]` |
| public divider lateral offset | family center in `{-0.035,0,+0.035}` m plus uniform `[-0.01,+0.01]` |
| side-camera azimuth offset | family center in `{-8,0,+8}` degrees plus uniform `[-2,+2]` |
| side-camera field of view | uniform `[18,26]` degrees |
| high-camera azimuth offset | uniform `[-6,+6]` degrees |
| high-camera field of view | uniform `[26,34]` degrees |

Start pose, robot, controller, obstacle size/color, resolution, control frequency, target radius, hold duration, horizon, collision definition, query broker, shadow setting, and four hidden-state construction remain as in `active_vision_protocol_v1.md`. V2 adds hidden-state-independent, visual-only route ribbons centered on the two public candidate polylines: left is yellow `[0.9,0.8,0.1,1]`, right is cyan `[0.1,0.8,0.8,1]`, width `0.008` m, and height `table_z+0.001` m. They have no collision geometry and are identical across the four paired states.

The implementation may correct a prospective schema or integrity bug only in a new pushed commit before formal rollout. It may not tune these ranges, families, cameras, verifier thresholds, or gates from formal outcomes. If fewer than 54/60 train layouts satisfy the frozen Phase 5A physical/information integrity checks, formal collection stops and v2 is declared a generator failure; validation and test are not opened.

## 3. Shared verifier and decision contract

Every policy uses the same frozen RGB verifier and state machine. The verifier receives only the purchased 224 x 224 RGB and public candidate identity. It returns `blocked`, `clear`, or `unobserved`. The frozen color-shape rules are:

- left obstacle: `R>170, G<150, B<150`; right obstacle: `B>150, R<150, G<140`;
- left route ribbon: `R>170, G>150, B<100`; right route ribbon: `R<100, G>170, B>150`;
- at least 100 route-specific obstacle pixels is `blocked`;
- otherwise at least 100 route-specific ribbon pixels is `clear`;
- otherwise the result is `unobserved`.

The route ribbons make absence evidence observable only after purchase; they are public and do not reveal the hidden state in V0. The verifier cannot use a precomputed visibility label, hidden geometry, or oracle outcome. The geometric selector uses only its deliberately simple projection score before purchase, so learned-greater, learned-equal, and learned-worse outcomes all remain possible.

At B=1, a `clear` preferred candidate is executed, `blocked` is rejected, and `unobserved` stops. No policy may silently execute after an unresolved observation. Stop is never task completion. The B=2 fallback and all-three-view results are secondary budget/upper-bound analyses and are not part of the primary B=1 ranking.

Offline caching may store every view, but each policy receives images through a fresh observation broker. No selector may read unpurchased pixels, embeddings, masks, entropy, filenames, hidden coordinates, collision labels, or future frames.

## 4. Seven baselines and one primary method

| ID | Method | Budget | Frozen rule |
| --- | --- | ---: | --- |
| B0 | Always stop | 0 | Never executes; degeneracy control. |
| B1 | Nominal execute | 0 | Executes the public preferred candidate without querying. |
| B2 | Conservative no-view | 0 | With no coverage evidence, marks the candidate unresolved and stops. |
| B3 | Best fixed view | 1 | Chooses one of `V_left/V_right/V_high` using validation completion only; ties resolve in that order. Test is never consulted. |
| B4 | Random view | 1 | Uniform over three views, seeds `[7001,7002,7003,7004,7005]`; report each and their pre-specified mean. |
| B5 | Candidate-agnostic learned selector | 1 | Same network, optimization, seeds, and view features as the primary method, but candidate identity, preference, and route coordinates are zeroed. |
| B6 | Candidate-aware geometric selector | 1 | Selects maximum public candidate-path coverage after projecting the three route segments into each calibrated view and subtracting axis-aligned 2D bounding boxes of public occluders; it does not ray trace or inspect pixels. Ties resolve `V_left,V_right,V_high`. |
| M | Candidate-aware learned selector | 1 | Uses the learned score specified below and selects its maximum; ties resolve `V_left,V_right,V_high`. |

All-three-view (cost 3) and oracle candidate chooser are diagnostic upper bounds, not baselines in the matched B=1 ranking. Candidate-aware B<=2 fallback is a secondary budget curve. No method is selected or renamed after test evaluation.

## 5. Learned-selector training freeze

One training example is a `(layout group, candidate, paid view)` triple. Its counterfactual target is computed only from the four train hidden states:

`gain = mean(1[shared verifier produces the oracle-correct clear/blocked decision] - 1[verifier decision is incorrect])`.

`unobserved` contributes zero. The training score is `utility = gain - lambda * view_cost`, with `view_cost=1` for every B=1 view and `lambda=0.10`. Oracle truth creates offline targets only and is never an inference input.

The model is a small frozen specification rather than a hyperparameter search:

- V0 encoder: three convolution blocks `(3->16->32->64)`, each `3x3`, stride 2, padding 1, ReLU, then adaptive average pooling to 64 values.
- Public geometry encoder: two linear layers `64,64` with ReLU. Inputs are normalized candidate waypoints, route preference, candidate identity, candidate-to-camera projection summary, camera position/quaternion/FOV, and public occluder pose/size. Candidate-agnostic B5 zeroes candidate waypoints, preference, identity, and projection summary before this encoder.
- Score head: concatenated 128 values -> linear 64 -> ReLU -> linear 1. The same weights score all three views.
- Loss: per-view utility MSE weight `1.0` plus pairwise hinge-ranking loss weight `0.5`, ranking margin `0.10`. Pairs whose target utilities are equal are omitted. No additional auxiliary loss.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 64, gradient norm cap 5.0, maximum 200 epochs.
- Validation: choose the lowest validation utility-MSE epoch; early stopping patience 20; equal validation loss chooses the earliest epoch. No test-aware tuning.
- Training seeds: `[1701,1702,1703]`. The primary learned policy averages the three raw view scores and then applies `argmax(score - 0.10 * cost)`. Individual seeds are robustness results, not alternative shots at significance.
- Numeric preprocessing statistics are fit on the 60 train groups only. There is no image augmentation, pretrained-weight download, architecture search, threshold sweep, or post-test retraining.

Training and validation contain 60 and 20 independent layout clusters respectively, not `state x candidate x view` independent samples. The latter are correlated observations used inside each cluster.

## 6. Metrics and confirmatory comparison

The primary endpoint is collision-free task completion over all decision trials, with stop and timeout counted as non-completion. The synchronized constraint endpoint is collision rate over all trials. Execution coverage, collision conditional on execution, erroneous rejection of a successful candidate, stop rate, correct double-blocked stop rate, unresolved rate, query count, observation latency, and decision-flip rate are mandatory secondary metrics.

The sole confirmatory comparison is M versus B6 at B=1. For each sealed test layout, compute both policies over the four hidden states and two route preferences, then form the paired layout-level difference:

`Delta_CF = mean_layout(CF_completion_M - CF_completion_B6)`.

Similarly compute `Delta_collision`. Resample the 40 layout groups as indivisible paired clusters 10,000 times with `PCG64` seed `20260917`; report percentile 95% confidence intervals and raw numerators/denominators. No scene-, candidate-, or frame-level bootstrap is permitted. Train/validation/development layouts never enter a test interval.

A method-contribution result requires both:

1. the two-sided 95% clustered-bootstrap interval for `Delta_CF` has lower bound greater than zero; and
2. the one-sided 95% upper bound for `Delta_collision` is at most `+0.025` absolute.

All other baseline contrasts are secondary. For simultaneous learned-M versus B0-B5 contrasts, use Holm-Bonferroni at family-wise 0.05 and still report unadjusted estimates/intervals. Budget B=2 and diagnostic upper bounds are descriptive.

## 7. Sample size and power limitation

The 12 Phase 5A layouts are not part of this calculation. In development, the layout-level B6 proxy was exactly 0.50 and every fixed-view score exactly 0.25, so those paired variances degenerated to zero and cannot justify a small formal sample. Across the five registered random-view seeds, the layout-cluster SD of geometric-minus-random completion ranged from `0.0942` to `0.1410`. This protocol rounds the maximum upward to `s_plan=0.142`.

With 40 independent test layout groups, two-sided alpha `0.05`, power `0.80`, and the normal paired approximation, the planned minimum detectable absolute effect is:

`(1.959964 + 0.841621) * 0.142 / sqrt(40) = 0.0629`, or 6.3 percentage points.

This is a planning approximation, not guaranteed power for learned-versus-geometric differences. Effects smaller than 6.3 points are explicitly underpowered at the planned variance. Realized clustered intervals govern the result. The test set is not enlarged after viewing results, and the 60 train or 20 validation groups cannot be borrowed to improve the confirmatory interval.

## 8. Pre-registered interpretation fork

If M satisfies both confirmatory conditions against B6, the permitted claim is: a learned candidate-aware view selector improves collision-free completion over the simple public-geometry rule at matched B=1 without exceeding the registered collision-rate margin. This is the Phase 5 method contribution; it is still limited to the fixed-controller controlled-occlusion benchmark.

If M does not satisfy both conditions, including equality, a confidence interval crossing zero, or failure of the collision constraint, no learned-selector advantage is claimed. The permitted conclusion is: action-related view selection is mechanistically valuable in this benchmark, but its value is realizable by the simple geometric rule; the contribution is downgraded from learning-method innovation to mechanism validation. This remains a valid answer to H2. Individual seeds, secondary baselines, B=2, or OpenVLA results cannot overturn this fork.

Neither branch authorizes a whole-task safety statement. Phase 5D may begin only after the Phase 5B/C artifacts, frozen model hashes, and this interpretation fork are reported without modification.

## 9. Prospective clarification A: staged opening and train gate

Date: 2026-09-17. This clarification is additive and precedes all v2 manifest/generator/checker code and every formal rollout. It does not replace the original pre-registration or its recovery bundle.

The canonical manifest fixes all 120 exact public layouts and split assignments before rollout. Manifest visibility does not unseal a split: code, humans, training, selection, and reports may consume only rows whose split is currently opened. In particular, no validation or test environment may be instantiated during the train gate. Test public geometry is fixed but must not be used for feature normalization, debugging, model selection, threshold selection, or code-path decisions.

Execution proceeds through three irreversible stages:

1. **Train gate only:** collect exactly 60 train layouts, 240 scenes, and 480 canonical candidate trajectories. Run the integrity/mechanism checker and stop for a gate decision.
2. **Training and validation:** only after train PASS, train the three registered seeds and candidate-agnostic controls, then collect/use the 20 validation layouts for checkpoint selection and B3 fixed-view selection. Push the selected checkpoint hashes, frozen normalization statistics, chosen B3 camera, and final analysis implementation before test.
3. **One frozen test:** only after the preceding freeze, instantiate and collect the 40 test layouts once. No retraining, threshold change, seed replacement, camera change, or sample-size extension follows test inspection.

The train gate is fixed as follows:

| Gate | Requirement |
| --- | --- |
| T0 scope | Exactly 60 registered train layouts, 240 scenes, 480 canonical route files/results, four RGB views per scene, and one exact representative replay per layout. No validation/test output path exists. |
| T1 provenance | All rows use one clean pushed Guard HEAD and one manifest/protocol hash; every query preserves physical/controller/RNG hash; all 60 representative replays are byte-identical. |
| T2 hard integrity | All required files parse; no duplicate IDs, missing states/routes/views, NaN/Infinity, or hidden/public schema mismatch. `V0` pixels and every public non-visual input are identical across the four states in all 60 layouts. Required pass: 60/60. |
| T3 physical mechanism | For both routes, clear states complete collision-free and blocked states collide under `route_collision_v1`. Required pass: at least 54/60 layouts. |
| T4 paid-view mechanism | The frozen RGB verifier produces the oracle-correct clear/blocked result from at least one paid view for each route and no single paid view resolves both routes in every layout. Required per-layout pass: at least 54/60. Across the full train split, at least 20 layouts have a left-candidate view unavailable from the best right/high alternative and at least 20 have the symmetric right-candidate property. |
| T5 accounting | All 60 layouts, including every T3/T4 failure, remain in the train corpus and report. No regeneration, substitution, exclusion, or tuning from formal outcomes is allowed. |

T0, T1, T2, and T5 are hard 60/60 requirements. T3 and T4 use the pre-registered 54/60 tolerance for sampled geometry. Any hard-gate failure or either mechanism count below 54 stops v2 as a generator failure; validation and test stay sealed. A passing train gate authorizes training and validation only, not test.
