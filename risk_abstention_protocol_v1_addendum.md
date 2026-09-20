# Phase 5F Prospective Model and Decision Addendum

Date: 2026-09-20. This addendum is part of the Phase 5F preregistration. It must be committed and visible on the remote before contract-probe or smoke code is written and before any Phase 5F environment is instantiated. It fixes the implementation choices intentionally left open in `risk_abstention_protocol_v1.md`.

## 1. Risk model

One example is `(layout, hidden_state, proposed_route, purchased_camera)` with binary target `1` iff the frozen `route_collision_v1` outcome for that proposed route collides. Formal training uses all three paid cameras from the new 60-layout train corpus; labels are training targets only and never inference inputs.

The candidate-aware model has two encoders. RGB is normalized to `[0,1]` and passed through three `3x3`, stride-2, padding-1 convolution blocks `3->16->32->64`, each followed by ReLU, then adaptive average pooling to 64 values. Public geometry uses the existing 31-value `geometry_features` schema and an MLP `31->64->64` with ReLU after each layer. Concatenated features pass through `128->64->1` with ReLU before the scalar logit. The candidate-agnostic model has the identical architecture and training examples but zeroes normalized geometry indices `[0:17]` exactly as the frozen Phase 5B ablation.

Training minimizes unweighted binary cross entropy with logits. Optimizer is AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 64, gradient-norm cap 5.0, maximum 200 epochs, validation-NLL early stopping patience 20, earliest epoch on exact ties, and deterministic seeds `[7101,7102,7103]`. Train-only feature mean/scale are frozen; scales below `1e-8` become `1.0`. No augmentation, pretrained download, architecture search, class weighting, or seed replacement is allowed.

Inference averages the three retained raw logits before calibration. Candidate-aware and candidate-agnostic ensembles are calibrated and thresholded separately.

## 2. Calibration and threshold

Temperature scaling uses validation only. For each ensemble, choose `T` from the inclusive grid `0.25,0.26,...,4.00` to minimize validation binary NLL of `sigmoid(mean_logit/T)`; exact ties choose the larger `T`. No binning, isotonic regression, threshold sweep against test, or post-test recalibration is permitted.

The validation operating point targets execution coverage `q=0.50`. Sort calibrated validation risks ascending. Let `n` be the number of normal left/right trials and `k=floor(0.50*n)`. If scores `r[k-1] < r[k]`, set `theta=(r[k-1]+r[k])/2`. If they tie, set `theta=r[k]`. Execution uses the strict rule `p_collision < theta`; therefore a boundary tie abstains as a group and achieved validation coverage may be below, never above, 50%. For `k=0`, set `theta=0`; for `k=n`, set `theta=1+2^-24`. Test reports the frozen threshold and realized coverage without adjustment.

## 3. Camera rules and B=1 boundary

The candidate-aware primary buys exactly one camera selected before pixel access by maximum public geometric route coverage, using the frozen Phase 5E coverage function and tie order `v_left > v_right > v_high`. It then evaluates risk for that camera only.

The candidate-agnostic comparator buys one fixed camera. Validation computes calibrated NLL for each of `v_left`, `v_right`, and `v_high` using the candidate-agnostic ensemble and chooses the minimum; ties resolve in that same order. Its threshold is then fit using only that chosen validation camera. Best-fixed, random, and geometric baselines use their preregistered rules but cannot select a second view. No policy observes an alternate-route image or switches candidates. Query count is exactly one for every non-proposer-stop normal trial.

## 4. Denominators and accounting

The all-trial denominator is every registered `(layout, hidden_state, route preference)` row, including abstentions and failures. `collision_rate = collisions/all_trials`; `execution_coverage = executed/all_trials`; `abstention_rate = abstained/all_trials`; and `collision_free_completion = collision_free_completions/all_trials`.

Conditional collision risk uses `collisions/executed`; if `executed=0`, risk and its upper bound are defined as `1.0`, so all-stop cannot pass. Safe-execution retention uses `executed_and_oracle_safe/oracle_safe_proposals`; oracle-safe means the frozen route rollout is collision-free and completes. Proposer failures/stops, invalid records, unresolved query failures, and model non-finite outputs remain separate counts and stay in the all-trial denominator. A non-finite risk is a hard model failure and forces abstention; it cannot be dropped.

## 5. Risk intervals and co-primary decision

All uncertainty is clustered by layout. Use 10,000 paired bootstrap resamples of complete layout groups with NumPy `PCG64` seed `20260922`. For conditional collision risk, recompute the ratio from aggregate collision and execution counts in each resample; a zero-execution resample contributes `1.0`. The risk-control upper bound is the 95th percentile. For safe-retention difference, recompute both ratios per resample and use percentile `[2.5%,97.5%]`; success requires the lower bound strictly above zero.

The candidate-aware policy succeeds only if its collision-risk upper bound is `<=0.05` and its paired safe-retention lower bound versus candidate-agnostic is `>0`. This is an intersection-union decision with no endpoint substitution. Point estimates, completion, marginal collision rate, calibration metrics, or a favorable baseline comparison cannot rescue either failure.

## 6. Contract probe and smoke

Before any train/validation collection or GPU training, a pure contract probe must cover finite-logit validation, sigmoid/calibration arithmetic, threshold ties and edge cases, strict execution semantics, camera tie rules, denominator reconstruction, bootstrap determinism, hidden-input rejection, tuple roundtrip, and fresh-process byte equality. A simulator probe must show query-preserved qpos, qvel, ctrl, RNG, and time.

The 8-layout seed-61008 train-only smoke uses a deterministic, non-trainable fixture risk function solely to test the interface. The fixture returns the route-specific obstacle-pixel fraction from the single purchased image, clipped to `[0,1]`; it is not a model result and creates no checkpoint. Smoke contains 64 normal trials and requires: exact namespace/scope; identical V0 and pre-query public input across four hidden states; exactly one registered paid query per trial; no unpurchased image access; finite fixture risk; 64/64 tuple reconstruction and replay; clear routes complete; blocked routes trigger the frozen collision oracle; and all reported denominators reproduce mechanically. Smoke performance is descriptive and cannot tune any frozen model, calibration, threshold, camera, or formal split rule.

Only smoke PASS opens the new formal 60-layout train split. Train PASS then opens the 20-layout validation split. The seed-63060 sealed test remains closed until model checkpoints, train normalization, validation temperatures, validation thresholds, fixed camera, power addendum, test checker, and evaluator are hash-frozen and pushed.
