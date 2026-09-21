# Phase 6C: Continuous Hidden-Geometry Active Observation Protocol v1

Date: 2026-09-21. Status: prospective protocol only. This document must be
committed and visible on the remote before any Phase 6C simulator code,
manifest, probe, particle run, RGB training, or formal collection is created.

Phase 6A7 and Phase 6B remain immutable controlled cue-ROI results. Phase 6C
uses a new natural-geometry scene, namespace, seeds, manifests, and raw roots.
It does not reuse their formal layouts, RGB crops, labels, checkpoints, or
sealed-test outcomes. The older `belief_active_vision_protocol_v1.md` used
"6B" for a future continuous task and "6C" for learned particles. The actual
cue-ROI RGB study now occupies Phase 6B, so this additive protocol names the
continuous task Phase 6C. Learned particles move to a possible Phase 6D and
remain forbidden here.

## 1. Question and claim boundary

Phase 6C asks:

> When obstacle position and route clearance are continuous and uncertain,
> does an adaptive second view outperform the strongest public-layout fixed
> B=2 sequence, and can an ordinary particle filter approximate the same
> decision?

The first stage uses metric-depth observations and a known geometric
likelihood. It isolates belief update and view planning from learned visual
recognition. RGB prediction is a later, separately frozen stage and cannot be
opened by this protocol. Simulator truth is available only to the generator,
physical checker, labels, and post-run audits; it is never a planner input.

The permitted claims are separated:

- `reference belief > best fixed` establishes a continuous active-observation
  mechanism under the registered sensor model;
- `ordinary PF near reference belief` establishes that a finite ordinary PF
  is an adequate approximation at the registered compute budget;
- `ordinary PF > MAP` attributes value to retaining multiple hypotheses;
- none of these is an RGB, OpenVLA, real-world, or learned-particle claim.

## 2. Public scene, hidden geometry, and route truth

Every public layout contains one fixed start, one target, three public route
splines (`left_route`, `center_route`, `right_route`), the official fixed
controller, camera calibration, static occluders, and robot geometry. Every
method may execute any route after either query; no proposer can permanently
restrict the action set.

The continuous latent state is

```text
z = (g, x_L, d_L, w_L, x_M, d_M, w_M, x_R, d_R, w_R)
```

where `g in [-1,1]` is a common correlation variable, `x_i` is obstacle
position along route `i`, `d_i` is its signed lateral displacement from that
route centerline, and `w_i` is its lateral half-width. The frozen candidate
sampler uses

```text
g             ~ Uniform(-1, 1)
x_i           ~ Uniform(0.04, 0.10) m
w_i           ~ Uniform(0.025, 0.055) m
d_i           = 0.030*g + epsilon_i m
epsilon_i     ~ Uniform(-0.070, 0.070) m
```

with a deterministic Latin-hypercube draw and registered seed. The common
term creates correlated route hazards without making routes identical. Public
layout geometry may move the route splines, occluders, and cameras but cannot
depend on the sampled `z` or any method outcome.

For route `i`, the physical checker executes the official controller and
computes signed minimum robot-obstacle distance over the full trajectory,
`c_i(z)`. The frozen clearance threshold is `tau_clear = 0.004 m`. A route is
feasible only when it has no robot/static/self collision and
`c_i(z) >= tau_clear`. The label is produced by this physical result, never by
a hand-authored bit or requested stratum.

Candidate generation targets three development strata using the measured
margin `m_i = c_i - tau_clear`:

- clearly safe: `m_i >= 0.008 m`;
- boundary: `|m_i| <= 0.003 m`;
- clearly blocked: physical collision or `m_i <= -0.004 m`.

Strata are sampling/audit targets, not route labels. A candidate whose actual
controller result misses its requested stratum is rejected before any policy
is run. Candidates are considered in increasing deterministic candidate index;
the first admissible set is retained. Every rejection and its physical reason
is logged. More than 100 candidate attempts for one retained world stops the
probe and requires a new protocol; ranges or thresholds cannot be changed
inside a run. Each route must appear in all three strata, with eight retained
probe worlds per stratum.

No cue cube, color code, text, state id, filename, or metadata may encode `z`
in the scene. Obstacle geometry and its natural occlusion are the only hidden
causes of image/depth variation.

## 3. Observation contract

The free initial view `V0` and paid cameras `q_branch`, `q_a`, `q_b`, `q_c`
are rendered from fixed public calibrations. Full RGB and metric depth are
archived. The non-learned Phase 6C planner receives only the registered depth
features from a purchased view plus public calibration; it cannot receive RGB,
segmentation ids, geom names, layout ids, latent parameters, clearances,
collisions, or route outcomes.

- `V0` is free and incomplete. It may initialize the prior but may not certify
  any route by itself in the probe.
- `q_branch` is a wide, occluded view. Its fixed depth rays localize the likely
  hazard region but are censored/quantized to `0.008 m`, coarser than the
  clearance decision boundary.
- `q_a`, `q_b`, and `q_c` are directional views centered on the left, center,
  and right route boundary respectively. Their fixed ray sets provide a
  route-local boundary measurement with registered Gaussian likelihood scale
  `sigma_specialist = 0.0015 m`; neighboring routes may be unobserved.

The deterministic feature extractor consists only of preregistered depth rays,
camera intrinsics/extrinsics, validity masks, and censoring flags. The known
likelihood is `y_v = h_v(z, public_layout) + epsilon_v`, where `h_v` is the
registered ray-cast prediction. Valid specialist residuals use the Gaussian
scale above; `q_branch` uses its registered quantized/censored likelihood.
Missing or occluded rays contribute no evidence and can never mean clear.

Every query records state before/after, RGB/depth hashes, extracted-feature
hash, camera calibration hash, cost, and remaining budget. State mutation is a
hard failure. A broker releases only the purchased feature; unpurchased views
remain inaccessible. Information gain and route-feasibility change are design
diagnostics, but the active planner selects views by expected decision value,
not global entropy alone.

Before the probe is admitted, an observation-design audit must show:

1. no forbidden metadata or unpurchased pixels enter any method;
2. every specialist has greater expected precision for its registered route
   than the other specialists on at least 80% of candidate worlds;
3. no single paid view maximizes expected decision value in every world;
4. after `q_branch`, at least two unresolved observation histories require
   different optimal second specialist views;
5. those histories are visible consequences of depth/occlusion geometry, not
   post hoc symbolic outcomes.

## 4. Frozen belief representations

All methods use the same prior, likelihood, purchased observations, route risk
definition, action rights, utility, and tie-breaks.

### 4.1 MAP planner

MAP retains exactly one geometry. It uses 16 deterministic multistart points,
at most 512 likelihood evaluations after each observation, and chooses the
lexicographically first maximizer. Its route-risk proxy integrates only the
registered local Gaussian sensor scale around that one estimate; it cannot
retain a second mode or borrow particles from another method.

### 4.2 Ordinary particle filter

The ordinary PF is frozen at `N=2048` particles. It samples the registered
prior, accumulates log likelihoods, subtracts the maximum log weight before
normalization, and fails on non-finite values or zero total weight. Effective
sample size is `ESS=1/sum(w_k^2)`. It performs deterministic systematic
resampling when `ESS < 1024`, then applies bounded Liu-West rejuvenation with
`a=0.98`. The resampling offset and rejuvenation noise come from the episode's
registered PF seed. No learned proposal, feature, update, or likelihood is
allowed.

Each PF transition logs particle-bank hash before/after, normalized weight
hash, ESS, resampling flag, rejuvenation hash, posterior route risks, selected
view/action, and wall-clock latency. A fresh Python process with the same
manifest and seed must reproduce these fields byte-for-byte.

### 4.3 Reference-belief oracle

The reference uses the same prior, purchased depth features, likelihood, B=2,
and action set. It approximates the continuous posterior with a deterministic
scrambled-Sobol bank of `2^18` samples. It cannot read true `z`, physical
clearance, collision labels, or unpurchased views. It is a high-compute
reference-belief upper bound, not a full-state oracle. QMC convergence is
checked against the nested `2^17` prefix; an action disagreement or route-risk
difference above `0.005` fails the corresponding probe case.

The learned particle proposal/update method is not implemented or run in
Phase 6C. Opening it requires the registered terminal branch in Section 10 and
a new Phase 6D protocol.

## 5. Common decision and execution rule

The primary budget is exactly two paid queries (`B=2`). Every paid camera costs
`0.05`; `V0` is free. Terminal utility is frozen as

```text
collision-free completion   +1.00
collision                   -4.00
safe stop                   -0.25
each paid query              -0.05
```

For every route, a belief method estimates
`r_i = P(c_i < 0.004 m or another registered collision occurs | history)`.
The execution threshold is `r_i <= 0.05`. This is a common decision threshold,
not a formal safety certificate. Calibration is reported separately.

At every decision point:

1. among routes with estimated risk at most 5%, choose the route with maximum
   expected terminal utility;
2. ties resolve shorter public route, then
   `left_route > center_route > right_route`;
3. if no route is eligible and budget remains, buy the legal view with maximum
   expected value of information minus `0.05` cost;
4. view ties resolve `q_branch > q_a > q_b > q_c`;
5. if the best query has non-positive expected value, or B=2 is exhausted with
   no eligible route, stop.

Execution uses the same complete official route controller for every method.
Only the selected route is executed in a policy trial, while counterfactual
route rollouts used for labels are stored separately and cannot enter the
planner. Not falsified does not mean safe; an unobserved route retains prior
risk.

## 6. Baselines and matched permissions

The probe and later formal comparison include:

1. `best_fixed_B2`: enumerate all 17 ordered no-replacement sequences of zero,
   one, or two paid views. Early execution/stop uses the common rule. For each
   public layout, select the sequence with best prior expected utility using
   only public geometry, the frozen prior, and likelihood; actual hidden draws
   and outcomes are forbidden.
2. `MAP_adaptive`: the one-scene representation and adaptive AVOI views.
3. `ordinary_PF_adaptive`: the frozen 2048-particle representation and
   adaptive AVOI views.
4. `reference_belief_adaptive`: the `2^18` QMC reference and adaptive AVOI
   views.

All four have the same B=2, paid cameras, view costs, candidate routes,
route-switch rights, execution threshold, controller, and stopping rule.
Information-gain-only and random-view rows may be added as diagnostics in the
formal addendum, but cannot replace the fixed baseline or receive different
permissions.

## 7. One-layout continuous probe

The first authorized implementation target is namespace
`continuous_active_vision_probe_v1`, generator seed `66421`, PF seed `66422`,
and bootstrap seed `66423`. It contains one public layout, 24 retained
continuous worlds, 72 physical counterfactual route trajectories, all four
paid depth/RGB views for offline audit, and all 17 fixed B=2 sequences. It is a
development mechanism probe, not a paper-scale estimate.

The probe hard gates are:

- **C0 physical truth:** all 72 official-controller trajectories agree with
  signed clearance/collision labels; every route has eight safe, eight
  boundary, and eight blocked retained worlds.
- **C1 observation integrity:** state-preserving queries, broker isolation,
  no cue/metadata leakage, registered depth-feature replay, and complementary
  specialist precision all pass.
- **C2 genuine branching:** after the same `q_branch` first query, at least two
  unresolved outcome histories choose different second specialist views; each
  branch contains at least four retained worlds and leads to a physically
  correct execute/stop decision.
- **C3 fixed impossibility:** no one of the 17 public-layout fixed sequences
  reproduces the reference adaptive action on all 24 worlds. Reference mean
  utility exceeds best fixed by at least `0.10`, with no higher collision rate.
- **C4 finite inference:** ordinary PF has finite normalized weights, records
  ESS/resampling/degeneration, and reproduces every posterior/action byte-for-
  byte in a fresh process.
- **C5 permissions:** every method has identical view/action/controller rights
  and no hidden truth reaches inference.
- **C6 reference stability:** nested-QMC convergence satisfies Section 4.3.

Any C0/C1/C4/C5/C6 failure is infrastructure failure. C2/C3 failure is a task
design failure: archive the root and amend the scene/observation design before
scaling. PF need not beat MAP in this probe; that comparison is reported but
is not an infrastructure gate.

## 8. Formal grouped study after probe PASS

No formal data is authorized until all probe gates pass and a prospective
split/power addendum is committed. That addendum must use namespace
`continuous_active_vision_formal_v1`, generator seed `66431`, PF seed `66432`,
and bootstrap seed `66433`, with no layout id or seed reused from 6A7 or 6B.

The planned pool is 30 train/development layout groups, 15 validation groups,
and at most 80 sealed-test groups. Each group owns all 24 latent worlds,
mirrors, views, counterfactual routes, and derived records. Train/development
may debug the generator and estimate layout-cluster variance. Validation may
only freeze likelihood scales, the MAP optimizer, PF hyperparameters, risk
calibration, and evaluator. Sealed test is opened once after every hash is
frozen.

The required sealed-test group count is computed before test opening as

```text
s_plan = max(SD of train-layout paired PF-minus-fixed utility, 0.20)
n_raw  = ((z_0.975 + z_0.80) * s_plan / 0.10)^2
n_test = clamp(round_up_to_10(ceil(n_raw)), 30, 80)
```

The maximum 80 test ids are generated and sealed prospectively; the first
`n_test` ids in manifest order are used. If the formula exceeds 80, the study
stops or downgrades power rather than borrowing Phase 6A7/6B data. The addendum
must freeze exact geometry ranges, rejection counts, camera/ray definitions,
likelihood/calibration artifacts, manifests, and hashes before formal runs.

## 9. Endpoints and inference

Layout group is the resampling and uncertainty unit. Worlds within a layout
are paired repeated conditions, not independent samples. The two-sided paired
layout-cluster bootstrap uses 10,000 replicates and seed `66433`.

Co-primary comparisons are:

1. reference-belief adaptive utility minus best-fixed B=2 utility (mechanism);
2. ordinary-PF adaptive utility minus best-fixed B=2 utility (finite method).

Holm correction controls familywise two-sided alpha at `0.05`. A positive
claim requires a corrected lower confidence bound above zero and a mean
utility difference of at least `0.10`. PF collision rate is a synchronous
constraint: its paired increase over best fixed must have a one-sided 95%
upper bound no greater than `0.02`.

Mandatory decomposed task metrics are collision-free completion, whole-episode
collision, execution-conditional collision, stop rate, route-switch success,
mean paid queries, mean utility, and latency. Belief metrics are posterior NLL
and Brier score for route feasibility, collision-risk calibration/reliability,
ESS distribution, resampling count, particle-degeneration count, QMC
convergence, and the paired reference-minus-PF utility/completion/collision
gap. Budget `B=0`, `B=1`, and `B=3` are diagnostic curves; B=2 remains primary.

## 10. Frozen terminal interpretations

- If reference belief clearly beats fixed and ordinary PF approaches reference
  within utility `0.05` with no material collision/calibration deficit,
  ordinary PF is sufficient. Learned particles do not open.
- If reference clearly beats fixed but ordinary PF is statistically
  indistinguishable from fixed and its reference utility gap is at least
  `0.10`, a learned proposal or update is a justified Phase 6D question. A new
  protocol must choose one learned component; Phase 6C is not backfilled.
- If reference is within utility `0.05` of best fixed or its corrected interval
  includes zero, the continuous task has not established adaptive-view value.
  Redesign scene/view geometry under a new protocol; do not train particles.
- If every method's collision rate exceeds `0.05`, or clear-route physical
  preflight fails, diagnose controller/clearance generation first. No belief
  conclusion is permitted.
- If PF beats MAP but does not beat fixed, report a representation effect
  without an active-policy endpoint. If PF matches reference but learned PF is
  later no better, the valid conclusion is that learned particle machinery is
  unnecessary.

## 11. Provenance and run order

Every retained or rejected world records protocol/manifest/code HEAD, public
layout hash, latent-label file hash, RGB/depth/feature hashes, camera hash,
controller hash, route-trajectory hash, physical clearance/collision result,
method config and seed, particle/QMC hashes, broker ledger, and checker
version. Model-visible observations and oracle labels live in separate files.
Fresh-process replay is part of the runner, not a post hoc summary field.

The only authorized order is:

1. commit and push this protocol;
2. implement pure latent/prior/likelihood/PF/planner contract tests;
3. implement one-layout scene, physical preflight, broker, runner, and checker;
4. generate the seed-66421 manifest and run the 24-world probe;
5. stop and report C0-C6; do not decide expansion inside the runner;
6. after probe PASS, commit a formal split/power addendum;
7. run train/development, then validation, freeze every artifact/hash, and run
   the sealed test once;
8. only after the terminal interpretation may a separate RGB Phase 6C-RGB or
   learned-particle Phase 6D protocol be proposed.

No Phase 6C code, data, or GPU run is authorized by merely drafting this file.
