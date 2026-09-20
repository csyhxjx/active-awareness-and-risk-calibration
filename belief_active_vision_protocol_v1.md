# Phase 6: Belief-Space Active Observation and Action Revision

Date: 2026-09-20. Status: prospective protocol. No Phase 6 simulator, data, model training, smoke, validation, or test exists at this commit. Phase 5F remains frozen but unexecuted and cannot be reused as Phase 6 evidence.

## 1. Research decomposition

Phase 6 asks whether maintaining multiple plausible hidden scenes improves cost-sensitive observation and action revision. It is deliberately staged:

1. **6A exact discrete contract:** use the existing four hidden states `S={00,10,01,11}` to verify belief updates and planning exactly. This is a logic baseline, not a particle-learning experiment.
2. **6B continuous task:** introduce continuous hidden obstacle geometry and observations whose value cannot be represented by four categories. Compare simple representations under identical observations, action rights, costs, and budgets.
3. **6C learned particles:** add learned proposal, matching, or weight-update components only if ordinary particle filtering shows a benefit over a single-scene estimate in 6B.

No stage may use an old Phase 5 sealed test, tune a frozen selector, or reinterpret Phase 5E as a belief result.

## 2. Phase 6A exact four-state belief

The belief is the complete vector `b=(p00,p10,p01,p11)`, finite, nonnegative, and summing to one. The default prior is uniform. A query result `o` updates it by Bayes' rule:

`b'(s) = P(o|s,q)b(s) / sum_t P(o|t,q)b(t)`.

The likelihood table is explicit input to the updater and must be normalized over observations for every `(state,query)`. A zero-evidence observation is a hard contract failure, not silently converted to a uniform posterior. Updates occur only after a purchased observation and must be order-replayable.

Available terminal actions are `left_route`, `right_route`, and `stop`. State truth defines left blocked by the first bit and right blocked by the second. Terminal utility is `+1` for collision-free completion, `-C_collision` for collision, and `-C_stop` for stop, with pure-contract defaults `C_collision=4`, `C_stop=0.25`. Query utility subtracts its registered cost. The exact planner enumerates all remaining legal query/outcome branches and terminal actions; ties resolve `left_route > right_route > stop > lower-cost query > query id`. It returns both action and expected value.

6A comparisons are exact belief versus its MAP point-estimate ablation. MAP ties resolve `00 > 10 > 01 > 11`; after selecting one state, the point estimator plans as if its probability were one. Both methods receive the same prior, likelihoods, query set, costs, budget, and action-switch permission. Required unit cases include: normalized posterior, contradictory observations, zero evidence, uninformative query, complementary queries, unequal costs, query-order replay, MAP ties, and a paired case where exact belief queries then switches route while MAP executes the wrong route.

6A success only establishes correctness of belief and decision code. It cannot support a continuous-task, particle-filter, vision, or learned-method claim.

## 3. Phase 6B continuous hidden-geometry task

Each episode samples two opaque obstacles with continuous centers, widths, and depths. The hidden vector is

`z=(x1,y1,w1,d1,x2,y2,w2,d2)`

from prospectively frozen bounded continuous distributions. Public geometry contains the start, target, three or more route splines, camera calibration, static occluders, and route-control limits. The initial `V0` must be observationally aliased across a registered set of materially different `z` values, but exact pixel identity is not required once continuous nuisance variables are introduced; leakage is instead tested by a preregistered hidden-state prediction audit.

At least three non-equivalent candidate routes are required so action revision is not a deterministic two-route demonstration. A route's collision outcome depends continuously on obstacle position and extent, including near-boundary cases. All methods may, after every observation, execute any public route, buy another legal observation within budget, or stop. No method is restricted to the original candidate once another is shown safer.

Observations are pre-action RGB from at least three cameras with complementary fields of view. Each camera has a different fixed positive cost, and at least one obstacle region must be partially visible from two views with different noise/occlusion profiles. No single view may dominate every layout. Budget is a scalar cost cap rather than a view count. All methods share the same camera set, likelihood/evidence interface, route set, controller, stopping rule, cost cap, and wall-clock accounting.

The new task must pass development gates before formal comparison: continuous collision boundary verified by rollouts; V0 leakage below a frozen prediction threshold; complementary-view information gain; no universally optimal camera; at least three layouts where the optimal route changes after observation; state-preserving queries; deterministic replay; and matched action/query permissions across methods. Failed layouts are retained and reported.

## 4. Representation ladder

The first continuous comparison freezes three non-learned representations:

1. **Single-scene estimate:** maintain one MAP geometry, update it after each observation, and plan as if it were true.
2. **Exact discrete belief diagnostic:** only in 6A and in optional tiny discretized 6B sanity fixtures; it is not presented as scalable continuous inference.
3. **Ordinary particle filter:** fixed prior sampler, particle count ladder preregistered before data, explicit likelihood, systematic resampling at a fixed effective-sample-size threshold, and optional deterministic rejuvenation kernel shared across methods where applicable.

The primary representation question is ordinary PF versus single-scene MAP at matched observation/action budget: does retaining multiple hypotheses improve expected decision utility, collision-free completion, or risk-cost tradeoff? Particle count, resampling threshold, likelihood bandwidth, and compute cap must be frozen in a later implementation addendum before formal data.

Dense quadrature or simulator truth may be a small diagnostic upper bound but never an inference input. Every method logs posterior summaries, chosen query/action, accumulated observation cost, simulator calls, inference latency, and terminal physical outcome.

## 5. Learned particle network gate

Learned particles are forbidden until the ordinary PF implementation passes its contract and a fresh development comparison shows a preregistered positive multi-hypothesis signal over MAP. If that gate passes, a new addendum must choose exactly one learned component first: proposal distribution, observation-matching likelihood, or weight update. It may not learn all three simultaneously in the first comparison.

The learned method uses the same particle count, query/action permissions, observation budget, and terminal planner as ordinary PF. Training data, architecture, loss, seeds, calibration, checkpoint choice, and compute cap are frozen before training. Report GPU time, latency, memory, and simulator calls; decision gains without compute accounting are incomplete.

The learned-method claim requires improvement over ordinary PF, not merely over MAP. If PF beats MAP but learned PF does not beat PF, the valid conclusion is that maintaining multiple hypotheses matters but learned particle machinery is unnecessary. If PF does not beat MAP, learned particle training does not open.

## 6. Endpoints and fairness

The future formal protocol must define a single primary expected utility combining collision-free completion, collision loss, stop loss, and observation cost, with all coefficients frozen before formal collection. Mandatory decomposed outcomes are collision-free completion, collision rate, stop rate, route-switch rate, observation cost, posterior calibration, effective sample size, and inference latency.

Every comparison is paired by base layout and latent draw. Layout families are the resampling unit. All methods receive identical raw observations only when purchased, identical route options, identical action-switch rights, and identical total cost budget. A learned method may not buy extra views, use oracle geometry, or amortize unreported simulator calls.

## 7. Execution order

1. Commit and push this protocol before Phase 6 code.
2. Implement pure 6A belief/update/planner functions and deterministic tests only.
3. Freeze a continuous-task development addendum with distributions, routes, cameras, costs, budgets, gates, and seeds.
4. Build one-layout probes, then a small development suite; stop if continuous ambiguity, complementary information, or action switching is absent.
5. Freeze and compare MAP versus ordinary PF; exact belief remains a diagnostic.
6. Open learned particles only after the multi-hypothesis gate passes.
7. Freeze new grouped formal splits and one sealed test only after representation, compute, and statistical contracts are complete.
