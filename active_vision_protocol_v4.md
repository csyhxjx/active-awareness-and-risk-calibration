# Phase 5E: Route-Level Proposer Protocol

Status: preregistration draft for review. This protocol is independent of frozen Phase 5B and terminal Phase 5D results. It must be committed and pushed before any proposer training, smoke rollout, or GPU training.

## 1. Scope and proposer contract

The proposer consumes `V0 RGB + instruction + public route metadata` and emits three finite logits in the fixed order `[left_route, right_route, stop]`. `left_route` and `right_route` are valid candidates; `stop` is explicit abstention and never counts toward proposer coverage. The proposal is the deterministic argmax with ties resolved `left_route > right_route > stop`. Invalid, non-finite, or missing logits are `proposer_failure` and do not enter the coverage denominator.

Architecture variants are preregistered as: (A) instruction-only, (B) V0 plus instruction, and (C) V0 plus instruction plus public route geometry. Public geometry contains only the registered route metadata and no hidden-state or collision-derived value.

## 2. Observation budgets and tuple

At `B=0`, execute the proposer argmax route without buying a view. At `B=1`, verify only the argmax route; `clear` executes it, while `blocked` or `unobserved` stops. Looking at the alternate route is not allowed in B=1 and is reserved for an explicitly separate B=2 variant. Every trial records the five-tuple `proposer_output`, `mapped_candidate`, `selector_decision`, `executed_action`, `physical_outcome`; all failure classes are mechanically derived from it.

Stop samples may come only from public invalid instructions or public no-feasible-route controls defined before training. They may not be synthesized from hidden states, collisions, paid views, oracle margins, or old test data.

## 3. Data and splits

Training uses nominal train layouts only. Hidden states, collision outcomes, paid views, oracle margins, and all 5B/5D test data are prohibited. Layout groups, including all four hidden states, remain within one split. The new smoke namespace is 8 new train-only layout groups x 4 hidden states x 2 preferences = 64 trials. A future formal addendum will define new grouped train/validation/sealed-test namespaces and a new test seed; 5B's 40 test layouts remain permanently excluded.

## 4. Contract probe before GPU training

The pure contract probe must pass first: logits-to-choice behavior for ordinary, tied, all-zero, NaN, infinite, missing, and invalid logits; identical V0/instruction decision hashes across all four hidden states; query-before/after equality for qpos, qvel, ctrl, RNG, and simulation time; no paid view in proposer input; complete tuple serialization/reconstruction; and fresh-process byte-identical proposal/decision output. Only after this probe may training code be written or run.

## 5. Smoke gates

The 64-trial smoke does not train a selector and does not open validation/test. Gates are: proposer valid non-stop coverage >=90%; nominal preference accuracy >=95%; identical proposal hashes across paired hidden states; proposer-stop <=5% on nominal normal tasks; candidate-recall failure = 0; clear route completion and blocked-route frozen-oracle outcome; and complete mechanically reconstructable tuples. Any hard-gate failure stops expansion. Low coverage is a proposer/data-contract failure and cannot be repaired by training a selector.

## 6. Formal ladder and outcomes

The fixed ladder is: always-stop; proposer argmax B=0; best-fixed-view B=1; random-view B=1; candidate-aware geometric B=1; candidate-agnostic learned B=1; candidate-aware learned B=1; all-views and oracle as diagnostic upper bounds only. The mechanism endpoint is active geometric B=1 versus proposer-blind B=0 on collision-free completion without collision-rate worsening. The method endpoint is learned candidate-aware strictly better than geometric. If only the mechanism endpoint holds, the conclusion is that active vision works but simple geometry suffices. If neither holds, report no observed active-vision gain under this proposer and budget.

## 7. Power and freeze

The primary unit is the layout group and all bootstrap resampling is clustered by layout. Before formal data, compute `s_plan` from development-group variance using the preregistered conservative upper bound `s_plan = max(s_dev, 90th percentile of leave-one-group-out s_dev)`; use two-sided alpha 0.05, power 0.80, and the smallest detectable absolute paired completion effect `d_min=0.10`. Required groups are `ceil(((z_(1-alpha/2)+z_power)*s_plan/d_min)^2)` and are capped at 60 formal test groups. If the cap is exceeded, downgrade the claim or stop; do not borrow 5B/5D data. A smoke-only numeric addendum may fill in `s_dev` but may not change the formula, alpha, power, effect, or cap.

## 8. B/C boundary

OpenVLA chunk verification (B) and nominal-only LoRA OpenVLA (C) are independent future protocols with their own manifests, smoke gates, and new test namespaces. Their results cannot be backfilled into 5E or described as route-proposer results.
