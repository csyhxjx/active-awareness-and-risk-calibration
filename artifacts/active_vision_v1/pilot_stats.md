# Phase 5A active-vision development pilot

Canonical scope: 12 layouts x 4 hidden states = 48 scenes and 96 candidate trajectories. The 12 deterministic replay audits are verification reruns, not added samples. Run HEAD: `6d42eb4224b43176242745e3c441c70d51fb7668`.

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| P0 | PASS | query_invariant=True, replays_exact=True |
| P1 | PASS | passing_layouts=12, required=10 |
| P2 | PASS | passing_layouts=12, required=12 |
| P3 | PASS | passing_layouts=12, required=10 |
| P4 | PASS | left_unique_layouts=6, right_unique_layouts=6, required_each=4 |
| P5 | PASS | paired_examples=24, required=6 |

All pre-registered P0-P5 gates pass. `V0` is pixel-identical within every four-state layout group. The RGB-only color/shape audit was committed before collection and separates the relevant route in 12/12 layouts. It is an automated single-auditor development check, not an inter-rater study.

## Fixed-controller metrics

The table crosses both public route preferences over all 48 scenes (96 decision trials). Physical candidate labels are reused; these are not 96 additional rollouts. A stop is never counted as completion.

| Policy | Collision-free completion | Collision | Execution coverage | False rejection | Stop | Feasible stop | Mean views |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always stop (B=0) | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 75.0% | 0.00 |
| Nominal preferred route (B=0) | 50.0% | 50.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.00 |
| Fixed V_left (B=1) | 25.0% | 0.0% | 25.0% | 0.0% | 75.0% | 50.0% | 1.00 |
| Fixed V_right (B=1) | 25.0% | 0.0% | 25.0% | 0.0% | 75.0% | 50.0% | 1.00 |
| Fixed V_high (B=1) | 25.0% | 0.0% | 25.0% | 0.0% | 75.0% | 50.0% | 1.00 |
| Random view (B=1, 5 seeds) | 25.6% | 0.0% | 25.6% | 0.0% | 74.4% | 49.4% | 1.00 |
| Candidate-aware side view (B=1) | 50.0% | 0.0% | 50.0% | 0.0% | 50.0% | 25.0% | 1.00 |
| Candidate-aware fallback (B<=2) | 75.0% | 0.0% | 75.0% | 0.0% | 25.0% | 0.0% | 1.50 |

The candidate-aware B=1 rule removes collisions but completes only when the preferred candidate is clear; its 25% feasible-stop rate is the cost of refusing an unobserved fallback. B<=2 closes that gap: it completes every scene with at least one clear route (75% overall) and correctly stops in double-blocked scenes. Always-stop has zero collision but also zero completion, so it is not a winning policy.

These fixed-controller numbers establish the observation/decision mechanism only. They do not establish a learned selector advantage and are not an OpenVLA improvement. Same-budget learned, fixed, random, and geometric comparison requires a separately frozen Phase 5B contract. Frozen OpenVLA proposals enter only after that mechanism comparison; proposer failures and candidate recall must then be reported separately.

## Evidence

- `view_audit.png`: all four hidden states for one layout; free images are identical while paid evidence is candidate-dependent.
- `paired_decisions.png`: same initial input and candidate, followed by clear/blocked observations that justify different correct decisions.
- `route_outcomes.png`: all 12 layouts and eight state-candidate outcomes; no failed layouts were removed.

## Limitations

This development benchmark deliberately uses color-coded obstacles and candidate-aligned side views. It validates H1 information value and closes the fixed-controller decision chain, but it does not by itself validate H2 learned view selection, H3 generalization, moving-camera cost, or safety of natural OpenVLA behavior. Development layouts and derivatives remain excluded from future validation and test sets.
