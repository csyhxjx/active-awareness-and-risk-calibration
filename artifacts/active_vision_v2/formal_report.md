# Phase 5B frozen formal comparison

The single sealed test contains 40 independent layout groups, 160 hidden-state scenes, 320 crossed-preference decision trials, 320 canonical candidate trajectories, and 40 exact replay audits. Integrity, provenance, finite-value, file-hash, paired-isolation, and freeze checks all pass. Test collection HEAD is `74fc801`.

## Matched-budget results

| Policy | Collision-free completion | Collision | Execute | Collision / execute | Erroneous rejection / successful candidate | Stop | Correct double-blocked stop | Unresolved | Mean queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 always stop | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0 |
| B1 nominal execute | 50.0% | 50.0% | 100.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 |
| B2 conservative stop | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0 |
| B3 fixed `v_left` | 35.6% | 10.6% | 46.2% | 23.0% | 28.7% | 53.8% | 78.8% | 28.7% | 1.0 |
| B4 random mean | 39.6% | 15.5% | 55.1% | 28.0% | 20.8% | 44.9% | 70.2% | 21.6% | 1.0 |
| B5 candidate-agnostic learned | 43.8% | 18.8% | 62.5% | 30.0% | 12.5% | 37.5% | 62.5% | 12.5% | 1.0 |
| B6 candidate-aware geometric | 50.0% | 25.0% | 75.0% | 33.3% | 0.0% | 25.0% | 50.0% | 0.0% | 1.0 |
| M candidate-aware learned | 50.0% | 0.0% | 50.0% | 0.0% | 0.0% | 50.0% | 100.0% | 0.0% | 1.0 |

B4 is the pre-specified mean of seeds 7001-7005. Cached offline retrieval latency is defined as zero; query count is the observation-cost endpoint. A stop never counts as completion, so B0/B2 cannot win through zero collisions.

## Confirmatory fork

M and B6 both complete 160/320 trials. The paired layout-cluster completion difference is `0.000`, with 10,000-sample two-sided 95% bootstrap CI `[0.000, 0.000]`. The registered requirement that the lower bound be greater than zero therefore fails.

M has 0/320 collisions and B6 has 80/320; the paired collision difference is `-0.250`, with one-sided 95% upper bound `-0.250`, so the synchronized collision constraint passes. Passing that constraint does not substitute for a strictly positive completion effect.

The pre-registered second branch applies: no learned-selector method advantage is claimed. Action-related view selection is mechanistically useful in this benchmark, but the primary completion value is realizable by the simple public-geometry rule. M's lower collision rate is reported as an important secondary result and does not alter the frozen fork.

This is a fixed-controller controlled-occlusion result, not an OpenVLA improvement and not a whole-task safety claim. Phase 5D must separately report proposer failure and candidate recall.
