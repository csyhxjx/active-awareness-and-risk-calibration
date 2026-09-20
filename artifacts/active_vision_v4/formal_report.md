# Phase 5E formal result

The single sealed seed-57040 test contains 40 new layout groups, 160 scenes, 320 canonical trajectories, and 40 exact replays. Integrity, provenance, and freeze checks pass. The frozen route proposer produced 320/320 valid non-stop routes, 320/320 preference accuracy, zero proposer-stop, and zero proposer failure.

| Policy | Collision-free completion | Collision | Execution | Stop |
| --- | ---: | ---: | ---: | ---: |
| Always stop | 0.0% | 0.0% | 0.0% | 100.0% |
| Proposer blind B=0 | 50.0% | 50.0% | 100.0% | 0.0% |
| Best fixed B=1 | 36.875% | 11.875% | 48.75% | 51.25% |
| Random B=1 | 40.125% | 17.125% | 57.25% | 42.75% |
| Candidate-agnostic learned B=1 | 36.875% | 11.875% | 48.75% | 51.25% |
| Candidate-aware geometric B=1 | 50.0% | 25.0% | 75.0% | 25.0% |
| Candidate-aware learned B=1 | 50.0% | 0.0% | 50.0% | 50.0% |

Mechanism endpoint (geometric B=1 minus proposer-blind B=0): completion delta `0.000`, layout-bootstrap 95% CI `[0.000,0.000]`; collision delta `-0.250`, CI `[-0.250,-0.250]`. The registered strict completion-improvement condition fails.

Method endpoint (learned-aware minus geometric): completion delta `0.000`, 95% CI `[0.000,0.000]`; collision delta `-0.250`, CI `[-0.250,-0.250]`. The registered strict learned completion advantage also fails.

The permitted conclusion is: under this route proposer and B=1 budget, no active-vision collision-free-completion gain was observed. Both observation policies reduce collision, and learned-aware reduces it further, but neither improvement can be promoted to the preregistered completion endpoint.

Protocol deviation: the frozen evaluator reused the seven 5B executable baselines and did not emit the preregistered all-views and oracle diagnostic upper bounds. Those diagnostics are non-primary, but the omission is recorded rather than repaired after test inspection. No second evaluation or test corpus is authorized.
