# Phase 6B sealed RGB result

Hard gates: **PASS**. 
Scope: 12 layout groups, 96 scenes, 288 physical candidate trajectories.

## Detector layer

- ROI accuracy: `1.000000` over `384` paid-view crops.
- 10-bin ECE: `0.000000039`.
- The full hidden-state confusion retains the frozen structural aliases: `000 -> 110` and `111 -> 100`; controls are not in the main planner denominator.

## Planner layer

- RGB-vs-oracle adaptive decision error: `0.000000` (`0/72`).

## Task layer

| Method | Scope | n | Completion | Collision | Stop | Mean queries | Mean utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_fixed_b2_rgb | control | 24 | 0.500000 | 0.000000 | 0.500000 | 1.500000 | 0.300000 |
| best_fixed_b2_rgb | main | 72 | 0.666667 | 0.000000 | 0.333333 | 1.666667 | 0.500000 |
| oracle_roi_adaptive | control | 24 | 0.500000 | 0.500000 | 0.000000 | 2.000000 | -1.600000 |
| oracle_roi_adaptive | main | 72 | 1.000000 | 0.000000 | 0.000000 | 2.000000 | 0.900000 |
| rgb_detector_adaptive | control | 24 | 0.500000 | 0.500000 | 0.000000 | 2.000000 | -1.600000 |
| rgb_detector_adaptive | main | 72 | 1.000000 | 0.000000 | 0.000000 | 2.000000 | 0.900000 |

The result is limited to the controlled cue-ROI mechanism. It is not evidence of natural obstacle understanding.
