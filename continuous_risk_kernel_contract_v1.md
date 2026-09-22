# Phase 6C T4 Risk-Kernel Contract v1

Date: 2026-09-22. Status: design and cost audit only. No probe, method
comparison, or new latent-world run is authorized by this document.

## Common definition

The only physical risk predicate is the exact MuJoCo 3.13.0 native-CCD
kernel: replay the registered nominal joint trajectory with 16 joint-space
intervals per control step, call `mj_forward`, and minimize
`mj_geomDistance(..., distmax=0.25)` over every robot collision geometry and
the one active target obstacle geometry. A value at `0.25 m` is right-censored
and means `distance >= 0.25 m`, never an exact distance. Route feasibility is
`reached and no static/self/route collision and clearance >= 0.004 m`.

MAP, ordinary PF (2048 particles), and the nested QMC reference (`2^17` /
`2^18`) MUST call this same predicate for route risk. The existing
`continuous_belief.route_clearances()` 60 mm envelope is a diagnostic
approximation only and is not a passing common-risk implementation.

## Supported geometry

The calibration evidence supports only: longitudinal obstacle center
`x=0.070 m`, one active target obstacle, other obstacles parked at `x=2.0 m`,
three half-widths `{0.025, 0.040, 0.055} m`, and the registered left/center/right
nominal trajectories. It does not support longitudinal variation, multiple
simultaneously active obstacles, changed robot/controller geometry, or a new
route spline. Any such extension requires a new calibration root and protocol.

## Cost audit

The fixed representative fixture (9 cases, 16 intervals) took `2.431270617991686`
seconds in one fresh worker process, with peak child RSS `359348 KiB`, 9/9
JSON results, and stdout SHA-256
`554aeff65be3e15ed2c9935c372aaf8d9e306b4845abdcf30f62a687fa3da5ca`.
This includes process startup. Conservative linear planning bounds are:

| workload | geometry evaluations | conservative wall estimate |
|---|---:|---:|
| one 9-case audit batch | 9 | 2.43 s measured |
| PF route risks, 2048 states x 3 routes | 6144 | about 27.7 min |
| QMC route risks, 262144 states x 3 routes | 786432 | about 59.3 h |

These are single-process, unbatched upper planning estimates, not method
results. They show that the exact kernel is feasible for calibration and small
audits but not yet computationally acceptable for the registered QMC query
tree. T4 compute-feasibility status is therefore **FAIL pending an exact
result-preserving batch/cache implementation**. No approximate model is
introduced. Any approximation must be separately registered and pass maximum
absolute error `<=0.001 m` and zero wrong-side decisions at `0.004 m`, broken
down by route, width, and offset sign.

## Required implementation gate

Before any probe runner is implemented, a batch/cache backend must reproduce
the exact 16-interval outputs and hashes on the 9 representatives and all 292
near-boundary cases, then repeat the cost audit. Until that gate passes,
MAP/PF/QMC and AVOI are NOT_RUN.
