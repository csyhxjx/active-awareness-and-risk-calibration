# Phase 6C physical calibration fixture

This is a diagnostic calibration run under `continuous_physics_calibration_amendment_v1`.
It does not modify or relabel the immutable seed-66421 manifest, trajectories, or C0 report,
and it does not generate a new formal probe.

## Gates

| gate | result |
|---|---|
| empty-scene symmetric controller paths | PASS |
| three physical strata on each route | FAIL |
| analytic versus physical clearance within 0.003 m | FAIL |
| no undeclared neighbor/static/self contact | PASS |
| fresh-process route replay | PASS |

The empty scene reaches the target on all three routes. The left/right mirror
error is `5.524288e-7 m`. All replay hashes match across independent Python
processes. No static-environment, self-collision, or undeclared neighboring
obstacle contact was observed in the selected fixture rows.

The calibration does not pass because the current analytic clearance remains
the old EEF-point/fixed-envelope approximation. Its discrepancy from the
official controller result exceeds the frozen `0.003 m` tolerance, and the
archived candidate states do not produce all three physical strata on the
left/right routes under the current controller. This is a calibration and
scene/controller failure, not a policy result.

No new 24-world manifest, C0 rerun, depth observation, MAP, PF, QMC, AVOI,
adaptive, or fixed-sequence analysis is authorized from this fixture.
