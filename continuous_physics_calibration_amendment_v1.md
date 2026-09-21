# Phase 6C Physical Calibration Amendment v1

Date: 2026-09-21. Status: prospective calibration protocol. This amendment
does not modify or reinterpret the seed-66421 manifest, its 72 preflight
trajectories, or the archived C0 failure. It opens a new calibration namespace
only after this file is committed and pushed.

## Purpose and claim boundary

The seed-66421 root is permanently retained as a physical-consistency failure.
The only permitted conclusion from it is that the current continuous scene /
controller / analytic-clearance combination is not yet a valid benchmark. This
amendment calibrates that combination; it does not produce an active-observation
or particle result.

## Frozen physical contract

1. The left, center, and right routes use one symmetric controller-path
   template, mirrored about the center route. Empty-scene controller runs must
   be recorded for all three routes before any obstacle world is admitted.
2. Clearance is the signed minimum distance from the complete robot swept
   collision geometry to every physically present obstacle over the complete
   controller trajectory. An EEF point or fixed-radius point surrogate is not a
   valid final measurement. The same collision geometry is used for labels and
   preflight checks.
3. The route lane spacing, obstacle height, obstacle longitudinal half-size,
   and obstacle lateral half-width range are explicit constants in the scene
   manifest and cannot be changed during generation. The calibration report
   records these values and the controller version.
4. Absent obstacles contribute no route-obstacle clearance term, but static
   environment, robot self-collision, and table contacts remain hard failures.
5. A candidate world is eligible only after the official controller preflight
   has measured every route. It must contain one safe, one boundary, and one
   blocked route according to the frozen physical thresholds, with no
   undeclared collision class. No candidate may be selected using a policy
   outcome.
6. Candidates are considered in deterministic candidate-index order. The
   first candidate satisfying the complete three-stratum contract is admitted;
   every rejection records its reason and measured geometry. Failed worlds are
   never hand-replaced, relabeled, or repaired after execution.
7. No clearance threshold, robot envelope, route spacing, obstacle dimension,
   or stratum tolerance may be relaxed after a failed preflight. A failure that
   cannot be resolved within this contract requires a new amendment.

## Calibration fixture gates

Before a new probe is generated, the fixture must pass:

- empty-scene left/center/right controller paths are mirror-symmetric within
  the preregistered trajectory tolerance and all reach the target;
- each route independently produces physical safe, boundary, and blocked
  examples under the same controller;
- analytic clearance and full swept-geometry physical clearance agree within
  the preregistered calibration tolerance on the fixture set;
- neighboring route obstacles do not create undeclared contacts in a route's
  safe or boundary fixture;
- every route/fixture replay passes in an independent Python process with
  matching trajectory and physical-result hashes.

The calibration fixture is diagnostic only. It is not a population sample and
cannot be used as a Phase 6C method comparison.

## New probe gate

Only after all calibration gates pass may a new 24-world probe be generated.
It must use a new seed and namespace, enumerate candidates in the fixed order,
run static manifest audit, and run C0 physical preflight before any depth
broker, MAP, PF, QMC, AVOI, adaptive, or fixed-sequence code is opened.
The seed-66421 namespace and all of its artifacts remain immutable and are not
eligible for reuse.

If the second C0 attempt fails because the symmetric route/controller contract
cannot be made physically consistent, Phase 6C continuous geometry closes; the
next task must be a new scientific protocol rather than another particle-code
repair.
