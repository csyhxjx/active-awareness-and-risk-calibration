# Phase 6C Continuous Geometry Diagnostic Plan v1

Date: 2026-09-22. Status: prospective development diagnostic. This plan does
not modify the immutable seed-66421 root and cannot produce a method result.

## Purpose

Separate three contracts:

- physical stratum truth comes only from the official controller trajectory,
  full robot collision geometry, contacts, completion, and minimum distance;
- an analytic approximation may rank or prescreen candidates but cannot label
  or admit them;
- a future MAP/PF/QMC risk model must use one shared geometry model whose error
  is validated against physical trajectories. Physical candidate admission by
  itself does not validate belief risk.

## Frozen diagnostic grid

The diagnostic uses only the existing public layout and controller. For each
of `left_route`, `center_route`, and `right_route`, it scans:

```text
obstacle x                     0.070 m
lateral half-width             {0.025, 0.040, 0.055} m
signed route-relative offset   +/-{0.065, 0.075, 0.085, 0.095} m
```

This gives 72 route-specific cases. Non-target route obstacles are moved
outside the reachable workspace. The grid is fixed before execution and is
not extended to manufacture a missing stratum. It is a conditional development
scan, not a draw from the registered prior and not a population estimate.

## Geometry measurement

MuJoCo collision geoms are the measurement surface. Robot and obstacle geoms
receive equal proximity `margin` and `gap`, which asks MuJoCo to emit distance
contacts without applying force before penetration. A calibration check must
show the controller trajectory hash is unchanged with proximity reporting on.

Every step stores full qpos, EEF position, waypoint phase, nearest robot and
obstacle geom names, obstacle route, signed MuJoCo distance, analytic EEF-point
distance, contact state, and controller completion state. The first penetrating
contact and global minimum pair are recorded. Robot components are grouped as
`gripper`, `wrist`, or `arm` from preregistered geom-name rules.

Each obstacle run is compared with the matching empty-scene controller trace.
This diagnoses whether an empty sweep is an adequate fast model; it does not
make that sweep physical truth.

## Development decisions

- If every route contains at least one physical safe, boundary, and blocked
  case, a later admission amendment may use physical preflight as the final
  label and deterministic first-admissible selection. Analytic clearance stays
  a prescreen only.
- If a stratum is absent, the public path or obstacle template requires a new
  amendment; the grid is not silently extended.
- If no fast geometry model meets the registered boundary tolerance, the next
  probe must use the more expensive MuJoCo swept-geometry computation. The
  2048-particle runtime target is subordinate to correct risk geometry.
- No new 24-world probe, observation run, MAP/PF/QMC comparison, or particle
  change is authorized by this diagnostic alone.
