# Phase 6C Nominal-Trajectory Physical Model Amendment v1

Date: 2026-09-22. Status: prospective calibration-only. This additive model
supersedes the obstacle-responsive controller *for a future benchmark only*.
The immutable `66421` C0 failure, calibration `66530`, and fixed 72-case
dynamic-controller scan retain their original meanings. This document does
not authorize a 24-world probe or any belief-policy run.

## Fixed physical model

- Record one joint-position trajectory per left/center/right route using the
  existing public OSC_POSE controller, with every obstacle parked outside the
  reachable workspace. Include the reset pose, each control-step qpos, the
  controller/robot/scene hash, completion and empty-scene static/self contacts.
- Treat those qpos sequences as the route definitions. In a candidate world,
  replay them kinematically in MuJoCo with no controller step or physics
  integration; obstacle contact must not change the subsequent qpos. This is a
  prescribed open-loop motion, not a claim that a torque-controlled arm would
  physically continue through an obstacle.
- At each pair of adjacent poses evaluate the full MuJoCo robot collision
  geoms at the endpoints and at seven equally spaced interior **joint-space**
  interpolants (`8` intervals per controller step). Set qvel to zero, update
  kinematics/contact with `forward`, and record the minimum signed robot-to-
  obstacle contact distance, geom pair, sample index, first penetration and
  static/self contacts. MuJoCo `geom_margin=geom_gap=0.25 m` enables proximity
  reporting without contact force; no contact inside this radius means only
  `clearance >= 0.25 m` (right-censored), not an invented exact distance.
- The same measured swept-geometry kernel must define physical strata and the
  future common MAP/PF/QMC risk predicate. The old `abs(d)-w-0.060` and EEF
  point-plus-envelope formulas are **prescreen diagnostics only**. Candidate
  selection cannot relabel a geometry result.
- Feasible means nominal route reaches the common goal, no route/static/self
  collision and minimum full-geometry distance >= `0.004 m`. Existing frozen
  strata remain: safe margin >= `0.008 m`, boundary |margin| <= `0.003 m`,
  blocked physical collision or margin <= `-0.004 m`; other outcomes are gaps.
  Static/self and neighboring-obstacle contact are separate infrastructure
  failures. `collision` here means kinematic geometric intersection, not
  reactive-control failure.

## Prospective calibration-only grid and gates

Under namespace `continuous_nominal_calibration_v1` (not a probe), scan each
route, each obstacle lateral half-width `{0.025,0.040,0.055} m`, both signed
offsets at magnitudes `0.040..0.130 m` in inclusive `0.002 m` steps, obstacle
longitudinal center `x=0.070 m`, and park non-target obstacles at `x=2.0 m`.
This is `3*3*2*46 = 828` deliberately conditioned cases, not a prior draw.
Obstacle center height `0.960 m`, longitudinal half-size `0.035 m`, centerline
spacing `0.270 m`, and the existing robot/waypoint template remain fixed.
Save the full qpos source, per-step minimum geom pair and distance, first
contact, trajectory hash, geom-model hash, and every case's final stratum.

Before any new probe, every route must show at least one safe, boundary, and
blocked case in this frozen grid; the empty recorded route must reach the goal
without static/self contacts. A representative case per route and stratum
must match byte-for-byte in a fresh Python process. Selected near-boundary
cases must be remeasured with 16 intervals per controller step and satisfy
`abs(c_8-c_16) <= 0.0005 m` and the same collision/feasibility decision. If
not, the 8-interval kernel cannot label worlds or score risk. No changing the
grid, dimensions, threshold or nearest-pair logic after seeing results.

Any *faster* approximate model proposed as the shared MAP/PF/QMC risk kernel
must, on all cases with `|c_16-0.004| <= 0.010 m`, satisfy maximum absolute
clearance error <= `0.001 m` and zero wrong-side decisions at `0.004 m`.
Evaluate by route, width and offset sign; averages alone cannot pass. If it
fails, use the full 16-interval MuJoCo kernel as the common risk model and
report computation cost; the 2048-particle runtime target is not grounds to
reuse a biased approximation. The geometry error test must be measured, not
inferred from admission. Model-visible inputs remain purchased observations;
only candidate particle geometry may enter the shared risk kernel.

## Stop and distribution locks

This scan cannot generate a 24-world manifest. Missing boundary strata,
interpolation instability, or replay failure stops this calibration branch
and requires a *new* prospective scene/model amendment. Passing calibration
only permits a separately frozen new-seed, new-namespace admission protocol.
It does not revive `66421`, authorize MAP/PF/QMC rollouts, or make conditional
stress samples representative of the registered prior. Any formal experiment
must freeze retained stratum proportions and distinguish its conditional
distribution from the belief prior before sampling.
