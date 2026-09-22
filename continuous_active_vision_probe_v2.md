# Phase 6C Probe Admission Protocol v2

Date: 2026-09-22. Status: frozen prospective admission protocol; no runner,
manifest, probe world, MAP/PF/QMC execution, or method comparison is opened by
this document. Machine-readable configuration is
`configs/continuous_active_vision_probe_v2.json`.

## Frozen identity and support

Namespace is `continuous_active_vision_probe_v2`; generator/PF/QMC/bootstrap
seeds are `66701/66702/66703/66704`; retained sample size is 24 worlds, eight
per route stratum (safe, boundary, blocked), with 72 counterfactual route
trajectories. Each trial has exactly one active target obstacle. Its x center
is fixed at `0.070 m`, half-width is one of `0.025/0.040/0.055 m`, signed
lateral offset magnitude is within the calibrated `0.040..0.130 m` grid at
`0.002 m` increments, and all non-target obstacles are parked at `x=2.0 m`.
Longitudinal variation, multiple active obstacles, new controllers, and new
route splines are outside support and require a new calibration and protocol.

## Candidate generation and physical labels

Use deterministic Latin-hypercube candidates with the frozen generator seed,
considered by increasing candidate index. Retain the first candidate meeting
the measured full-geometry stratum; record every rejected candidate and reason.
Maximum attempts per retained world is 100. Safe is clearance `>=0.012 m`,
boundary is `abs(clearance-0.004)<=0.003 m`, blocked is physical collision or
clearance `<=0`, and gap candidates are rejected for quota purposes. Physical
labels use the exact 16-interval native-CCD kernel and cannot be replaced by
the analytic 60 mm envelope.

## Observation and methods

`v0` is free; paid views are `q_branch`, `q_a`, `q_b`, `q_c`; budget is exactly
B=2 and each paid query costs `0.05`. Only registered metric-depth rays,
validity/censor flags, and public camera calibration reach methods. RGB,
segmentation, latent state, clearance, collision, outcome, filenames and
metadata side channels are forbidden. Queries must preserve simulator state.

MAP, ordinary PF (`N=2048`, ESS threshold 1024, Liu-West `a=0.98`) and nested
QMC (`2^17` prefix and `2^18` reference) use the same exact full-geometry risk
predicate. The strongest fixed baseline exhausts all 17 ordered B=2 sequences
using public layout/prior/likelihood only. All methods have identical route,
view, controller, stop and query permissions. Learned particles and RGB are
closed.

## Metrics and gates

Primary metrics are collision-free completion, collision rate, stop rate,
route-switch success, paid queries, utility, latency, risk calibration and
QMC convergence. Utility is `+1/-4/-0.25/-0.05`; risk threshold is `0.05`.
The gate order is physical truth, observation isolation, fresh-process replay,
QMC stability, common-kernel cost, then method comparison. Any infrastructure
gate failure stops the run and supports no method claim. Missing strata stop
and require a new protocol. The exact-kernel batch/cache cost gate must pass
before any runner implementation or method evaluation. Approximation, if ever
proposed, requires a separate amendment and maximum error `<=1 mm` with zero
wrong-side decisions at `0.004 m` by route/width/sign.

## Terminal branches

Reference minus fixed must have a corrected lower confidence bound above zero
and mean gain at least `0.10` for an active-view claim. PF adequacy additionally
requires utility within `0.05` of reference and collision-rate paired upper
delta no greater than `0.02`. Otherwise report the registered negative branch,
archive the root, and do not train learned particles. If all methods exceed
`0.05` collision, diagnose physical generation first.

No probe execution is authorized until this protocol and config are committed
and the exact-kernel T4 cost gate is separately passed.
