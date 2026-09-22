# Phase 6C Physical Admission Amendment v2

Date: 2026-09-22. Status: prospective rule only. This amendment records the
physical-truth boundary after the `66421` failure and the fixed 72-case
diagnostic; it does not authorize a new probe.

## Three separate contracts

1. **Physical truth / final stratum.** A world-route pair is labeled safe,
   boundary, blocked, or gap only from the official controller trajectory,
   complete robot collision geometry, signed minimum distance, contact class,
   and completion. The physical result is the sole final stratum authority.
2. **Candidate search.** An analytic geometry formula may cheaply prescreen or
   rank candidates. It cannot assign a final stratum, replace controller
   preflight, or cause a failed world to be admitted.
3. **Belief risk.** MAP/PF/QMC may compute route risk only under a common
   registered geometry model. That model must be compared against physical
   controller/swept-geometry outcomes on development fixtures before a risk
   threshold is interpreted. Physical admission does not validate a biased
   analytic risk model.

## Required admission procedure

- Enumerate candidate worlds in deterministic candidate-index order under a
  new seed and namespace. The conditional scan distribution and accepted
  stratum proportions must be frozen before generation; retained stress cases
  are not reinterpreted as draws from the original prior.
- Use analytic geometry only for prescreening. Run the complete official
  controller and swept-geometry measurement for every route before admission.
- Admit the first candidate set whose measured route labels contain the
  required safe, boundary, and blocked strata, with no undeclared static,
  self, or neighboring-route contact. Record every rejection and measured
  reason. Never hand-relabel, lower a threshold, or select using policy
  outcomes.
- If the public route/controller template cannot produce all three strata,
  stop and write a new scene amendment. Do not expand the grid post hoc.

## Risk-model gate

Before MAP/PF/QMC policy use, compare the common geometry model against the
physical swept-geometry result on the calibration fixture. If the fast model
misses the frozen boundary tolerance, use the complete geometry computation in
the risk evaluation even if 2048-particle throughput is lower. Runtime is not
permission to substitute a known-biased clearance formula.

## Current decision

The fixed diagnostic found no physical boundary cases: left `20 blocked/4
safe`, center `22 blocked/2 safe`, right `21 blocked/3 safe`. Therefore this
amendment freezes the distinction and the stop rule, but it does not freeze an
admissible candidate distribution or open `continuous_active_vision_probe_v2`.
