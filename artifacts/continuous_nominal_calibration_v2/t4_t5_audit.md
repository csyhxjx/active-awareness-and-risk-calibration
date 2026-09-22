# T4/T5 Audit Record

Date: 2026-09-22

This is a design and cost audit only. It does not authorize a probe, MAP, PF,
QMC, AVOI, or method-comparison run.

| requirement | status | evidence |
|---|---|---|
| One common physical risk definition for MAP/PF/QMC | FAIL | `continuous_risk_kernel_contract_v1.md:8-19`; exact MuJoCo 3.13 native CCD is specified, but `guard/active_vision/continuous_belief.py:66-74,164-170` still uses the 60 mm envelope approximation. |
| Full supported geometry is explicitly bounded | PASS | `continuous_risk_kernel_contract_v1.md:21-28`; `continuous_active_vision_probe_v2.md:8-18`; config `latent_support`. |
| 16-interval exact-kernel cost measured | PASS (measurement) | `continuous_risk_kernel_contract_v1.md:30-35`; 9 representatives, 2.431270617991686 s, 359348 KiB RSS, stdout SHA-256 `554aeff65be3e15ed2c9935c372aaf8d9e306b4845abdcf30f62a687fa3da5ca`. |
| Full PF/QMC cost is currently acceptable | FAIL | `continuous_risk_kernel_contract_v1.md:36-48`; conservative estimates are about 27.7 min for 6144 PF route evaluations and 59.3 h for 786432 QMC route evaluations. |
| Exact result-preserving batch/cache gate | MISSING | Required on 9 representatives and all 292 near-boundary cases before any runner; no implementation or result exists. |
| Approximation error/decision-side gate | MISSING | No approximation is introduced. Any future approximation must be separately amended and meet <=1 mm absolute error and zero wrong-side decisions. |
| Probe namespace, seeds, sample size and quotas frozen | PASS | `continuous_active_vision_probe_v2.md:8-18`; `configs/continuous_active_vision_probe_v2.json`. |
| Candidate rejection, gap handling and physical labels frozen | PASS | `continuous_active_vision_probe_v2.md:20-29`; config `strata`. |
| Observation budget, permissions and common methods frozen | PASS | `continuous_active_vision_probe_v2.md:31-44`; config `observation` and `methods`. |
| Metrics, gates and stopping branches frozen | PASS | `continuous_active_vision_probe_v2.md:46-69`; config `metrics`, `gates`, `stop_rules`. |
| New probe execution authorized | FAIL | T5 is prospective only; the protocol explicitly requires the T4 exact batch/cache gate first. |

## Scope boundary

The calibration supports only fixed obstacle x=`0.070 m`, one active target,
other obstacles at x=`2.0 m`, widths `0.025/0.040/0.055 m`, and the three
registered nominal routes. Longitudinal variation, multiple active obstacles,
new controllers, and new route splines require a new calibration root and a
new protocol. Existing `export_v2c`, `grid_v2c`, and v1 roots are untouched.

## Blocking next gate

Implement an exact result-preserving batch/cache backend. It must reproduce
the 16-interval output bytes and hashes for the 9 representative cases and
the 292 near-boundary cases, then repeat the cost audit. Until that evidence
passes, MAP/PF/QMC and the probe remain `NOT_RUN`.
