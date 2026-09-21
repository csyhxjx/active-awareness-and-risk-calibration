# Phase 6C seed-66421 C0 failure attribution

This is a read-only analysis of the immutable preflight archive. It does not modify the manifest, trajectories, labels, or failure report.

Scope: `24` worlds, `72` routes; mismatches: `17`.

## Margin discrepancy

`delta = physical controller margin - analytic margin`.

Overall mean delta `-0.004917 m`, SD `0.008683 m`, range `[-0.034977, 0.001902] m`.

| group | n | mean delta (m) | sd (m) | min (m) | max (m) |
|---|---:|---:|---:|---:|---:|
| by_route:center_route | 24 | 0.000165 | 0.000266 | -0.000000 | 0.000996 |
| by_route:left_route | 24 | -0.008412 | 0.009689 | -0.026477 | 0.001902 |
| by_route:right_route | 24 | -0.006505 | 0.009575 | -0.034977 | 0.001902 |
| by_expected_stratum:blocked | 24 | -0.004494 | 0.006408 | -0.017783 | 0.000996 |
| by_expected_stratum:boundary | 24 | -0.003946 | 0.008131 | -0.026477 | 0.001902 |
| by_expected_stratum:safe | 24 | -0.006312 | 0.010767 | -0.034977 | 0.001902 |

## Contact and mismatch summary

Collision/contact counts: `{"first_contact:route_obstacle": 31, "route_obstacle": 31}`.

Mismatches by route: `{"center_route": 0, "left_route": 8, "right_route": 9}`.
Mismatches by expected stratum: `{"blocked": 0, "boundary": 9, "safe": 8}`.

## Interpretation boundary

- The archived preflight summaries contain trajectory hashes and first contacts, not stepwise records; route-convergence attribution is therefore limited to route-level and contact-level evidence.
- No adaptive, fixed, MAP, PF, QMC, or observation result is included or inferred.
- A positive analytic/physical discrepancy is diagnostic evidence about the approximation and/or controller sweep, not a relabeling permission.
