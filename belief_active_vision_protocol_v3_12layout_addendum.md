# Phase 6A5: Twelve-Layout Infrastructure-Corrected Amendment

Date: 2026-09-21. Status: prospective. The failed Phase 6A4 root, manifest,
raw data, checker, and report remain immutable evidence. Its failures are
infrastructure failures, not adaptive-policy outcomes:

1. image-contract failure: specialist color permutation disagreed with the
   frozen observation table;
2. physical-scene failure: `branch_dev_04` declared clear routes that collided;
3. provenance failure: no fresh-process replay fingerprints were recorded.

No Phase 6A4 artifact is overwritten or promoted.

## Frozen design

- 12 independent public layouts, seed `66121`, new namespace and new bundle.
- Eight hidden states per layout: six main states plus controls `000/111`.
- 96 scenes and 288 candidate route executions.
- Color semantics are globally fixed. No color permutation is allowed.
- Layout variation is limited to cue positions, camera positions, occluder pose,
  route lane spacing, obstacle x-position, and other public geometry.
- Every layout manifest row contains its complete state-view-observation table,
  generated before any run. Outcomes are consumed from this table; they are not
  inferred after the run from image pixels.
- Every layout enumerates all 17 ordered fixed sequences of length 0-2. Fixed
  selection sees public layout metadata only; adaptive and fixed share budget,
  candidates, stopping, updater, tie-breaks, and executor.

## Physical preflight hard gate

Before a layout is admitted to the formal run, the official controller executes
all eight states and all three routes. For every route it records trajectory
hash, length, minimum clearance margin, nearest collision geometry, and collision
status. A route whose state bit is clear must complete collision-free with
minimum clearance at least `0.004 m`; otherwise the entire layout is rejected
and a new deterministic candidate is sampled. A blocked route must show the
registered obstacle contact. No route can be relabeled after collection.

Preflight artifacts are committed with the manifest and are distinct from the
main policy results. The checker refuses a formal result without a passing
preflight record for every layout.

## Provenance hard gate

The formal runner writes, before completion:

- Git HEAD and manifest SHA-256;
- checker version;
- per-scene simulator state fingerprint;
- independent Python-process replay fingerprint;
- every image SHA-256;
- every physical trajectory SHA-256;
- the full state-view-observation table.

Any missing or mismatched item marks the root `provisional` and blocks promotion
to a result. The new namespace cannot reuse `run_0ca1175` or any Phase 6A4
artifact.

## Statistical interpretation

Hard infrastructure gates are per-layout and mandatory: image contract,
clear-route preflight, hidden-state/public-metadata isolation, fresh-process
replay, and complete fixed-sequence enumeration. Adaptive gain is a scientific
outcome, not a per-layout gate. Report layout-cluster means and bootstrap
intervals for utility, completion, collision, stop, and query cost; layouts with
no gain remain valid heterogeneity rather than failures.

Only after this amendment's new run passes all hard gates may RGB perception be
separately preregistered. Continuous hidden geometry, ordinary particle
filtering, and learned particle updates remain frozen.
