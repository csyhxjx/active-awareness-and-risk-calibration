# Phase 6A7: ROI-Contract Twelve-Layout Development Amendment

Date: 2026-09-21. Status: prospective protocol only. This amendment follows
the passing Phase 6A6 one-layout repair probe. It does not modify or rehabilitate
the permanently failed Phase 6A4 and Phase 6A5 roots. No implementation change,
manifest generation, rollout, RGB training, or particle experiment is
authorized before this document is committed and visible remotely.

## 1. Observation Contract

The mechanism receives only a preregistered cue ROI from each purchased view.
The complete RGB frame is archived for provenance and manual audit but is not a
policy input. Pixels outside the ROI, including hidden geometry, robot links,
backgrounds, and lighting changes, cannot update the belief or affect view or
route selection.

Each layout manifest row defines every ROI rectangle and the complete
`state x view -> outcome` table before execution. Color semantics are globally
fixed to the passing Phase 6A6 detector contract; no color permutation is
allowed in this experiment. Same-outcome ROI detector outputs must agree and
different outcomes must be separable. Whole-image hashes may differ. Any
unregistered object, route obstacle, robot body, or hidden-state cue entering a
registered ROI is an infrastructure failure for the whole layout.

## 2. Physical Generation and Preflight

Candidate layouts are drawn deterministically from new seed `66212`. Public
variation is restricted to registered cue positions, paid-camera poses,
occluder geometry, route lane spacing, obstacle position, and target geometry.
The generator uses one frozen route template family; it may not special-case
`branch_dev_04` or any layout id.

Before admission, every candidate is preflighted in all eight hidden states and
all three routes with the same official controller used in the formal run.

- A clear route must complete without route-obstacle, static-environment, or
  robot self-collision.
- When actual obstacles are present, its minimum geometric clearance to every
  non-target active obstacle must be at least `0.004 m`.
- An absent obstacle is excluded from clearance computation because no physical
  geom exists at its nominal location.
- Static environment and robot collision checks remain mandatory regardless of
  obstacle presence.
- Each preflight row records completion, collision classes, nearest physical
  geom, minimum distance, trajectory length, and trajectory SHA-256.

Any failed candidate is rejected before the formal manifest is frozen. The
generator advances deterministically to the next candidate and records the
candidate index, failure reasons, and cumulative rejection count. Rejected
candidates remain in the preflight ledger. No admitted layout may be edited,
relabelled, or repaired after formal execution.

## 3. Scale and Matched Comparison

The admitted manifest contains exactly 12 layouts, eight states per layout,
96 scenes, and 288 canonical route executions. Six states
`001,110,010,101,011,100` form the adaptive-gain analysis; `000/111` are
reported separately as controls.

Every layout enumerates all 17 ordered fixed sequences of length zero, one, or
two without replacement. The best fixed B=2 policy may use only public layout
metadata and the frozen validation-free rule; it cannot inspect hidden state,
ROI pixels, oracle outcomes, collision results, adaptive decisions, or route
outcomes. Adaptive and fixed methods share exactly the same B=2 budget, cue ROI
inputs, belief updater, stopping rule, candidate routes, tie-breaks, and
physical controller.

Adaptive gain is a scientific outcome, not a per-layout infrastructure gate.
After hard gates pass, compare adaptive exact with best fixed B=2 using layout
as the cluster unit. Report layout-level and pooled differences in utility,
collision-free completion, collision, stop, and mean query cost, with a
deterministic layout bootstrap and all heterogeneous/no-gain layouts retained.

## 4. Provenance Hard Gate

Every scene record must contain:

- protocol id, Git HEAD, manifest SHA-256, checker version, and seed;
- ROI bounds, ROI hash, full-image hash, detector output, and expected outcome;
- route trajectory SHA-256, trajectory length, clearance, nearest physical
  geom, collision classes, completion, and terminal reason;
- simulator/controller/RNG/time fingerprint before queries;
- an independently launched Python process fingerprint and byte comparison.

The fresh-process field is computed from the subprocess output during the
formal run. Constants, placeholders, inherited booleans, and post hoc summary
patches are prohibited. A missing or mismatched field makes the scene
provisional and fails its layout. The runner records its shard id when multiple
GPUs are used; shards have disjoint layout ids and are merged only after each
has closed its own summary.

## 5. Run Order

1. Commit and push this amendment.
2. Implement the generator, preflight, runner, and checker without touching the
   Phase 6A4/A5 raw roots or manifests.
3. Generate a new candidate/preflight ledger and admitted manifest under a new
   namespace from seed `66212`.
4. Perform a static audit of counts, uniqueness, observation tables, ROI bounds,
   color semantics, rejection ledger, hashes, and public/hidden fields.
5. Commit and push the manifest, preflight ledger, implementation, and checker
   before any formal rollout.
6. Run the 12 layouts once. Multiple GPUs may execute disjoint manifest shards.
7. Run all hard infrastructure checks before computing adaptive comparisons.
8. If any hard gate fails, archive the root unchanged and stop. Do not repair
   then recompute within the same protocol.

## 6. Hard Gates and Terminal Fork

Every admitted layout must pass ROI observation semantics, hidden/public input
isolation, physical preflight and formal route consistency, real fresh-process
replay, complete provenance, and enumeration of all 17 fixed sequences. These
are infrastructure gates. Adaptive gain is evaluated only after all 12 layouts
pass them and may vary legitimately by layout.

If this experiment again fails ROI isolation, physical-route validity, replay,
or another infrastructure gate, the controlled color-cue expansion is closed.
No fourth repair amendment or another 12-layout color-cue rerun is permitted.
The next work must redesign either a natural-RGB active-vision task or a
continuous-hidden-geometry task under a new scientific protocol.

Only an all-hard-gates PASS may authorize a separately preregistered RGB
perception stage. Continuous geometry, ordinary particle filtering, and learned
particle proposals/updates remain frozen regardless of the adaptive effect size
in this development run.
