# Phase 6A3 Physical Adaptive-Branching Gate

## Scope

This is the one-layout mechanism gate registered in
`belief_active_vision_protocol_v2_addendum.md`. Route obstacles determine real
controller collision outcomes. Four physical color cues implement the frozen
state-view-observation partition. The cues are an oracle-readable RGB upper
bound, not a learned or natural obstacle recognizer.

- implementation HEAD: `cd8990ee8564a96a64b94549a589385f2db7b875`
- layout: `branch_demo_00`
- states: `001, 110, 010, 101, 011, 100`, uniform prior `1/6`
- budget: `B=2`
- raw root: `/internsdata/yewenhao/guard_workspace/belief_active_vision_v2/branch_demo_cd8990e_attempt4`
- raw summary SHA-256: `81c8ccc810f6318469804a6aedab53a9783ff7dc21c461189dac845c24403d03`
- contract SHA-256: `be6356d1dff48cd6957845ca151f89181a28961726be59f054d09b0e5290561b`

## Development Attempts

| Attempt | Result | Retained diagnosis |
| --- | --- | --- |
| `46e3869_attempt1` | FAIL | `q_branch` and `q_b` cue cubes occluded by robot geometry |
| `08472ad_attempt2` | FAIL | `q_branch` fixed; `q_b` still occluded |
| `8ccde82_attempt3` | provisional PASS | same-process fresh-env replay only; excluded by final provenance audit |
| `cd8990e_attempt4` | PASS | all image, belief, fresh-process replay, physics, and policy gates pass |

No failed root was reused as passing evidence. The corrections changed only
cue/camera poses; prior, observation table, planner, costs, routes, and
acceptance criteria remained fixed.

## Contract and Image Audit

The complete prior, 24-cell observation table, three two-state posteriors, six
singleton posteriors, terminal actions, adaptive traces, and all 17 fixed
sequences are in `contract.json`.

- `V0`: byte identical across all six states.
- `q_branch`: three distinct image groups, exact within each family.
- `q_a`, `q_b`, `q_c`: yellow/purple distinguish their registered pair;
  all four out-of-family states are pixel-identical gray.
- Every camera query preserves simulator/controller/RNG state, and every state
  fingerprint reproduces exactly in a separate Python process.

Manual review agrees with the hash partition: the claimed symbols are visible
as surface colors and are not inferred from filenames. State labels in
`image_audit.png` are report annotations outside the stored input PNGs.

## Physical and Policy Results

All 18 canonical route executions agree with the route bits: every clear route
completes collision-free and every blocked route collides.

| Policy | Completion | Collision | Stop | Mean queries | Mean utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact adaptive | 6/6 | 0/6 | 0/6 | 2.0 | 0.9 |
| Best fixed B=2 | 4/6 | 0/6 | 2/6 | 1.667 | 0.5 |

All six ordered pairs of distinct specialist views tie as the best fixed
policies. Each resolves two families and leaves the third ambiguous. Sequences
that spend one query on `q_branch` cannot condition their fixed second camera
and resolve only one family.

Adaptive traces are:

- `family_a`: `q_branch -> q_a`, then `left_route` for `001` or `right_route` for `110`.
- `family_b`: `q_branch -> q_b`, then `left_route` for `010` or `center_route` for `101`.
- `family_c`: `q_branch -> q_c`, then `left_route` for `011` or `center_route` for `100`.

## Decision

The preregistered one-layout genuine-branching gate passes. This establishes a
controlled adaptive-view mechanism gain over every fixed B=2 sequence with
matched candidates, updater, stopping rule, and execution rights. It does not
establish RGB generalization or justify particles.

The 12-layout stage is eligible to be reopened only by a new prospective
amendment that freezes how cue geometry, camera roles, layout families, and
strong fixed baselines vary. No 12-layout rollout, RGB training, continuous
geometry, particle filter, or learned particle network was run here.
