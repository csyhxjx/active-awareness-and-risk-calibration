# Phase 6A4: Twelve-Layout Exact-Belief Development Amendment

Date: 2026-09-21. Status: prospective. This amendment authorizes only the
discrete exact-belief development expansion after the Phase 6A3 physical
branching gate. RGB training, particle filtering, learned updates, validation,
and sealed testing remain closed.

## Scope

- 12 independent public layouts, generated from frozen seed `66120`.
- Each layout has eight hidden states: six main states
  (`001,110,010,101,011,100`) and two controls (`000,111`).
- Total: 96 scenes and 288 candidate route executions.
- Every layout has the same three-family branching observation contract, but
  varies public cue positions, color permutation, camera poses, occluder pose,
  route lane spacing, obstacle x-position, and target geometry.
- The state-view-observation partition, route controller, posterior, costs, and
  stopping rule are frozen before generation. No runtime result may alter them.

## Matched comparison

For each layout, enumerate all 17 ordered paid-camera sequences of length 0, 1,
or 2 without replacement. Fixed selection may use only public layout metadata;
it cannot inspect hidden state, image pixels, collision outcomes, posterior
labels, or adaptive results. Adaptive and fixed policies share exactly the same
query budget, candidate routes, exact updater, terminal rule, tie-breaks, and
physical executor.

Report the six main states separately from `000/111` controls. Per layout and
pooled summaries include completion, collision, stop, query count, utility,
adaptive decision tree, all fixed-sequence outcomes, posterior supports, image
partition hashes, route traces, and fresh-process replay hashes.

## Acceptance gates

1. 12/12 layouts preserve initial-input isolation and broker state/RNG/time
   invariants.
2. 12/12 layouts have exact within-outcome and distinct-between-outcome image
   partitions for all four paid views; no filename or metadata cue is used.
3. 12/12 layouts have all 24 route outcomes correct (six main plus two control
   states, three routes each), with failures retained and no silent repair.
4. The adaptive policy has valid route decisions on all six main states in each
   layout; controls are reported but excluded from the primary adaptive-gain
   denominator.
5. A majority of layouts have adaptive utility strictly above their best fixed
   B=2 utility, and pooled main-state completion is reported with layout-group
   bootstrap intervals.
6. The fixed baseline is not weakened: every layout's best fixed result is the
   maximum over all 17 sequences under the shared rules.

If the majority-layout or image-contract gate fails, stop at development and
report the failure mode. Do not add models, tune the planner, or reinterpret a
single-layout mechanism fixture as a general result.

## Provenance

The manifest, generator commit, environment, seed, raw run root, checker,
figures, and hashes are committed after the amendment and before execution.
All failed attempts remain outside the canonical artifact directory and are
listed in planning. This amendment does not authorize RGB perception or any
continuous hidden-geometry experiment.
