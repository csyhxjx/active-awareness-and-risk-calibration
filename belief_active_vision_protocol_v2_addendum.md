# Phase 6A3: Genuine Adaptive-Branching Addendum

Date: 2026-09-20. Status: prospective protocol clarification. This addendum
preserves the completed Phase 6A2 oracle demo as evidence that observation can
support route revision. It withdraws authorization for the registered
12-layout expansion until a stronger adaptive-branching gate passes. No RGB
training, particle method, validation split, or test split is authorized.

## 1. Completed-result correction

The Phase 6A2 comparison against fixed `q_left -> q_front` is not evidence that
adaptive view selection outperforms a strong fixed policy. The second fixed
view is uninformative in that layout. Exhaustive evaluation over all eight
hidden states shows that the exact planner is action-for-action identical to
the public-layout fixed sequence `q_left -> q_right` with early stopping.

The retained Phase 6A2 claim is therefore limited to: a paid observation can
falsify the preferred route, a second observation can certify an alternate
route, and route switching can recover collision-free task completion.

## 2. Strong fixed baseline

Every later development comparison must include the best fixed query sequence
chosen from public layout information only. It may stop early as soon as the
shared execution condition is met and may execute any route after observing.
It receives the same free input, observation likelihoods, candidate routes,
query budget, belief updater, execution threshold, tie breaks, and switching
rights as the adaptive planner.

For budget `B`, enumerate every ordered camera sequence of length at most `B`,
including early-stop branches. Selection of the best sequence may use only
development layouts and public geometry. Report both its expected utility and
its per-hidden-state decisions; a deliberately uninformative fixed sequence is
not a valid headline baseline.

## 3. Genuine branching requirement

Direct deterministic `clear/blocked` observations are insufficient in the
current three-independent-bit task: continuation implies all previously
checked routes were blocked, so the next query can be fixed in advance. The
replacement diagnostic must use partial observations that reduce the hidden
hypothesis set without immediately certifying a route.

At least one public layout must implement all of the following:

1. The same first camera has at least three positive-probability outcomes.
2. After each outcome, at least two hidden hypotheses remain and no route meets
   the shared execution condition.
3. The three posterior branches require three distinct second cameras to
   resolve; the required second camera is a function of the first outcome.
4. Every registered branch contains a feasible route, so success cannot be
   obtained by learning only to stop.
5. The adaptive `B=2` policy resolves every branch and selects a feasible route.
6. Exhaustive enumeration proves that no public-layout fixed sequence of at
   most two paid cameras, even with early stopping and route switching, resolves
   every branch.

Three branches are required because with only two branch-specific follow-up
views, a fixed `B=2` policy could query both follow-up views and bypass the
branching camera. A valid construction may use three complementary pairs of
scene hypotheses, but the likelihood table must be fixed by scene geometry
before execution and cannot depend on the evaluated policy.

The first oracle fixture uses the six nontrivial states below. `000` and `111`
remain physical and leakage controls but are excluded from the adaptive-gain
denominator. They must be reported separately and cannot be used to improve the
headline comparison.

| First outcome | Remaining states | Required follow-up |
| --- | --- | --- |
| `family_a` | `{001, 110}` | `q_a` |
| `family_b` | `{010, 101}` | `q_b` |
| `family_c` | `{011, 100}` | `q_c` |

Within its registered pair, `q_a`, `q_b`, or `q_c` distinguishes the two
states. Outside that pair it returns the same `not_applicable` symbol for both
members of every other pair. Each pair contains complementary route bits, so
the first outcome alone certifies no route; every one of the six states still
has at least one clear route. The four paid camera roles are therefore
`q_branch`, `q_a`, `q_b`, and `q_c`; their physical poses may vary with public
layout geometry, but their partition contract may not change after execution.

This table makes the fixed-policy test explicit. `q_branch` plus one specialist
resolves one family, while any two specialists resolve at most two families.
The checker must nevertheless enumerate every ordered sequence rather than
assuming this argument.

## 4. Observation semantics

The oracle development interface now permits partial symbols such as
`family_a`, `family_b`, `family_c`, or a calibrated likelihood vector. It must
not convert every purchased image directly into route-level `clear/blocked`.
`unobserved` remains non-evidence. The exact eight-state posterior remains the
reference implementation; no particle approximation is needed or allowed for
this gate.

For each camera, archive the fixed hidden-state-to-observation partition and
the corresponding RGB image. The broker continues to hide unpurchased images.
Camera identity, filename, public geometry, and metadata must not leak the
hidden state. Querying must remain state preserving.

## 5. Pre-expansion acceptance gate

Before the 12-layout suite, archive:

- at least two paired traces with the same public layout and free input but
  different first observations, different second cameras, and successful
  execution of feasible routes;
- the third branch needed for the fixed-sequence impossibility proof;
- the complete adaptive decision tree and every fixed sequence of length at
  most two;
- posterior support after each observation and the terminal decision;
- matched-budget utility, completion, collision, stop, and query counts;
- state/RNG/time preservation and fresh-process replay hashes.

The gate passes only if the adaptive policy resolves all registered branches
and every fixed sequence fails on at least one branch under the same execution
rule. If this fails, stop and redesign the observation geometry; do not expand
layout count and do not interpret a weak-fixed comparison as active-view gain.

## 6. Downstream scope

Only after this gate passes may a new amendment reopen the 12-layout
development suite. That amendment must retain the strong fixed baseline and
measure adaptive gain at the layout-group level. RGB recognition remains a
later separately frozen stage. Ordinary and learned particle filters remain
deferred until continuous hidden geometry makes exact enumeration inadequate.
