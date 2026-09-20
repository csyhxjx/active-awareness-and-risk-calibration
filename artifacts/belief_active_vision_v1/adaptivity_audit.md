# Phase 6A2 Adaptivity Audit

Date: 2026-09-20. Audited implementation: `40d2169`.

## Question

Does the Phase 6A2 exact planner select a second camera conditionally, or can a
public-layout fixed sequence with early stopping reproduce it?

## Result

Exhaustive enumeration of all ordered, no-replacement fixed sequences of length
zero, one, or two finds two tied optima: `q_left -> q_right` and
`q_right -> q_left`, both with early execution as soon as a route satisfies the
shared zero-blocked-probability rule. The former exactly matches the adaptive
planner in all eight hidden states.

Under the uniform eight-state prior, both policies complete 6/8 states, collide
in 0/8, stop in 2/8, and purchase an average of 1.5 images. Including the frozen
query cost, both have expected utility `0.6125`.

The next-best one-view policies have utility `0.3250`; the registered weak
`q_left -> q_front` control has utility `0.3000`. Thus the result is not a
near-tie hidden by metric choice: the registered control omitted a strictly
better public fixed sequence.

| State | Exact planner | Fixed sequence with early stop | Equal |
| --- | --- | --- | --- |
| `000` | `q_left -> left_route` | `q_left -> left_route` | yes |
| `001` | `q_left -> left_route` | `q_left -> left_route` | yes |
| `010` | `q_left -> left_route` | `q_left -> left_route` | yes |
| `011` | `q_left -> left_route` | `q_left -> left_route` | yes |
| `100` | `q_left -> q_right -> right_route` | `q_left -> q_right -> right_route` | yes |
| `101` | `q_left -> q_right -> stop` | `q_left -> q_right -> stop` | yes |
| `110` | `q_left -> q_right -> right_route` | `q_left -> q_right -> right_route` | yes |
| `111` | `q_left -> q_right -> stop` | `q_left -> q_right -> stop` | yes |

The reason is structural. A direct route observation is deterministic and a
single `clear` result immediately permits execution. The planner continues only
after `blocked`, so there is only one continuing history and the next camera
can be chosen before the episode begins.

## Claim boundary

The existing demo proves route-revision rescue, query isolation, and physical
execution. It does not prove that conditional view selection beats a strong
fixed sequence. The 12-layout expansion is blocked until the genuine branching
gate in `belief_active_vision_protocol_v2_addendum.md` passes.
