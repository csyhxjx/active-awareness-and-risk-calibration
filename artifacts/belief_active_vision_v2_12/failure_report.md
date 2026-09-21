# Phase 6A4 Twelve-Layout Development: Stopped at Gate

The frozen manifest contains 12 layouts, 8 states per layout, and 24 route
trials per layout. The run completed all 12 layouts and generated 96 scenes and
288 route executions. The result is retained at
`/internsdata/yewenhao/guard_workspace/belief_active_vision_v2_12/run_0ca1175`.

The symbolic exact planner shows a strict adaptive-versus-best-fixed gap on
11/12 layouts. That is not sufficient for a pass. The image checker found that
the layout-level color permutation changed specialist cue colors without
updating the frozen observation table, so several layouts violate the required
state-view-image contract. The physical checker also found that
`branch_dev_04` has left-route collisions in main states `010` and `011`,
making adaptive completion only 4/6 there.

The first 12-layout runner did not yet record fresh-process fingerprint files;
that is an additional provenance gap. No result is promoted to a canonical
positive expansion artifact. The 12-layout gate therefore fails, and no RGB,
continuous-geometry, particle-filter, or learned-particle experiment is
authorized.
