# Phase 5E post-smoke split and power addendum

Date: 2026-09-20. This addendum fills only values permitted by `active_vision_protocol_v4.md`; it does not change its gates, model, formula, split sizes, or seeds.

The train-only smoke passed all hard gates: 64/64 valid non-stop proposals, 64/64 route-preference accuracy, zero proposer-stop, zero candidate-recall failure, exact hidden-state invariance, and 64/64 expected clear/blocked physical transfers. All three registered ablations achieved 64/64 preference accuracy, so this smoke does not demonstrate incremental use of V0 or geometry beyond instruction.

No active-versus-blind selector contrast exists in the proposer-only smoke, so `s_dev` for the formal mechanism endpoint is unavailable rather than estimated as zero. The preregistered floor therefore controls: `s_plan=0.20`. With alpha 0.05 two-sided, power 0.80, and `d_min=0.10`, the formula gives `ceil(((1.959964+0.841621)*0.20/0.10)^2)=32` layout groups. The already frozen 40-group sealed test exceeds this planning minimum. This calculation is prospective and does not borrow Phase 5B/5D outcomes.

The exact 60/20/40 formal layout rows are frozen in `configs/active_vision_v4/formal_manifest.json`, SHA-256 `a34781be5b8617da18fc89b69bff2bc6fc2296a6e7c86a8ed7617855dc7778d8`. Seeds remain train `55060`, validation `56020`, test `57040`, and scene `20260921`. All IDs use the new `av4_` namespace. Test remains sealed until train collection, validation-only choices, checkpoint hashes, and evaluator are frozen and pushed.
