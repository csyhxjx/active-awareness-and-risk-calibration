# Phase 5E route-proposer smoke

The 8-layout train-only smoke passed: coverage `64/64`, preference accuracy `64/64`, stop `0/64`, candidate-recall failure `0`, paired hidden-state outputs exact, and physical transfer `64/64`. Instruction-only, V0+instruction, and V0+instruction+geometry each achieved 100% accuracy and coverage. No selector was trained and no validation/test layout was opened.

This confirms the route-level proposal contract but does not show that the proposer uses V0 or geometry: the instruction-only control is equally perfect. The primary proposer remains the preregistered three-seed mean of V0+instruction+geometry logits.
