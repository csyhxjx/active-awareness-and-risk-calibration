# Paper positioning after Phase 5E

## 1. Mechanism characterization: B=1 converts unsafe execution into abstention

The primary result is a scoped negative/机制 result, not a completion-gain claim. In the frozen `B=1` contract, the policy may inspect only the proposer's argmax route and then either execute that same route or stop. It cannot select the alternate route. Consequently, observation can improve collision-free completion only when it prevents an execution that would both collide and otherwise be counted as non-completion; rejecting a blocked proposal lowers collision but does not create a newly completed trial. This is the B=1 safety-conversion theorem for this benchmark, not a theorem about active vision generally.

The complete sealed-test ladder supports that account. Blind execution completes 50% and collides on 50%. Candidate-aware geometric observation preserves 50% completion while reducing collision to 25% through 25% abstention. Candidate-aware learned observation also preserves 50% completion while reducing collision to 0% through 50% abstention. Neither completion contrast is positive: both registered deltas are zero with layout-clustered 95% CI `[0,0]`.

The controls sharpen the result. Best-fixed and candidate-agnostic learned selection are identical at 36.875% completion, 11.875% collision, 48.75% execution, and 51.25% stop. Registered random view averages 40.125% completion, 17.125% collision, 57.25% execution, and 42.75% stop. Thus the candidate-agnostic model's very different training behavior does not translate into a sealed-test advantage. The evidence favors candidate-conditioned risk filtering, while rejecting a learned completion advantage.

Every completion statement must retain the qualifiers: route-level proposer, controlled two-route task, one paid view, and single-candidate `B=1` verification. The safety endpoint was secondary in Phase 5E, so collision reductions are reported as mechanism evidence rather than substituted for the failed primary endpoint.

## 2. Interface diagnosis: OpenVLA integration failed before active selection

Phase 5D is an integration limitation, not evidence that OpenVLA cannot propose useful behavior. All 64 chunks were finite and shape-valid, and changing the route instruction changed all 32 matched chunks. Outputs were exactly invariant to hidden state for identical V0 and instruction. However, 84.60% of values had absolute magnitude at most 0.1 and the local OpenVLA contract defines them as end-effector deltas, while the frozen Phase 5D projection interpreted the last row as a normalized absolute table-frame waypoint. Every trial therefore mapped to `stop`, yielding frozen non-stop coverage `0/64`.

The read-only diagnosis favors delta/absolute semantic mismatch over constant task-OOD collapse, but does not retroactively repair Phase 5D. OpenVLA B/C extensions require new protocols and cannot inherit the route-level 5E result.

## 3. New scientific questions

1. **Risk-controlled abstention:** at a validation-frozen operating point, can a B=1 observer satisfy a prospective collision-risk bound while retaining more safe executions than fixed or candidate-agnostic abstention? This is the only opened next phase.
2. **Marginal value curve:** how does risk and retained safe execution change with observation cost and number of views? In the current two-candidate world, B=2 alternate-route inspection is an illustrative ceiling, not an independent experiment; a scientific budget curve needs a richer candidate/view space and a new protocol.
3. **Visual-decidability prediction:** can a pre-query predictor identify whether a candidate violation will be resolvable from a given camera, separating observation selection from downstream risk classification? This remains future work and is not part of the opened risk study.
