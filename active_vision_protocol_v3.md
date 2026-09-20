# Phase 5D OpenVLA Proposer Integration Protocol

Date: 2026-09-20. Version: `active_vision_v3`. Status: design and preregistration draft; no OpenVLA inference, selector training, new manifest, or 5D rollout has run.

## 1. Scope and relationship to Phase 5B

Phase 5B is a completed fixed-controller controlled-occlusion experiment. Its sealed 40-layout test is an immutable result and is not reused for 5D tuning, validation, or confirmation. Phase 5D asks a new question: when the proposer is an OpenVLA model that does not know the hidden obstacle state, can the same-budget active observation loop reduce execution risk relative to blind execution under the same proposer?

The 5B result remains unchanged: learned and geometric selectors had equal completion, while learned had fewer collisions. Phase 5D may establish transfer to an OpenVLA proposer; it cannot retroactively turn the 5B result into a learned-selector superiority claim.

## 2. Frozen architecture: chunk projected to fixed candidates

Use architecture A. The candidate set is exactly `{left_route, right_route, stop}`, matching 5B. The world, route controller, collision oracle, query broker, and paid views remain the 5B interfaces. OpenVLA proposes a short action chunk; it does not directly define a new long-horizon route or new physics.

The proposer runs once before the decision. There is no in-rollout re-proposal in this phase. A chunk has a fixed horizon `H_chunk=8` environment actions and is recorded in full. It is mapped deterministically to the fixed candidate whose terminal end-effector waypoint has the smallest Euclidean distance to the chunk's terminal end-effector position, after transforming the chunk into the public table frame. Ties resolve `left_route`, then `right_route`, then `stop`. A mapping is valid only when all seven action values are finite, within the frozen action bounds, and the terminal pose is finite; invalid output is a proposer failure. The exact projection code, waypoint coordinates, action bounds, and pose transform must be committed before inference.

The mapped candidate is the proposer output for all downstream policies. The nominal B=0 arm executes the mapped route candidate without a paid view; it does not execute the raw chunk. This keeps 5D dynamics comparable to 5B. The raw OpenVLA chunk is still replayed and stored as proposer evidence. A separate diagnostic may execute the raw chunk only after a new protocol; it is excluded from the primary result.

## 3. Proposer variants and prediction

The first proposer is the frozen local OpenVLA checkpoint in greedy, sampling-disabled inference. Because the controlled occlusion task is outside its demonstrated task distribution, the preregistered prediction is that zero-shot non-stop route coverage may be below the 60% smoke threshold; an all-stop proposer therefore fails the coverage gate.

If zero-shot coverage is below threshold, the only authorized repair branch is OpenVLA plus LoRA. LoRA training may use nominal train layouts only, with no hidden obstacles, hidden-state labels, collision outcomes, or paid views. The model must remain ignorant of hidden obstacle state. The LoRA base revision, train layout IDs, instruction templates, rank, alpha, learning rate, steps, seed, and checkpoint hash are frozen before the repair run. If the LoRA branch remains below threshold, Phase 5D stops as not testable for this proposer; a small non-VLA proposer requires a separately preregistered protocol and cannot be relabeled as OpenVLA evidence.

## 4. Single proposal and decision semantics

Each trial records the following immutable five-tuple before outcome aggregation:

`(proposer_output, mapped_candidate, selector_decision, executed_action, physical_outcome)`.

`proposer_output` contains input image hashes, instruction, model revision, tokenizer revision, inference configuration, raw action chunk, and chunk hash. `mapped_candidate` is `left_route`, `right_route`, or `stop`, plus mapping distance and validity. `selector_decision` contains purchased views, verifier outputs, selected/rejected candidate, and stop reason. `executed_action` contains the fixed route ID or explicit stop. `physical_outcome` contains collision, completion, timeout, and oracle event details.

Failure categories are derived mechanically from this tuple:

- **Proposer failure:** no finite valid chunk or no valid projection. The proposer did not provide a usable candidate.
- **Proposer stop:** the proposer maps to `stop`; this is a valid proposer decision and is reported separately from selector stop.
- **Candidate-recall failure:** the proposer maps to no member of the public candidate set even though a valid candidate exists. Under the total three-way projection this is predicted to be structurally zero; any nonzero count signals an interface or implementation defect.
- **Selector failure:** the proposer produced a valid route candidate, the policy purchased its allowed views, and the selector made an incorrect execute/reject/stop decision conditional on that proposer-valid denominator.

Selector metrics must therefore report both all-trial denominators and the conditional denominator `proposer_valid && mapped_candidate in {left_route,right_route}`. Proposer-stop, selector-stop, unresolved, timeout, collision, and candidate-recall counts are separate fields; none may be merged after the run.

## 5. OpenVLA determinism and replay

Inference is batch size 1, greedy decoding, sampling disabled, fixed image preprocessing, fixed instruction text, and fixed action postprocessing. Record the model checkpoint/revision hash, tokenizer files and hash, transformers and torch versions, CUDA device, `CUBLAS_WORKSPACE_CONFIG`, deterministic-kernel settings, environment variables, seed, input image SHA-256, output chunk bytes and SHA-256, and inference stdout/stderr.

An exact replay from a fresh process and environment must reproduce input hashes, output chunk bytes, mapped candidate, query history, fixed-route actions, and physical branch hashes. The smoke audit set is all 64 `(layout, hidden_state, route_preference)` trials, so no trial can be silently omitted from replay accounting. Replay error is required to be zero for serialized chunk/actions and branch state/RNG hashes; numerical closeness is not an acceptable substitute. Any failure is recorded and blocks expansion until diagnosed under a new committed revision.

## 6. Fresh 5D data namespace and sealed test

The 5D manifest builder must use a new namespace and new seed. The 5B 40-layout test manifest and every one of its derived states are permanently excluded from 5D. A prospective formal expansion uses new grouped train/validation/test layouts, with test generated from a distinct 5D test seed (`54040`) and sealed before any test-dependent choice. The exact train and validation seeds, family allocation, and sample size are fixed in the full 5D protocol only after the smoke passes; no 5B test layout may reappear.

The smoke uses 8 new train-only layout groups, balanced across the registered occlusion families, four hidden states per group, and both route preferences. It does not train a selector, tune a view policy, or open validation/test. With one proposer inference per trial and at most three fixed-candidate route rollouts, the expected budget is 64–192 proposer/route units plus exact replay audits; actual counts are logged rather than inferred.

## 7. Smoke gates and decision rule

The smoke is run only after the projection, tuple schema, observation broker, and deterministic replay tests pass. It has six gates:

| Gate | Pass condition |
|---|---|
| D0 interface and isolation | Projection is deterministic; query does not change qpos/qvel/ctrl/RNG; no hidden-state label, oracle outcome, or hidden geometry enters proposer inputs or selector features (paid RGB pixels are the deliberately permitted observation); tuple schema is complete. |
| D1 OpenVLA replay | Fresh-process greedy replay reproduces every input/output chunk and branch hash exactly on audit trials. |
| D2 proposer mapping | At least `60%` of all smoke trials produce a valid non-stop route mapping (`left_route` or `right_route`); proposer-stop is reported separately and does not count toward coverage. |
| D3 candidate recall | Candidate-recall failure is exactly `0`; any nonzero result is an interface defect and stops the phase. |
| D4 route transfer | On proposer-valid mapped routes, clear states complete and blocked states collide under the fixed 5B route oracle; all failures remain in the denominator. |
| D5 attribution | Proposer, proposer-stop, selector, collision, timeout, and unresolved categories can be reconstructed from the five-tuple without manual relabeling. |

If D0, D1, D3, D4, or D5 fails, stop and fix only under a new committed protocol/code revision. If only D2 fails, run the preregistered nominal-only LoRA repair branch once; if its coverage is still below 60%, stop 5D as not testable for OpenVLA. No selector training or formal 5D expansion occurs before all smoke gates pass.

## 8. Later 5D comparison, conditional on smoke

The primary comparison is the same proposer and same mapped candidate under nominal B=0 versus candidate-aware active observation with B=1. The selector may execute, reject, or stop, but all methods share the route candidate, verifier, budget, and collision oracle. The key endpoints are collision-free completion, collision rate over all trials, execution coverage, conditional selector error, proposer-valid coverage, proposer-stop rate, candidate-recall rate, unresolved rate, query count, and exact replay rate.

The expected high-probability outcomes are preregistered: the geometric rule may again match learned completion, or proposer coverage may be too low to answer the question. If geometric matches learned, report mechanism value without a learned-method claim. If coverage stays below 60% after the authorized LoRA branch, report that active observation for this OpenVLA proposer was not testable under the budget. Neither outcome permits test reuse or post hoc rescue.

Phase 5D does not claim whole-task safety. Any later OpenVLA integration must separately report candidate recall, proposer failures, action-chunk horizon limits, and failures caused by the proposer rather than by visual selection.
