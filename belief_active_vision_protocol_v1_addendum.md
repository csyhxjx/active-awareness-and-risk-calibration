# Phase 6A2: Three-Route Closed-Loop Development Addendum

Date: 2026-09-20. Status: prospective development protocol. This addendum supersedes the protocol's immediate jump from four-state logic to continuous geometry: an eight-state, three-route closed loop is inserted first. It does not alter or delete the completed four-state unit baseline, and it does not authorize particle learning.

## 1. Task and state

The route set is ordered `left_route`, `center_route`, `right_route`; `stop` is terminal abstention. All routes share one fixed OSC controller, start, target, tolerance, horizon, and collision oracle. The hidden state is the three-bit vector `z_L z_M z_R`, yielding all eight states from `000` through `111`, where `1` means the corresponding route obstacle is present. The initial free RGB, robot state, instruction, public layout, and filenames must be identical across all eight states of a layout.

The proposer supplies only the preference order `left > center > right`; after every purchased observation, every method may execute any route or stop. No method is forced to execute the first proposal. Clear routes must complete without registered obstacle collision and blocked routes must collide under `route_collision_v2`.

## 2. Observations and budget

There are four paid pre-action RGB cameras: `q_left`, `q_right`, `q_front`, and `q_high`. The primary budget is two image units; B=0, B=1, and B=3 are descriptive budget-curve points. After the first purchase the policy may choose a different second camera based on its updated belief. Querying never advances simulation or changes qpos, qvel, act, ctrl, controller state, RNG, or time.

Camera names do not define information. Every development layout records an oracle visibility table whose entry for `(camera,route)` is `clear`, `blocked`, or `unobserved` conditional on the route bit. `unobserved` has likelihood one for every value of that route bit and therefore cannot update belief. The table is derived from registered scene visibility geometry before policy execution, not selected from outcomes. Four RGB images are cached for audit, but the first development stage consumes oracle visibility outcomes and is explicitly an information upper bound.

## 3. Exact belief and decisions

The exact belief has eight probabilities in lexicographic state order. The prior is uniform. Oracle `clear/blocked` observations deterministically eliminate inconsistent states; `unobserved` leaves the belief unchanged. A query is chosen by exact enumeration of observation branches to maximize expected terminal utility minus `0.05` per purchased image. Terminal utility is `+1` for collision-free completion, `-4` for collision, and `-0.25` for stop. The planner may stop early when further observation has no positive decision value.

Execution is permitted only for a route whose posterior blocked probability is exactly zero during the oracle development stage. Among executable routes, preference ties resolve left, center, right. If none is certified and a valuable legal query remains, query; otherwise stop. All baselines share this execution condition and route-switch permission.

## 4. Minimal demonstration

The first implementation is one layout and all eight hidden states, but its report highlights state `100` with left preference. The registered adaptive trace is `q_left -> blocked -> q_right -> clear -> right_route -> collision-free completion`. The matched fixed sequence is `q_left -> q_front`; in this layout `q_front` is unobserved for center/right, so it exhausts B=2 without a certified route and stops. Both policies receive the same initial input, budget, action set, oracle updater, and execution rule.

The demo passes only if: all eight V0 images are byte-identical; all 24 route transfers match their bits; all paid queries preserve full simulator state; adaptive uses no more than two images and completes the `100` case by switching routes; fixed uses the same budget and stops; tuple/replay hashes are exact; and the four cached paid images plus both decision traces are archived. Failure stops before the 12-layout suite.

## 5. Twelve-layout development suite

Subject to demo PASS, build exactly 12 new development layouts with seed `66012`, eight states each, 96 scenes, 288 canonical route transfers, and four cached paid views per scene. Layouts vary public occluders, route lengths, mirror, camera pose, and oracle visibility table. All attempts are retained; development layouts never enter formal splits.

Gates are: physical correctness in at least 10/12 layouts with every failure retained; zero initial hidden-state leakage by byte/hash and schema audit; at least four layouts with a positive second-query decision value; at least four layouts with one-observation early execution; at least four layouts where B=2 remains unresolved; at least six preferred-route-blocked cases rescued by a route switch; and no fixed camera or fixed length-two sequence resolving every layout. Broker/state/replay/tuple gates require 12/12.

The ideal independent-clearance references `1-(1/2)^2=0.75` for two checked routes and `1-(1/2)^3=0.875` for three are diagnostic arithmetic only, not gates or expected model scores.

## 6. Baselines and later RGB stage

The development comparison contains blind preferred-route execution, always-stop, validation-style fixed sequence (development diagnostic only), random without replacement, public geometric coverage, maximum entropy reduction, exact-belief decision value, all views, and full-state oracle. Every observation strategy shares candidate routes, switching rights, updater, budget, execution threshold, and physical controller.

After oracle development gates pass, a separate amendment must freeze the RGB classifier, confidence likelihood, calibration, execution condition, grouped formal splits, risk margin, bootstrap unit, and sample size before training or formal collection. Existing Phase 5 data and tests remain prohibited. The continuous ordinary-PF stage and learned-particle stage remain closed until the discrete RGB closed loop is complete.
