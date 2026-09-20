# Phase 6A2 Three-Route Closed-Loop Demo

## Scope

This is the preregistered one-layout oracle-observation development demo from
`belief_active_vision_protocol_v1_addendum.md`. It is not an RGB result, a
12-layout development result, or evidence for a particle method.

- implementation HEAD: `05e951e0731623674e1db637b32c2c9aefde9319`
- layout: `belief_demo_00`
- hidden state used for the paired policy demonstration: `100`
- primary observation budget: `B=2`
- raw root: `/internsdata/yewenhao/guard_workspace/belief_active_vision_v1/demo_05e951e`
- raw summary SHA-256: `996201f610aaa2d31d38b682585fe84822c650bf6b6545cd3d7c568f3fc34fb9`

## Gate Results

| Gate | Result |
| --- | --- |
| Eight hidden states have byte-identical free `V0` | PASS |
| Three route transfers have the registered physics in all states | PASS, 24/24 |
| Camera queries preserve simulator and RNG state | PASS |
| Adaptive policy uses at most two paid views | PASS |
| Matched fixed policy uses the same two-view budget | PASS |
| Adaptive and fixed reruns are byte-exact | PASS |

Across the 24 canonical route trials, all 12 clear routes completed without a
collision and all 12 blocked routes collided without being counted as a clean
completion.

## Paired Decision Trace

The two policies receive the same free image in hidden state `100`.

- Adaptive: `q_left -> blocked`, `q_right -> clear`, execute `right_route`,
  collision-free completion.
- Fixed: `q_left -> blocked`, `q_front -> unobserved`, stop.

This establishes the requested minimum chain: an observation changes the
candidate set, a second adaptively selected observation certifies an alternate
route, and the controller switches route and completes. The matched fixed
sequence cannot obtain enough evidence under the same budget.

## Interpretation Lock

Route observations in this demo are generated from simulator truth. The result
therefore validates belief update, adaptive query selection, route revision,
broker isolation, and physical execution only. It is an oracle upper bound and
does not validate RGB recognition, cross-layout generalization, ordinary
particles, or learned particle updates. The registered 12-layout development
gate remains unopened.
