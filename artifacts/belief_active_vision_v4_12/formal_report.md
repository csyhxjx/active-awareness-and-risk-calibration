# Phase 6A7 ROI-Contract Twelve-Layout Result

The seed-66212 development run passes every registered infrastructure hard
gate. Scientific comparison is therefore authorized for the six main states;
the `000/111` controls remain separate.

## Hard gates

| Gate | Result |
| --- | --- |
| ROI detector semantics and registered-geom isolation | PASS, 96/96 scenes |
| Hidden/public isolation (`v0` identical across eight states per layout) | PASS, 12/12 layouts |
| Route physics and formal/preflight trajectory parity | PASS, 288/288 routes |
| Clear-route active-obstacle clearance >= 0.004 m | PASS |
| Independent-process replay | PASS, 96/96 scenes |
| All 17 fixed B=2 sequences | PASS, 12/12 layouts |
| Required provenance and archived trajectory/image hashes | PASS |

The formal root contains 960 PNG files (full RGB plus cue ROI for five cameras
in 96 scenes), 288 route trajectory files, and 96 independently generated
replay fingerprints. The complete root has 1,361 files. The minimum non-null
clear-route preflight clearance is `0.03767778545320298 m`; all-absent controls
correctly use null rather than a fictitious obstacle distance.

## Main-state comparison

| Metric | Adaptive exact | Best fixed B=2 envelope | Difference (adaptive - fixed) |
| --- | ---: | ---: | ---: |
| Collision-free completion | 72/72 (100%) | 48/72 (66.7%) | +33.3 pp |
| Collision | 0/72 | 0/72 | 0 pp |
| Stop | 0/72 | 24/72 (33.3%) | -33.3 pp |
| Mean queries | 2.000 | 1.667 | +0.333 |
| Mean utility | 0.900 | 0.500 | +0.400 |

Every layout has the same result: adaptive completes `6/6`, while the exhaustive
fixed-sequence envelope completes `4/6`. Consequently, the deterministic
10,000-replicate layout bootstrap (seed `66212`) has degenerate 95% intervals:
completion difference `[+0.3333,+0.3333]`, collision difference `[0,0]`, stop
difference `[-0.3333,-0.3333]`, query difference `[+0.3333,+0.3333]`, and
utility difference `[+0.4,+0.4]`.

“Best fixed” is the maximum expected result over all 17 preregistered fixed
sequences under the frozen six-state prior and observation contract, not a
per-hidden-state choice. Four sequences tie at the envelope; `q_b -> q_a` is
only the checker's representative.

## Controls and claim boundary

Across the 24 control scenes, all 36 clear routes complete collision-free and
all 36 blocked routes contact the registered obstacle. Controls are excluded
from the adaptive-gain denominator as preregistered.

This establishes adaptive view-selection gain in the controlled, exact-belief,
ROI color-cue mechanism fixture. It is not evidence for a learned RGB observer,
natural-RGB generalization, continuous hidden geometry, a particle filter, or a
learned particle network. Those stages remain frozen pending a separate RGB
amendment.

Raw summary SHA-256:
`3fc1e2863488172675ae8f7c6327994a2a91e9970e12020bde107b34641d4925`.
Hard-gate checker SHA-256:
`b978c118409ef7c1c913885cd3ada748508573b9c2b72558aca21c499ce540be`.
