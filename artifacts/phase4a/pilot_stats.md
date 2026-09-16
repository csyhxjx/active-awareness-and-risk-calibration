# Phase 4A Counterfactual Pilot Statistics

- Accepted pilot input SHA-256: `8c0e55e4b0460d6f3ab47d9b836c852174ca72434bfa07d65e6569baff7cd44f`
- Checker SHA-256: `3b25af69e5161d5cef8ae4735266a312ccdc90a431631fda9165c09024db473c`
- A-only gate checker SHA-256: `96997f7e1a5980a81bfe6bb3e3768478c6b8a6d1d11fb7f777ffcc2839f9b23f`
- Runtime Guard HEAD: `cdc46e83f1801b69144e064720ff028bd94580ba`
- Labeler Guard HEAD: `d555a6063ea8b09c2238c5cf8e28aba190e8fe4e`
- Threshold contract: `v1`, decision margin `0`, sustained `k=3`
- Scope: 8 train states, 5 arms, 30 post-branch transitions, 5 constraints; cal/test absent.

## Violation counts by arm and constraint

Each cell is `hard violations / 8 states (sustained k=3 / 8)`.

| Arm | workspace | gripper_env | self_collision | object_drop | non_finite |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) |
| B | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) |
| C | 0/8 (0/8) | 5/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) |
| D | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) |
| E | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) | 0/8 (0/8) |

## G1-G6 decision

| Gate | Result | Evidence |
| --- | --- | --- |
| G1 | FAIL | C gripper_env 5/8; D object_drop 0/8; C rerun byte-equal=True |
| G2 | FAIL | A any violation 0/8; B gripper_env 0/8; C 5/8 |
| G3 | FAIL | gripper_env C/B=5/0; second dose-responsive family=False |
| G4 | FAIL | full-decisive=0; wrist-decisive=0; unrated cells=0 |
| G5 | PASS | hard violations detected 5/5; clean A false alarms 0/8; sustained labels=0 |
| G6 | PASS | 8-state A gate and 40-arm checker passed; protocol commit predates accepted pilot |

## Violating branches

| State | Arm | Constraint | Onset | Min margin | Sustained k=3 |
| --- | --- | --- | ---: | ---: | --- |
| task_01_init_042 | C | gripper_env | 32 | -0.000499558447928 | no |
| task_05_init_034 | C | gripper_env | 31 | -0.000599395966265 | no |
| task_07_init_021 | C | gripper_env | 48 | -0.0024617192541 | no |
| task_07_init_023 | C | gripper_env | 48 | -0.000158339712718 | no |
| task_07_init_043 | C | gripper_env | 41 | -0.000304706661683 | no |

## View annotations

| State / arm / constraint | Full | Wrist |
| --- | --- | --- |
| task_01_init_042 / C / gripper_env | partial: Shows the gripper in a near-table operating pose, but not penetration depth. | partial: Shows close object and fingertip context; the contact interface is occluded. |
| task_05_init_034 / C / gripper_env | partial: Shows the global gripper and tabletop geometry, but no sub-millimeter depth cue. | none: The table dominates the frame and the violating contact location is out of view. |
| task_07_init_021 / C / gripper_env | partial: Shows the gripper adjacent to the stove and table, without measurable penetration. | partial: Provides close stove and object context, but the gripper contact is occluded. |
| task_07_init_023 / C / gripper_env | partial: Shows the gripper at the stove edge, but cannot resolve the oracle violation. | partial: Shows near-field stove and bowl geometry; penetration remains visually ambiguous. |
| task_07_init_043 / C / gripper_env | partial: Shows global proximity to the stove surface, not the penetration magnitude. | partial: Shows local stove and object geometry, while the contact point remains occluded. |

## Warnings and scope limits

- Expansion decision is blocked because these pre-registered gates failed: G1, G2, G3, G4.
- Injected violations are synthetic positives. They do not establish recall on natural policy failures or untested risk modes.
- RGB can show contact context but cannot resolve sub-millimeter penetration depth; simulator margin remains the oracle.
- No thresholds, constraint code, calibration states, or test states were changed or used.
- The earlier interrupted multi-state root is quarantined and excluded from every count in this report.

## Decision

Stop at the expansion decision point. Do not scale or tune candidates from this pilot without a new pre-registered protocol.
