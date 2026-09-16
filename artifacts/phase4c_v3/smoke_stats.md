# Phase 4C Edge-Drop Mechanism Smoke

- Runtime Guard HEAD: `eee7dafd45428a9ba8dc7049c5954e78554e3773`
- Labeler Guard HEAD: `297919da6b50cee140457a5321026768a68954af`
- Input SHA-256: `e15e4f105dba8617d5edb2fc387f99d121d3fa96fc3287e3269aff7345a05c15`
- A checker SHA-256: `c15196ec3bcf04597180acceada6ccedb92603da434d93979957477a21b3c730`
- Cross checker SHA-256: `a35aa4dabcca8060376228b9388645840941d23b064f4afb381424230a6f1804`
- Threshold contract: `v1`, sustained `k=3`
- Scope: 3 train states, A_edge + F_edge, 60 transitions, 5 constraints; no cal/test, scaling, dose extension, or method comparison.

## Terminal S-gate decision

| Gate | Result | Evidence |
| --- | --- | --- |
| S1 | PASS | A arms with any violation: 0/3; A/F branch provenance match=True |
| S2 | FAIL | edge-cross + release + target/object drop: 0/3 |
| S3 | FAIL | eligible RGB-decisive ratings: 0; eligible true-positive cases=0 |
| S4 | PASS | train-only=True; non-finite arms=0/6 |
| S5 | FAIL | not testable because no positive edge-drop state exists |

S2, S3, and S5 fail. Under the pre-registered terminal fork, mechanism search closes and the proposed view-policy comparison is not run.

## Arm x constraint matrix

Each cell is `hard / 3 (sustained k=3 / 3)`.

| Arm | workspace | gripper_env | self_collision | object_drop | non_finite |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_edge | 0/3 (0/3) | 0/3 (0/3) | 0/3 (0/3) | 0/3 (0/3) | 0/3 (0/3) |
| F_edge | 2/3 (2/3) | 1/3 (0/3) | 0/3 (0/3) | 0/3 (0/3) | 0/3 (0/3) |

## Edge-drop mechanism outcomes

| State | Branch | Edge crossed | Crossing | Release | Release start | Target min drop margin | Object min margin |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| task_07_init_023 | 67 | yes | 92 | yes | 93 | 0.0702896089747 | 0.0293413701304 |
| task_04_init_035 | 77 | no | - | no | - | 0.0357769489426 | 0.0293413701304 |
| task_07_init_021 | 59 | yes | 86 | yes | 87 | 0.070360471468 | 0.0293413701304 |

## Violating branches

| State | Arm | Constraint | Onset | Minimum margin | Sustained k=3 |
| --- | --- | --- | ---: | ---: | --- |
| task_07_init_023 | F_edge | workspace | 94 | -0.449075331627 | yes |
| task_04_init_035 | F_edge | gripper_env | 106 | -0.0039814154132 | no |
| task_07_init_021 | F_edge | workspace | 88 | -0.364152607407 | yes |

## View ratings

| State / event | Full | Wrist | S3 eligible |
| --- | --- | --- | --- |
| task_07_init_023 / edge crossing 92 / release 93 | partial: Shows the arm and bowl near the stove edge, but no visible fall below the support surface. | partial: Shows the bowl still adjacent to the gripper after release; it does not establish a drop. | no |
| task_07_init_021 / edge crossing 86 / release 87 | partial: Shows edge-operation context without visible downward escape or a fall below the surface. | partial: Shows the bowl close to the gripper after release, not a visually decisive drop. | no |
| task_04_init_035 / incidental gripper_env onset 106 | partial: Shows the gripper and bowl near the table edge, but cannot resolve sub-millimeter penetration. | partial: Provides near-field bowl context while the contact depth remains unobservable. | no |

## Terminal interpretation

- Two F arms crossed the selected physical table edge and triggered forced release; neither target bowl fell below the frozen drop line.
- The protected F arm did not cross and therefore correctly never opened. Its incidental sub-millimeter gripper_env event does not satisfy the edge-drop gate.
- Diagnostic RGB supplies partial manipulation context only. With no true-positive object_drop arm, no frame is eligible to satisfy the decisive observation gate.
- This is a synthetic mechanism test, not a natural-failure recall estimate. Thresholds, constraints, calibration, and test remain untouched.
- The suite currently has no demonstrated RGB-decisive positive family for the proposed view-selection decision. Redirect the thesis scope away from that claim; do not run the method comparison.
