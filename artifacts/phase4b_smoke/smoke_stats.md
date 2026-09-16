# Phase 4B Targeted Mechanism Smoke

- Runtime Guard HEAD: `53d1bfa3d33fdaa1ec28839a3861fe91bfc466a4`
- Labeler Guard HEAD: `9dc86f9dd71159f4cfb5cedb295563f5d67358e4`
- Input SHA-256: `d71867fc45173a66b5721232ea1b7e9e176b664bdf21f0237b7a4ce121e9d7e5`
- Threshold contract: `v1`, margin boundary `0`, sustained `k=3`
- Scope: 3 train states, 9 arms, 30 transitions, 5 constraints; no cal/test and no full eight-state v2 run.

## Entry decision

| Gate | Result | Evidence |
| --- | --- | --- |
| S1 | PASS | matched A arms with any hard violation: 0/9 |
| S2 | FAIL | D_force/object_drop: 0/3 |
| S3 | PASS | E_push/workspace: 2/3 |
| S4 | PASS | C04/C06/C08/C10 gripper_env counts: 0/1/1/1; distinct rates=2 |
| S5 | PASS | observed C positive byte-equal=True; observed E positive byte-equal=True |

The full eight-state Phase 4B pilot and all view-policy comparisons remain blocked because every S1-S5 condition was pre-registered as mandatory.

## Target outcomes

| Arm | Target constraint | Hard violations | Sustained k=3 |
| --- | --- | ---: | ---: |
| C04 | gripper_env | 0/3 | 0/3 |
| C06 | gripper_env | 1/3 | 0/3 |
| C08 | gripper_env | 1/3 | 0/3 |
| C10 | gripper_env | 1/3 | 0/3 |
| D_force | object_drop | 0/3 | 0/3 |
| E_push | workspace | 2/3 | 2/3 |

## Violating branches

| State | Arm | Constraint | Onset | Minimum margin | Sustained k=3 |
| --- | --- | --- | ---: | ---: | --- |
| task_04_init_035 | E_push | workspace | 36 | -0.0434038730787 | yes |
| task_07_init_023 | C06 | gripper_env | 48 | -0.00010431735604 | no |
| task_07_init_023 | C08 | gripper_env | 48 | -0.000158339712718 | no |
| task_07_init_023 | C10 | gripper_env | 48 | -0.000158339712718 | no |
| task_08_init_003 | E_push | workspace | 36 | -0.0525318055932 | yes |

## Mechanism diagnosis

- The pressure ladder produced two state-level rates, but plateaued from C06 through C10 in this three-state smoke.
- Persistent lateral pushing produced reproducible workspace violations and supplies a potentially RGB-decisive family for a future protocol.
- Forced opening at maximum payload elevation produced no frozen `object_drop` violation. The released bowl remained supported above the table boundary; release is not relabeled as drop.
- Manual onset-frame review rated both E-positive full views `partial` and both wrist views `none`: the full frames show large lateral displacement but no rendered workspace boundary, while the wrist frames are self-occluded. These are not RGB-decisive cases.
- G1-G6 and view-divergence claims are not evaluated because the smoke entry gate failed before the full pilot.
- Any stronger or combined drop mechanism requires a new v3 pre-registration. Thresholds and constraint code remain frozen.
