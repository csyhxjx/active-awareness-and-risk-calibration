# Phase 3E Offline Label Statistics

- Labels: `labels_full70_fe76a6c.jsonl` (350 rows)
- Collection: `phase3_full70_fe76a6c` (read-only input)
- A violation is the recorded boolean flag; continuous signed margins use values below zero.
- `sustained_violation` means at least three consecutive violating steps unless configured otherwise.
- Case 0 calibration: `tau=0.002` is a penetration-axis budget; the margin-axis decision boundary is `0`.
- The full70 warning list is empty; no threshold change is applied.

## Constraint x Split

| Split | Constraint | States | Any violation | Sustained | Min margin median | Worst min margin | Flag/margin mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| train | workspace | 50 | 0 | 0 | 0.047818 | 0.0330335 | 0 |
| train | gripper_env | 50 | 0 | 0 | 0.002 | 0.00102618 | 0 |
| train | self_collision | 50 | 0 | 0 | 0 | 0 | 0 |
| train | object_drop | 50 | 0 | 0 | 0.0263568 | 0.0256043 | 0 |
| train | non_finite | 50 | 0 | 0 | 0 | 0 | 0 |
| calibration | workspace | 20 | 0 | 0 | 0.0486318 | 0.031743 | 0 |
| calibration | gripper_env | 20 | 0 | 0 | 0.002 | 0.00112182 | 0 |
| calibration | self_collision | 20 | 0 | 0 | 0 | 0 | 0 |
| calibration | object_drop | 20 | 0 | 0 | 0.0264556 | 0.0252052 | 0 |
| calibration | non_finite | 20 | 0 | 0 | 0 | 0 | 0 |

## Warning List

- None.

## Integrity

- Unique states: `70`
- Constraints per state: `5`
- Non-finite margins: `0`
