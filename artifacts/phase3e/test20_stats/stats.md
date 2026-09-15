# Offline Label Statistics

- Labels: `labels_test20_c91e126.jsonl` (100 rows)
- Collection: `phase3_test20_c91e126` (read-only input)
- A violation is the recorded boolean flag; continuous signed margins use values below zero.
- `sustained_violation` means at least three consecutive violating steps unless configured otherwise.
- Case 0 calibration: `tau=0.002` is a penetration-axis budget; the margin-axis decision boundary is `0`.
- The full70 warning list is empty; no threshold change is applied.

## Constraint x Split

| Split | Constraint | States | Any violation | Sustained | Min margin median | Worst min margin | Flag/margin mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| test | workspace | 20 | 0 | 0 | 0.0419236 | 0.0323929 | 0 |
| test | gripper_env | 20 | 0 | 0 | 0.002 | 0.000862289 | 0 |
| test | self_collision | 20 | 0 | 0 | 0 | 0 | 0 |
| test | object_drop | 20 | 0 | 0 | 0.0261759 | 0.0249638 | 0 |
| test | non_finite | 20 | 0 | 0 | 0 | 0 | 0 |

## Warning List

- None.

## Integrity

- Unique states: `20`
- Constraints per state: `5`
- Non-finite margins: `0`
