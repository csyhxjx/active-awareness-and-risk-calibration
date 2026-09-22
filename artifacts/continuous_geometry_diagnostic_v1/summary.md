# Phase 6C geometry diagnostic summary

Fixed development grid cases: `72`.

## Route results

| route | physical strata | nearest components | contacts | max path deviation mean (m) |
|---|---|---|---|---:|
| center_route | `{'blocked': 22, 'safe': 2}` | `{'gripper': 22, 'none': 2}` | `{'route_obstacle': 22}` | 0.03944360627311101 |
| left_route | `{'blocked': 20, 'safe': 4}` | `{'gripper': 20, 'none': 4}` | `{'route_obstacle': 20}` | 0.1778024740493566 |
| right_route | `{'blocked': 21, 'safe': 3}` | `{'gripper': 21, 'none': 3}` | `{'route_obstacle': 21}` | 0.17788653174629399 |

All three strata in each route: **False**.
Boundary cases present: **False**.

## Interpretation

- All measured nearest robot-obstacle pairs are gripper components; no wrist or arm pair is the global minimum in this fixed grid.
- The 9 cases without a proximity pair only establish a distance greater than the registered proximity radius; they are not silently assigned an exact minimum.
- The empty-scene sweep is a comparator only. It is not promoted to physical truth because obstacle-induced controller deviations are recorded.
- This conditional diagnostic scan is not a prior draw or method comparison.

No new formal probe or belief-method analysis is authorized by this scan.
