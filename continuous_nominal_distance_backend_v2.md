# Phase 6C Nominal Distance Backend Amendment v2

Date: 2026-09-22. Status: prospective calibration backend only. The first
nominal scan attempt is an infrastructure failure: MuJoCo 2.3.7 proximity
contacts become discontinuous when the nearest collision pair changes (right
route at offsets `0.085/0.090/0.095 m`: approximately
`-0.1118/-0.0244/-0.0199 m`). Partial left (`204/276`) and complete center
(`276/276`) files are diagnostic only; right wrote zero cases. They cannot be
combined, classified as a completed grid, or used to open a probe.

Use the original project environment only to record the obstacle-free nominal
qpos trajectories and export the compiled scene XML/assets. Evaluate signed
distance in a separate, pinned MuJoCo `3.13.0` process via the public
`mj_geomDistance` API with native CCD enabled. That process performs no
dynamics or controller calls: it loads the exact exported robot/scene geometry,
sets the registered qpos/interpolants and candidate obstacle body pose, calls
`mj_forward`, and minimizes `mj_geomDistance(..., distmax=0.25)` over every
robot-collision-geom / target-obstacle-geom pair. It records the pair and
closest-point segment. A return equal to `distmax` is right-censored.

Before the full grid, the backend must pass:

1. XML and nominal qpos hashes match the export manifest;
2. distances at offsets `0.085,0.090,0.095 m` are finite and adjacent changes
   are <= `0.010 m` on all three routes;
3. exact repeated subprocess output is byte-identical;
4. penetrating/nonpenetrating sign agrees with 2.3.7 contacts on cases where
   the old engine reports a stable contact;
5. MuJoCo version and native-CCD flag are recorded.

Only then rerun the already frozen 828-case calibration grid from scratch in a
new `continuous_nominal_calibration_v2` namespace. The existing 8-vs-16
interpolation, `0.0005 m` refinement, and fast-risk-model `0.001 m` maximum
error/zero-side-flip gates remain unchanged. No 24-world probe is authorized
by this backend amendment.
