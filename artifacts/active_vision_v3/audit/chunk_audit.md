# Phase 5D read-only chunk audit

This audit reads only the 64 frozen zero-shot trials. It does not alter the Phase 5D projection, labels, or terminal result: non-stop coverage remains `0/64` under the frozen projection.

## Numeric result

- All chunks are `8 x 7`, finite, and bounded in `[-0.170238, 0.346054]`.
- Global mean is `0.006169`; standard deviation is `0.075431`.
- `84.60%` of values have absolute magnitude at most `0.1`; `96.65%` are at most `0.2`.
- Frozen projected terminal positions occupy a narrow region: x `[-0.000848, 0.012423]`, y `[0.018482, 0.023155]`, z `[1.015933, 1.017303]`.
- Left-route distances are `0.194664..0.253539` m; right-route distances are `0.200659..0.255535` m. Every nearest distance exceeds the frozen `0.12` m stop radius, so all 64 stop reasons are `proposer_stop`.
- Changing the route instruction changes the chunk in all `32/32` matched layout/hidden-state pairs. Paired L2 differences are `0.069569..0.087999` (mean `0.078809`).
- For each identical V0 and instruction, all four hidden states produce one unique chunk hash: `16/16` V0-instruction groups are byte-identical across hidden states.

The complete per-trial min/max/mean/std, terminal position, route distances, and stop reason are in `chunk_audit.json`.

## Interpretation boundary

The output is concentrated near zero but is not a constant or instruction-insensitive collapse. The local OpenVLA implementation explicitly documents the continuous output as end-effector deltas, while the frozen 5D projection treated the last delta row as a normalized absolute table-frame waypoint. The evidence therefore supports a delta/absolute semantic mismatch more strongly than task-OOD collapse. This diagnostic does not retroactively change 5D: under its frozen projection, coverage is still `0/64`, and no alternate projection result is computed or substituted.
