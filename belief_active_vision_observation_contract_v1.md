# Phase 6A6: Observation Contract Redefinition and Minimal Probe

Date: 2026-09-21. This is a design note and repair probe, not a 12-layout
amendment. Phase 6A4 and 6A5 outputs remain immutable failed infrastructure
roots; their symbolic gains are diagnostic only and excluded from claims.

Full RGB images remain archived, but whole-image hash equality is removed from
the mechanism contract. Each paid camera has a registered cue ROI and frozen
detector output. Same-outcome ROI semantics must agree; different outcomes must
be separable. Pixels outside the ROI are nuisance variation and cannot be used
by the mechanism policy. If hidden geometry enters the ROI, the probe fails.

The one-layout repair probe contains six main states, four paid views, all
three routes, the exact belief planner, official-controller clear-route
preflight, and independent Python-process replay. Clear routes require at least
`0.004 m` minimum clearance, no obstacle contact, nearest-obstacle id,
trajectory hash, and trajectory length. Failed candidates are rejected before
policy comparison, never relabelled.

Missing or mismatched ROI output, preflight, replay, HEAD, manifest hash, or
checker version makes the probe provisional. Only after this probe passes may a
new 12-layout amendment be drafted. RGB, continuous geometry, particle filters,
and learned particle updates remain frozen.
