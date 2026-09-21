# Phase 6A6 ROI Observation Repair Probe

The one-layout, six-state repair probe passes. Full RGB is archived, while only
the registered central cue ROI participates in the mechanism observation
contract.

- ROI semantics: PASS for all four paid views.
- Independent Python-process fingerprint replay: PASS, 6/6 states.
- Official-controller clear-route preflight: PASS.
- Collision-free routes with active obstacles satisfy the frozen `0.004 m`
  minimum clearance; scenes with no active obstacle record clearance as null.
- Every route records completion/collision, trajectory length, minimum clearance,
  and trajectory hash.

Canonical raw summary SHA-256:
`b7991844a4ce152d82dad0f83b9c97bcd3fb68514bed442e38f5525b9128f613`.

This repairs the minimal observation/replay/preflight infrastructure only. It
does not rehabilitate either failed 12-layout root and does not authorize RGB
training, continuous geometry, particle filters, or learned particle updates.
A new prospective 12-layout amendment is still required before expansion.
