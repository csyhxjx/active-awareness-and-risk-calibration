# Calibration acceptance supplement (2026-09-22)

This is a supplemental audit of immutable export_v2c/grid_v2c at source HEAD
82bc78a, not a new probe. Original evidence is bound by the pre-edit SHA-256
inventory in acceptance_v1/original_evidence_manifest.json.

Before running supplements, freeze these choices:

- Representative replay: first case index in each route x safe/boundary/blocked
  stratum; two fresh worker processes, exact raw-byte equality; compare original
  clearance and step minima. No replacement of a failing representative.
- Refinement: all original 292 cases with abs(c8 - .004) <= .010, at 16 intervals.
  Save full output, require <=.0005 m error, equal penetration (c<0) and feasibility
  (c>=.004) booleans. Equality is checked explicitly. Per-route/width/sign reports.
- Empty motion: all three frozen qpos sequences, 16 intervals, all obstacles parked;
  retain penetrating contacts with their names; no new contact exclusions. Collision
  filtering is the original compiled model's masks/exclusions, recorded with hashes.
- Contact attribution: retain the disputed left/-.100 case and all final +/- .080
  cases. For left -.100, -.080 and +.080, use the fixed local neighborhood
  offset +/- .001 m in 2.3.7, and repeat the center in a fresh process. Compare the
  old engine alone before interpreting new-engine signs. Failure to establish an
  objective old-engine defect leaves the original exclusion unsupported.
- Check exported vs original compiled geometry/poses before interpreting sign
  differences. Preserve all attempts, including mismatches.
- The .25 m query limit is a right-censored lower bound; never emit an uninitialized
  closest-point segment for a censored query. No changed numerical distance algorithm.

An audit failure closes admission, not the audit: finish independent evidence and
report PASS/FAIL/MISSING. Changes to geometry or labels require a new result root;
old outputs are never rewritten. Fast-risk accuracy, new worlds and policy methods
are outside this supplement.
