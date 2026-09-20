# Phase 5D proposer smoke report

Date: 2026-09-20  
Protocol: `active_vision_protocol_v3.md`  
Manifest: `configs/active_vision_v3/manifest.json`  
Manifest SHA-256: `82b2927b6bc55048753b95a38a8eecbb8b1fd5d6510e8ebf06e90f2e1f15b517`

## Scope

This is the registered 8-layout, train-only smoke: four hidden states, two route preferences, and 64 trials. The Phase 5B 40-layout test namespace was not reused. No selector was trained and no validation/test rollout was started.

## D0-D5 result

| Gate | Result | Evidence |
| --- | --- | --- |
| D0 protocol/manifest scope | PASS | 8 fresh train layouts, seed 54040, families F0-F3 (2 each) |
| D1 exact replay | PASS | 64/64 replay records exact |
| D2 non-stop route mapping | FAIL | 0/64 valid `left_route`/`right_route` mappings; required >= 60% |
| D3 candidate recall | PASS | 0 candidate-recall failures |
| D4 route transfer | NOT APPLICABLE | No non-stop candidates were produced |
| D5 provenance/completeness | PASS | finite tuples, frozen checkpoint and normalization key recorded |

The checker result is stored in `zero_shot_check.json`. Its `all_pass` value is false because D2 fails. All 64 OpenVLA chunks were finite and shape-valid, but the deterministic projection mapped every chunk to `stop`; `stop` is intentionally excluded from the D2 denominator.

## Proposer provenance

- checkpoint: `/internsdata/yewenhao/models/models/openvla-7b-oft-finetuned-libero-spatial`
- checkpoint tree SHA-256: `b88fb9250727ebe49e64d56b8b3355e47b4f543a1989fd960a68aed8c209ced5`
- normalization key: `libero_spatial_no_noops`
- device: NVIDIA GeForce RTX 3090 (`CUDA_VISIBLE_DEVICES=3`)
- batch size: 1; sampling: disabled; chunk: 8 x 7

The raw smoke payload remains at `/internsdata/yewenhao/guard_workspace/active_vision_v3/zero_shot_41357fb/smoke.json` and is outside the repository artifact tree because it contains the full rendered trial payload.

## Stop decision

The protocol permits one nominal-only LoRA repair when D2 fails. No nominal-only training dataset is present in the configured local dataset roots, so that repair cannot be run without inventing data or broadening the preregistered scope. Phase 5D therefore stops at the proposer smoke failure. There is no selector training, formal validation, held-out test, or claim about OpenVLA improvement.
