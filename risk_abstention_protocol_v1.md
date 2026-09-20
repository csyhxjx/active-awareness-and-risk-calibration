# Phase 5F: Risk-Controlled Abstention Protocol

Date: 2026-09-20. Status: preregistration only. No Phase 5F implementation, collection, training, validation, or test exists at this commit.

## 1. Scientific question and exclusions

Phase 5F asks whether one paid, candidate-conditioned observation can control collision risk at a fixed abstention operating point while retaining safe executions. It does not test completion gain, alternate-route recovery, OpenVLA, or B=2. Phase 5E selectors and its seed-57040 test are frozen and prohibited from tuning, training, threshold selection, power estimation, or evaluation. OpenVLA B/C remain separate paused protocols.

The route proposer and fixed routes retain the Phase 5E schema, but all layouts, rendered data, model fitting, and evaluation use a new `av5f_` namespace. The policy receives V0, instruction, public route metadata, and exactly one purchased candidate-conditioned RGB view. It outputs calibrated collision probability `p_collision`; it executes iff `p_collision <= theta`, otherwise abstains. It cannot inspect or execute the alternate route.

## 2. Prospective data sequence

- Contract/smoke: 8 train-only layout groups, seed `61008`, scene seed `20260922`.
- Formal train: 60 grouped layouts, seed `61060`.
- Validation: 20 grouped layouts, seed `62020`.
- Sealed test: 60 grouped layouts, seed `63060`.

Each layout contains all four hidden states and both public route preferences; derivatives never cross splits. Every ID and sampled geometry must be disjoint from Phases 5A-5E. Test is instantiated once after code, model hashes, camera rule, calibration rule, and evaluator are committed and pushed.

## 3. Models and baselines

The candidate-aware risk model uses the purchased RGB plus public candidate geometry. The candidate-agnostic ablation zeros candidate identity, preference, waypoints, and candidate-camera projection after normalization. Baselines are always-stop, blind-execute, validation-frozen best fixed view, registered random view, and candidate-aware geometric view. Architecture, loss, calibration method, optimizer, seeds, and camera tie-breaks must be added in a prospective amendment before implementation.

No hidden state, collision oracle, future frame, paid-but-unselected view, filename, or old test data is an inference input. Collision labels may train the risk head only on the new formal train split. Validation alone selects checkpoints and `theta`.

## 4. Co-primary endpoints

The operating point is frozen on validation to target execution coverage `q=0.50`; among equal thresholds choose the lower collision estimate, then the more conservative threshold. Test reports the achieved coverage without rethresholding.

The two co-primary test endpoints are:

1. **Risk control:** the one-sided 95% layout-cluster bootstrap upper confidence bound for collision conditional on execution is at most `5%`.
2. **Safe-execution retention:** among oracle-safe proposed routes, the executed fraction is strictly greater than the candidate-agnostic comparator, with a two-sided 95% layout-cluster bootstrap lower bound above zero for the paired difference.

Success requires both endpoints (intersection-union test); no alpha splitting is used because both claims must hold. Collision rate over all trials, abstention rate, collision-free completion, erroneous abstention, calibration error, and observation cost are mandatory secondary endpoints. Always-stop cannot pass safe-execution retention. Completion is not a primary endpoint and cannot overturn either co-primary failure.

## 5. Power and uncertainty

The layout group is the only resampling and power unit. Before formal test opening, a validation-only addendum computes paired cluster SD `s_val` for safe-execution retention and sets `s_plan=min(1,max(0.20,s_val,LOO90(s_val)))`. With two-sided alpha `0.05`, power `0.80`, and minimum detectable paired retention gain `0.10`, required test groups are `ceil(((1.959964+0.841621)*s_plan/0.10)^2)`, capped at 60. If more than 60 are required, the retention claim is declared underpowered and test does not open. Risk-control power is reported by the exact planned execution denominator and the 5% upper-bound criterion; it is not rescued by pooling old data.

## 6. Gates and interpretation

Before training: public-input schema, hidden-state isolation, query state/RNG/time preservation, tuple reconstruction, and fresh-process decision replay must pass. Smoke additionally requires 64/64 complete tuples, zero hidden leakage, correct physical oracle transfer, and finite risk outputs. Any hard failure stops.

If both co-primary endpoints pass, report risk-controlled abstention at B=1. If risk control passes but retention fails, report conservative safety filtering without evidence of improved selective utility. If risk control fails, report no demonstrated risk control regardless of completion or collision-rate point estimates. No branch authorizes a completion-gain claim.

## 7. Permanent boundary

In this two-candidate environment, B=2 checking of the alternate route is a demonstration of a mechanically available recovery path, not an independent experiment. A marginal-value study requires at least three non-equivalent candidate routes or heterogeneous view costs and a separate protocol. Visual-decidability prediction likewise requires its own protocol and labels.
