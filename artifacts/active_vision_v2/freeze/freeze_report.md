# Phase 5B train/validation freeze

The formal train gate passed all 60 registered layouts. The validation corpus then passed scope, provenance, paired-isolation, file-hash, finite-value, and replay checks for all 20 registered layouts. No test layout was instantiated before this freeze.

The final training run uses 360 train examples and 120 validation examples. Numeric normalization is fit on train only. Candidate-aware validation MSE values are `0.000299`, `0.000541`, and `0.000366` for seeds 1701-1703. Candidate-agnostic values are `0.250176`, `0.250379`, and `0.250249`. All best epochs are the earliest observed minimum and all histories are finite.

The three fixed cameras tie at validation utility `0.4`; the registered tie rule therefore freezes B3 as `v_left`. Six checkpoint hashes, normalization, the canonical manifest, protocol, validation integrity result, and final test analysis/checker code are locked by `freeze.json`.

Three incomplete or invalid training attempts remain outside the repository data artifact and are named in `freeze.json`. None contributed a checkpoint, model choice, hyperparameter change, or test decision. The test split may now be collected exactly once from a clean pushed freeze commit.
