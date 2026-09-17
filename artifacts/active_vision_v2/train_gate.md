# Phase 5B formal train gate

Date: 2026-09-17. Result: **PASS**.

The first formal stage ran only the 60 registered train layouts from Guard HEAD `f2a1e76c65fba39cb88a00cdb9d9bb1f2932380e`. It produced 240 scenes, 480 canonical candidate trajectories, 60 representative replays, 960 RGB PNGs, and 240 each of route, query, and metadata records. No validation or test layout directory was created.

| Gate | Result | Evidence |
| --- | --- | --- |
| T0 scope | PASS | 60 layouts, 240 scenes, 480 trajectories, 60 replays; validation/test outputs 0 |
| T1 provenance | PASS | clean pushed HEAD; query state hashes invariant; 60/60 replay records byte-identical |
| T2 hard integrity | PASS | 60/60 layouts; all files/hashes/schema/finite values and paired V0/public inputs valid |
| T3 physical mechanism | PASS | 60/60 layouts, required 54 |
| T4 paid-view mechanism | PASS | 60/60 layouts, required 54; left-unique 30 and right-unique 30, required 20 each |
| T5 accounting | PASS | all 60 registered layouts retained; physical and paid-view failure lists empty |

Provenance:

- manifest SHA-256: `22bcbe02f21fc29a798b787cc958bf7f5bfc87275abb7e9cb25d253fef862bdf`
- protocol SHA-256 at collection: `be64587388fb72830b5fe7db8e28f0f54788b53cd588951c6a29b17ef3e23c4f`
- raw summary SHA-256: `74c043e1b590be8d89b34bfba22a1469073f1133ed59887662c760bd874136f6`
- raw checker SHA-256: `8199d18f115e064f5ceae33e3ead388108ae1db89a0d7ca497c36756c1ea4f8c`
- runtime root: `guard_workspace/active_vision_v2/train_f2a1e76`

This PASS opens registered training and validation only. The 40-layout test split remains sealed until selected checkpoint hashes, train normalization statistics, B3 camera, and final analysis code are frozen and pushed.
