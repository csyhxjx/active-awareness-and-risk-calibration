# Environment Fingerprint

Recorded: 2026-09-15 UTC

## Machine

- Host: `7439302d014b`
- GPU: physical GPU 3, NVIDIA GeForce RTX 3090 (24 GiB)
- NVIDIA driver: 550.78
- Driver CUDA capability: 12.4
- PyTorch CUDA runtime: 12.1
- cuDNN: 8902
- OS kernel: Linux 4.15.0-213-generic x86_64

## Software

- Python: 3.10.21
- torch: 2.2.0
- torchvision: 0.17.0
- torchaudio: 2.2.0
- transformers: 4.40.1 (OpenVLA-OFT fork)
- flash-attn: 2.5.5
- robosuite: 1.4.1
- mujoco: 2.3.7
- libero: 0.1.0, Git `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- openvla-oft: 0.0.1, Git `e4287e94541f459edc4feabc4e181f537cd569a8`
- numpy: 1.26.4
- wandb: 0.13.1

## Paths

- Project root: `/internsdata/yewenhao`
- Conda environment: `/internsdata/yewenhao/miniconda3/envs/project`
- OpenVLA-OFT: `/internsdata/yewenhao/code/openvla-oft`
- LIBERO: `/internsdata/yewenhao/code/LIBERO`
- Guard: `/internsdata/yewenhao/guard`
- Runtime evidence: `/internsdata/yewenhao/guard_workspace`
- Checkpoint: `/internsdata/yewenhao/models/models/openvla-7b-oft-finetuned-libero-spatial`

## Frozen Artifacts

- Pilot manifest SHA-256: `2e3ef3bc7723fb59f823778b6f7201baff94959bb362f392815f0d45c66a7ecc`
- Hugging Face snapshot commit: `6d0231af0e48c5985f1ff86908f4674b84bc049b`

Checkpoint SHA-256:

```text
2809bd7be9422315c5ecbe91eea612f5b02925f37e0051728ad45bc993c79251  model-00001-of-00004.safetensors
a00a7c5f2b6586ccfc89c693a9c36f3552ff455ff2b1bfea3e92642e4cd2b6d3  model-00002-of-00004.safetensors
a894b7230a08b471af55c57dd7385fd3b51fae2c4d9307a36a8017ef57abf22a  model-00003-of-00004.safetensors
a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60  model-00004-of-00004.safetensors
809858636cf0a65009dd567d2f4e116249442790f02b8fe31f24500ea6118908  action_head--150000_checkpoint.pt
438d28e81e125166d0771762424aaf017de3a8daaebde06a5cb71157e62b3bf3  proprio_projector--150000_checkpoint.pt
4bd2e808805f9b67af090c37e70f239b3b4da7a6473ec621ead6702b16a302ec  lora_adapter/adapter_model.safetensors
```

## Compatibility Evidence

- Official LIBERO Spatial smoke evaluation on physical GPU 3: 20/20 success.
- Scene probe fixture: `data/scene_probe/company_server_20260915/`.
- All 10 scenes contain `robot0_*`, `gripper0_*`, exactly one `gripper0_grip_site`, and 5 free joints.
- No old-server probe fixture was available for a field-by-field diff.
- Guard parity used physical GPU 4 for off and physical GPU 5 for on; both are RTX 3090 cards on this machine.
- Guard off: 10/10 success. Guard on: 10/10 success.
- Parity: 132/132 chunks and 1014/1014 steps allclose, no missing/extra records, no episode diffs.
- Both constraint recordings contain 10 successful episodes and zero onsets.

Runtime evidence SHA-256:

```text
5714ca7288b411fb53b21653eb4ba4f5a6823e18b674b3b6c13e90e27c18c8c8  runs/official_actions_monitored_srv2.jsonl
5714ca7288b411fb53b21653eb4ba4f5a6823e18b674b3b6c13e90e27c18c8c8  runs/guard_actions_monitored_srv2.jsonl
95cca87ecec143c555fd6b3c2926a146349907e0a8588dafa08af5776a07a209  runs/constraints_off_srv2.jsonl
95cca87ecec143c555fd6b3c2926a146349907e0a8588dafa08af5776a07a209  runs/constraints_on_srv2.jsonl
68347a39e0277415443d9283353b68f0929bcf7a87733207e7b170e01aee6f04  logs/EVAL-libero_spatial-openvla-2026_09_15-01_41_08--parity_off_srv2.txt
68347a39e0277415443d9283353b68f0929bcf7a87733207e7b170e01aee6f04  logs/EVAL-libero_spatial-openvla-2026_09_15-01_41_08--parity_on_srv2.txt
```

## Phase 3 Collection-Shell Evidence

- Collection shell commit: `e28c510`.
- Stable state-hash fix: `02f009a`.
- Post-change parity ran Guard off on physical GPU 4 and Guard on on physical GPU 5.
- Both parity runs succeeded 10/10; 132/132 chunks and 1014/1014 steps match with no episode differences.
- Post-change action recordings are byte-identical to each other and to the pre-change `cd4e5bc` recording.
- Mini collection ran five frozen train states on physical GPU 6 with image window `[10,230)` and succeeded 5/5.
- All 475 action/image steps align with full-episode constraints; 950 full/wrist PNGs passed integrity checks.
- Replaying the official preprocessing path for step 10 of all five states reproduced both persisted images byte-for-byte as arrays.
- Collected files total 51,174,114 bytes. Mean episode size is 10,234,822.8 bytes, so `mean x 70 x 1.5` is 1,074,656,394 bytes (1.001 GiB). This estimate is based on five successful episodes averaging 95 persisted image steps. If all 70 episodes instead reach the 220-image-step timeout, the same estimate scales to about 2.5 GB; storage remains negligible.
- Available data-disk space at the gate was 8,381,061,267,456 bytes.
- Mini collection tree SHA-256 is `0f91761f3910fb5df7ca7eb8fbcac2241aed7765eab53aedb895d2a7b033e309`, computed over the sorted per-file SHA-256 listing.

Phase 3 runtime evidence SHA-256:

```text
5714ca7288b411fb53b21653eb4ba4f5a6823e18b674b3b6c13e90e27c18c8c8  runs/phase3_off_actions_e28c510.jsonl
5714ca7288b411fb53b21653eb4ba4f5a6823e18b674b3b6c13e90e27c18c8c8  runs/phase3_on_actions_e28c510.jsonl
95cca87ecec143c555fd6b3c2926a146349907e0a8588dafa08af5776a07a209  runs/phase3_off_constraints_e28c510.jsonl
95cca87ecec143c555fd6b3c2926a146349907e0a8588dafa08af5776a07a209  runs/phase3_on_constraints_e28c510.jsonl
f6c587d6fb5c5e16dbade33ef22a1d3ea7a07a1da34b51f40c19968f0fac5f9f  runs/mini5_actions_02f009a.jsonl
29a61da47f47d6f6d153bee7c9573c473b666bc73cdd438603bcd7bb0de7fdb8  runs/mini5_constraints_02f009a.jsonl
e5fa9f1d6689210376953bd44f2d38bdaccc2bb1d8c2eab124b441a7e761bf5b  logs/EVAL-libero_spatial-openvla-2026_09_15-02_16_27--phase3_parity_off_e28c510.txt
e5fa9f1d6689210376953bd44f2d38bdaccc2bb1d8c2eab124b441a7e761bf5b  logs/EVAL-libero_spatial-openvla-2026_09_15-02_16_27--phase3_parity_on_e28c510.txt
f7350b5446bc2991ab5cda43693f05ff0117792761bc392ed79aa286fa0d3772  logs/EVAL-libero_spatial-openvla-2026_09_15-03_08_17--phase3_mini5_02f009a.txt
```

## Full Train/Calibration Collection

- Full collection commit: `fe76a6c85aa06471d11ee39cdabf931710a2bc76`.
- Invocation selected `train,calibration` only: exactly 50 train and 20 calibration states. Test states were not selected and no test directory exists.
- One continuous run used physical GPU 4, clean Guard HEAD, image window `[10,230)`, and frozen manifest SHA-256 `2e3ef3bc7723fb59f823778b6f7201baff94959bb362f392815f0d45c66a7ecc`.
- All 70 state directories passed the integrity checker. There are 7,950 full-episode constraint steps and 7,250 action/image steps, with 14,500 full/wrist PNGs.
- Result: 68/70 success (97.1%); 2/70 reached `max_steps`: `task_05_init_034` and `task_07_init_043`. Both have complete artifacts and 230 constraints plus 220 image steps.
- Collection file content totals 780,745,781 bytes; filesystem allocation is 781,929,525 bytes. The largest episode is `task_05_init_034` at 24,393,671 bytes.
- The success-length projection from mini was 1,074,656,394 bytes (`mean x 70 x 1.5`); the conservative all-timeout projection is about 2.5 GB. Both are far below the measured 8.38 TB free space.
- A full collection scan found no `__numpy__` or base64 payloads in per-state constraints or the run-level constraint JSONL.
- Full collection tree SHA-256 is `202479aae82fab7a5c54a80de9106116dc8e6c4e564f480cce2400016d491303`, computed over the sorted per-file SHA-256 listing.

Full-run runtime evidence SHA-256:

```text
ae4a378ac73dd6de0c701c4b9111b4e5a0c6982ab45cf6eb1cbb3dc2e59566ea  runs/full70_actions_fe76a6c.jsonl
0c53768171c655e8a60f956c40199f09870c3d976808ffab0fa4c8159426c943  runs/full70_constraints_fe76a6c.jsonl
7dec1a3601bdae5d9911e05b1c9c8f3fe7137c2b5e7804c29ba6a3101c5f0641  logs/EVAL-libero_spatial-openvla-2026_09_15-03_33_30--phase3_full70_fe76a6c.txt
```
