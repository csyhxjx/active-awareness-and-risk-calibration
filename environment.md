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

- Official LIBERO Spatial smoke evaluation: 20/20 success on this machine.
- Scene probe fixture: `data/scene_probe/company_server_20260915/`.
- All 10 scenes contain `robot0_*`, `gripper0_*`, exactly one `gripper0_grip_site`, and 5 free joints.
