# LFG Open Source

[![arXiv](https://img.shields.io/badge/arXiv-2602.22091-b31b1b.svg)](https://arxiv.org/abs/2602.22091) [![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://lfg-ai.github.io/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/AppliedIntuitionResearch/LFG)

- **Paper:** https://arxiv.org/abs/2602.22091
- **Project page:** https://lfg-ai.github.io/
- **Pretrained checkpoint:** https://huggingface.co/AppliedIntuitionResearch/LFG

This repository contains the open-source local inference code path for LFG. It loads a trained `LFG` checkpoint and runs it on either a video file or an ordered sequence of RGB images.

This public version runs local inference only. It loads local checkpoint files and local video/image inputs.

All input and checkpoint arguments must be ordinary filesystem paths on the same machine. URI-style locations are rejected by the CLI.

## Installation

```bash
conda create -n lfg-infer python=3.10 -y
conda activate lfg-infer

# Install PyTorch for your CUDA/CPU platform first if needed:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt

# Optional: install the CLI entry point.
pip install -e .
```

By default the CLI uses `--device auto`, which selects CUDA when a local GPU is available and otherwise falls back to CPU. You can force a device with `--device cuda`, `--device cuda:0`, or `--device cpu`.

## Checkpoint

The pretrained LFG checkpoint is available on Hugging Face at [AppliedIntuitionResearch/LFG](https://huggingface.co/AppliedIntuitionResearch/LFG) (CC BY-NC 4.0). It predicts dense depth / 3D points, camera pose, per-point confidence, object segmentation, and per-pixel motion. Download it locally:

```bash
pip install -U huggingface_hub
hf download AppliedIntuitionResearch/LFG lfg_seg_motion_1.3b.pt --local-dir checkpoints
```

The loader reads the model configuration from checkpoint metadata and state-dict keys, builds the matching model, and loads the weights locally.

## Run On A Video

```bash
python infer.py /path/to/video.mp4 \
  --checkpoint checkpoints/lfg_seg_motion_1.3b.pt \
  --output-dir outputs/video_demo
```

If installed with `pip install -e .`, the same command is available as:

```bash
lfg-infer /path/to/video.mp4 \
  --checkpoint checkpoints/lfg_seg_motion_1.3b.pt \
  --output-dir outputs/video_demo
```

Useful video options:

```bash
python infer.py /path/to/video.mp4 \
  --checkpoint checkpoints/lfg_seg_motion_1.3b.pt \
  --frame-stride 3 \
  --max-frames 120 \
  --window-stride 1 \
  --output-dir outputs/video_dense
```

## Run On Images

Directory input:

```bash
python infer.py /path/to/frames \
  --checkpoint checkpoints/lfg_seg_motion_1.3b.pt \
  --output-dir outputs/frames_demo
```

Glob input:

```bash
python infer.py "/path/to/frames/*.jpg" \
  --checkpoint checkpoints/lfg_seg_motion_1.3b.pt \
  --output-dir outputs/glob_demo
```

Image files are sorted with natural numeric ordering, so `frame_2.jpg` comes before `frame_10.jpg`.

## Outputs

Each model window is written under:

```text
outputs/.../
  run_metadata.json
  window_000000/
    metadata.json
    predictions.npz
    depth/000.png
    confidence/000.png
    segmentation/000.png      # only when the checkpoint has a segmentation head
    motion/000.png            # only when the checkpoint has a motion head
    flow/000.png              # only when the checkpoint has a flow head
```

`predictions.npz` can contain:

| Key | Shape | Meaning |
|---|---:|---|
| `local_points` | `[M+N, H, W, 3]` | Per-frame local 3D point map. Depth is `[..., 2]`. |
| `points` | `[M+N, H, W, 3]` | Points transformed by predicted camera poses. |
| `conf` | `[M+N, H, W, 1]` | Confidence logits. |
| `camera_poses` | `[M+N, 4, 4]` | Predicted camera poses. |
| `segmentation` | `[M+N, H, W, C]` | Segmentation logits, if enabled. |
| `motion` | `[M+N, H, W, 1]` | Motion logits, if enabled. |
| `flow` | `[M+N, H, W, 2]` | Optical-flow logits, if enabled. |

For long videos or image sequences, inference streams sampled frames through sliding windows instead of decoding the full input into memory first. The first `M` predictions correspond to the input/history frames for that window; the next `N` predictions are autoregressive future predictions. The JSON metadata records the source frame indices and which slots are padded for short tail windows.

## Acknowledgments

LFG builds on the excellent [Pi3](https://github.com/yyfz/Pi3) project, whose model code is bundled under `Pi3/`. We thank the Pi3 authors for releasing their work. Pi3 in turn builds on [DINOv2](https://github.com/facebookresearch/dinov2) (Meta Platforms), bundled under `Pi3/pi3/models/dinov2/`.
