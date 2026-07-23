# TISR Challenge 2026 Track 1

This repository contains the experimental code used for **Track 1 of the PBVS @ CVPR 2026 Thermal Image Super-Resolution Challenge**.

The task is to reconstruct an 8× super-resolved thermal image from a single low-resolution input. The codebase is built on top of DRCT and BasicSR, with custom modifications to the loss function and the thermal dataset pipeline.

This README follows the provided project-summary format, but is simplified around the representative training script:

```text
train_depth_anything.sh
```

## Result

- Team/User: `junjun`
- Submission ID: `582110`
- Submission time: `2026-03-01 21:07`
- Score: `28.49`
- SSIM: `0.8445`
- Rank: **9th**

## Overview

This project focuses on a DRCT-based thermal image super-resolution model, with two main custom changes:

- An MSE + SSIM-based training objective for better structural preservation
- A paired dataset loader for thermal GT/LQ image training

The representative training entry in the current repository is:

```text
train_depth_anything.sh
```

In the current checked-in version, the configuration name still refers to a depth experiment, but depth conditioning is disabled. Therefore, this script should be regarded as the main remaining training configuration in the repository, not as an actively depth-enabled final model.

## Main Changes

### SSIM-based MultiLoss

Relevant files:

```text
drct/losses/multi_loss.py
drct/losses/__init__.py
```

A Gaussian-window-based SSIM loss was implemented and combined with MSE loss to better preserve structure in thermal image reconstruction.

The training objective is:

```text
MultiLoss = MSE + 0.02 * SSIMLoss
```

`SSIMLoss` is computed as `1 - SSIM`.

The implementation also includes:

- safe handling of list / tuple / dict model outputs
- target resizing when prediction and target shapes do not match

### Thermal Paired Dataset

Relevant file:

```text
drct/data/thermal_paired_dataset.py
```

A paired dataset loader was implemented to support the thermal image training pipeline.

Main features include:

- GT/LQ pairing based on filename stem
- paired random cropping
- optional LQ-side masking support
- validation/test padding and alignment

These changes make the training setup more robust for thermal SR experiments than the default generic image loader.

## Representative Training Configuration

The current main training script is:

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29522 \
  drct/train.py \
  -opt options/train/finetune_use_depth_anything.yml \
  --launcher pytorch
```

The corresponding config file is:

```text
options/train/finetune_use_depth_anything.yml
```

Important note:

```text
use_depth: false
use_depth_cond: false
```

So, despite the script name, the current checked-in version does **not** actively use depth maps during training.

## Dataset Structure

The basic dataset layout is:

```text
datasets/
└── track1/
    ├── thermal/
    │   ├── train/
    │   │   ├── GT/
    │   │   └── LR_x8/
    │   ├── val/
    │   │   ├── GT/
    │   │   └── LR_x8/
    │   └── test/
    │       └── sisr_x8/
    └── depth/
        ├── train/
        └── val/
```

For the representative training script, the active dataset paths are:

```text
datasets/track1/thermal/train/GT
datasets/track1/thermal/train/LR_x8
datasets/track1/thermal/val/GT
datasets/track1/thermal/val/LR_x8
```

The depth directories remain in the config, but they are not enabled in the current checked-in setup.

## Training

Run training with:

```bash
bash train_depth_anything.sh
```

or directly:

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29522 \
  drct/train.py \
  -opt options/train/finetune_use_depth_anything.yml \
  --launcher pytorch
```

Key settings in the current config:

- scale: `8`
- model: `DRCT`
- input channels: `3`
- training patch size: `384`
- optimizer: `Adam`
- learning rate: `1e-5`
- scheduler: `CosineAnnealingRestartLR`
- total iterations: `50000`
- initialization checkpoint: `experiments/pretrained_models/DRCT-L_X4.pth`

## Inference

Run inference with:

```bash
python drct/test.py -opt options/test/test_DRCT-L_SRx8_finetune_from_merged.yml
```

Some test paths in the repository may be environment-specific, so they should be checked before running inference on a different machine.

## Repository Layout

```text
PBVS_TSR/
├── drct/
│   ├── archs/
│   ├── data/
│   ├── losses/
│   ├── train.py
│   └── test.py
├── options/
│   ├── train/
│   └── test/
├── datasets/
├── results/
├── tb_logger/
├── utils/
└── train_depth_anything.sh
```

## Acknowledgements

- [DRCT](https://github.com/ming053l/DRCT)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [TISR](https://github.com/upczww/TISR/tree/master)
