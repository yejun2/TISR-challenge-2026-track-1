# TISR Challenge 2026 Track 1

This repository contains the experimental code used for **Track 1 of the PBVS @ CVPR 2026 Thermal Image Super-Resolution Challenge**.

The task is to reconstruct an 8× super-resolved thermal image from a single low-resolution input. The codebase is built on top of DRCT and includes modifications to the loss functions, dataset pipeline, input representation, and training strategy.

## Result

- Team/User: `junjun`
- Submission ID: `582110`
- Submission time: `2026-03-01 21:07`
- Score: `28.49`
- SSIM: `0.8445`
- Rank: **9th**

## Overview

This project investigates several modifications to a DRCT-based thermal image super-resolution model, with a particular focus on preserving structural information in thermal imagery.

The main experiments include:

- An MSE and SSIM-based training objective
- A paired dataset loader for thermal GT/LQ images
- Patch masking augmentation applied only to low-resolution inputs
- Mixed pretraining data generation
- Intermediate supervision using an `LR_x8 → LR_x2 → GT` pipeline
- A 5-channel input using additional high-pass features
- An experimental depth-conditioning branch based on Depth Anything

In the current checked-in version, depth conditioning is disabled. Therefore, the depth-related code should be regarded as an experimental branch rather than the final submission configuration.

## Main Changes

### SSIM-based MultiLoss

Relevant files:

```text
drct/losses/multi_loss.py
drct/losses/__init__.py
```

A Gaussian-window-based SSIM loss was implemented to complement pixel-wise reconstruction loss with structural similarity.

The final objective is defined as:

```text
MultiLoss = MSE + 0.02 × SSIMLoss
```

`SSIMLoss` is computed as `1 - SSIM`.

The implementation also handles model outputs in list, tuple, or dictionary form by selecting the final output or the `x8` output when available.

If the spatial size of the prediction differs from that of the target, the target is resized before computing the loss.

### Thermal Paired Dataset

Relevant file:

```text
drct/data/thermal_paired_dataset.py
```

A paired dataset loader was implemented to reliably match thermal GT and LQ images and support the experimental training pipeline.

Main features include:

- GT/LQ pairing based on filename stem
- Paired random cropping
- LQ-only patch masking
- Loading HP features and concatenating them with the thermal input
- Padding and spatial alignment during validation and testing

The current implementation includes support for HP feature inputs, while direct depth-map concatenation is not enabled in the checked-in version.

### Mixed Pretraining Data

Relevant file:

```text
utils/merge_image.py
```

The script creates mixed pretraining samples by cropping and combining regions from multiple thermal images.

```bash
python utils/merge_image.py
```

Generated samples are stored in:

```text
datasets/track1/thermal/train/LR_x8_mix_pretrain
datasets/track1/thermal/train/GT_mix_pretrain
```

### Input Masking

Patch masking is applied to the low-resolution input to evaluate whether the model can recover missing or degraded local regions.

Relevant configurations:

```text
options/train/train_finetune_use_masking.yml
options/train/train_finetune_use_masking_paired.yml
```

### HP Feature Input

A 5-channel input setting was tested by concatenating thermal images with separately extracted high-pass features.

Relevant configuration:

```text
options/train/finetune_use_hp_feat.yml
```

Training script:

```text
train_hp.sh
```

### Intermediate Supervision

In addition to directly reconstructing GT from `LR_x8`, a two-stage training pipeline was explored using an intermediate `LR_x2` target.

```text
LR_x8 → LR_x2 → GT
```

The `LR_x2` data can be generated using:

```bash
python utils/downsample2x.py <input_dir> <output_dir>
```

Relevant configuration:

```text
options/train/finetune_x2_from_opencv_pretrained.yml
```

Training script:

```text
train_x2_opencv.sh
```

## Experiment Configurations

| Experiment | Configuration | Description |
|---|---|---|
| Mixed pretraining | `utils/merge_image.py` | Generates mixed thermal samples for pretraining |
| Base x8 finetuning | `train_coslr_multiloss_multiscale_DRCT-L_SRx8_finetune_from_merged.yml` | Finetunes on the official training split after mixed pretraining |
| Input masking | `train_finetune_use_masking.yml` | Applies patch masking to LQ inputs |
| Masking with paired loader | `train_finetune_use_masking_paired.yml` | Combines masking with the paired thermal loader |
| Paired thermal loader | `train_finetune_from_paired_dataset.yml` | Uses the custom GT/LQ pairing pipeline |
| HP 5-channel input | `finetune_use_hp_feat.yml` | Concatenates thermal input with HP features |
| Intermediate x2 supervision | `finetune_x2_from_opencv_pretrained.yml` | Uses an `LR_x8 → LR_x2 → GT` training pipeline |
| Depth experiment | `finetune_use_depth_anything.yml` | Experimental depth-conditioning configuration |

## Depth Experiment

`train_depth_anything.sh` runs the following configuration:

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29522 \
  drct/train.py \
  -opt options/train/finetune_use_depth_anything.yml \
  --launcher pytorch
```

However, the current `finetune_use_depth_anything.yml` disables depth conditioning:

```text
use_depth: false
use_depth_cond: false
```

Therefore, running the current configuration does not use depth maps as model inputs.

Depth Anything-based conditioning was explored during development, but the active depth-enabled configuration is not included in the current repository state.

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

Additional derived data used in some experiments:

```text
datasets/track1/thermal/train/LR_x8_mix_pretrain
datasets/track1/thermal/train/GT_mix_pretrain
datasets/track1/thermal/train/LR_x2
datasets/track1/thermal/val/LR_x2
datasets/track1/thermal/train/HP_feats
datasets/track1/thermal/val/HP_feats
```

## Training

A representative training configuration is:

```text
options/train/train_coslr_multiloss_multiscale_DRCT-L_SRx8_finetune_from_merged.yml
```

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29521 \
  drct/train.py \
  -opt options/train/train_coslr_multiloss_multiscale_DRCT-L_SRx8_finetune_from_merged.yml \
  --launcher pytorch
```

Main training scripts:

```text
train_depth_anything.sh
train_hp.sh
train_masking.sh
train_paired.sh
train_x2_opencv.sh
```

## Inference

Run inference with:

```bash
python drct/test.py \
  -opt options/test/test_DRCT-L_SRx8_finetune_from_merged.yml
```

or:

```bash
bash test.sh
```

Some test configurations still contain absolute paths from the original server environment. Dataset, checkpoint, and output paths may need to be updated before running the code elsewhere.

## Project Structure

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
├── utils/
│   ├── merge_image.py
│   ├── downsample2x.py
│   ├── post_processing.py
│   └── fuzzy_images.py
├── datasets/
│   └── track1/
├── results/
├── tb_logger/
├── Visualization/
└── BasicSR/
```

## Personal Contributions

The main contributions in this repository include:

- Implementing a Gaussian-window-based SSIM loss
- Combining MSE and SSIM into a custom MultiLoss
- Implementing a paired thermal GT/LQ dataset loader
- Applying LQ-only patch masking augmentation
- Generating mixed pretraining samples
- Testing 5-channel inputs with HP features
- Exploring `LR_x8 → LR_x2 → GT` intermediate supervision
- Investigating a Depth Anything-based conditioning branch
- Organizing training and evaluation configurations

## Acknowledgements

This project builds on the following open-source repositories:

- [DRCT](https://github.com/ming053l/DRCT)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [TISR](https://github.com/upczww/TISR)
