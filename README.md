# 3D UNet for TOF-MRA Vessel Segmentation

A PyTorch implementation of 3D UNet for automatic vessel segmentation in Time-of-Flight Magnetic Resonance Angiography (TOF-MRA) images.

## Overview

This project implements a residual 3D UNet architecture for segmenting blood vessels from TOF-MRA scans. The model uses:

- **Residual Double Convolution blocks** with BatchNorm and ReLU activations
- **Skip connections** between encoder and decoder paths
- **Dice + BCE Combo Loss** for handling class imbalance
- **Sliding window inference** for processing full-volume scans
- **Foreground oversampling** during training to ensure vessel-rich patches

## Architecture

```
Input: (1, D, H, W) → [64→128→256→512] → Bottleneck → [512→256→128→64] → Output: (1, D, H, W)
```

The network follows the standard UNet encoder-decoder structure with:
- **Encoder**: 4 levels of DoubleConv + MaxPool3d downsampling
- **Decoder**: Transposed convolutions for upsampling with skip connections
- **Residual connections** in each DoubleConv block

## Files

| File | Description |
|------|-------------|
| `model.py` | 3D UNet architecture with residual blocks |
| `dataset.py` | PyTorch Dataset for NIfTI MRI data with augmentation |
| `train.py` | Training loop with mixed precision, Dice/BCE loss |
| `predict.py` | Sliding window inference for full-volume prediction |

## Requirements

```
torch>=1.9.0
numpy
nibabel
tqdm
```

## Usage

### Training

```python
python train.py
```

**Configuration** (edit in `train.py`):
- `TRAIN_IMG_DIR`: Path to training data (`raw/` and `gt/` subfolders)
- `VAL_IMG_DIR`: Path to validation data
- `BATCH_SIZE`: Default 2 (3D data is memory-intensive)
- `NUM_EPOCHS`: Default 50
- `LEARNING_RATE`: Default 1e-4

### Inference

```python
python predict.py
```

**Configuration** (edit in `predict.py`):
- `TEST_IMG_DIR`: Path to test images
- `OUTPUT_DIR`: Directory for prediction outputs
- `MODEL_PATH`: Path to trained checkpoint

## Data Format

Expected directory structure:
```
dataset/
├── raw/          # Input TOF-MRA NIfTI files (*.nii.gz)
└── gt/           # Ground truth segmentation masks (*.nii.gz)
```

- **Input**: 3D NIfTI files with vessel intensity values
- **Mask**: Binary segmentation (0=background, 1=vessel)

## Features

- **Mixed Precision Training**: AMP with GradScaler for faster training
- **Robust Normalization**: Percentile-based (99.5th) intensity normalization
- **Foreground Oversampling**: Rejection sampling for vessel-rich patches
- **Sliding Window Inference**: Memory-efficient full-volume prediction
- **Learning Rate Scheduling**: ReduceLROnPlateau based on validation Dice

## License

MIT
