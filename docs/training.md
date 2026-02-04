# Training Guide

This guide covers model training, baseline results, and tips for achieving good performance on the NEON tree classification dataset.

## Quick Training with Examples Script

The repository includes a flexible training script that supports all modalities:

```bash
# Train RGB classifier
uv run python examples/train.py \
    --modality rgb \
    --csv_path _neon_tree_classification_dataset_files/metadata/large_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5

# Train hyperspectral classifier
uv run python examples/train.py \
    --modality hsi \
    --csv_path _neon_tree_classification_dataset_files/metadata/combined_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5 \
    --batch_size 16

# Train LiDAR classifier
uv run python examples/train.py \
    --modality lidar \
    --csv_path _neon_tree_classification_dataset_files/metadata/high_quality_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5

# External test set (train on large, test on high_quality)
uv run python examples/train.py \
    --modality rgb \
    --csv_path _neon_tree_classification_dataset_files/metadata/large_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5 \
    --external_test_csv _neon_tree_classification_dataset_files/metadata/high_quality_dataset.csv
```

## Baseline Results

Single-modality baseline results using the `combined` dataset configuration (47,971 samples, seed=42):

| Modality | Test Accuracy | Model | Hyperparameters | Notes |
|----------|---------------|-------|-----------------|-------|
| **RGB (Species)** | **75.9%** | ResNetRGB | lr=5e-5, wd=5e-4, bs=256 | 167 species classes, optimized |
| **RGB (Genus)** | **72.2%** | ResNetRGB | lr=5e-5, wd=5e-4, bs=256 | 60 genus classes, coarser taxonomy |
| HSI | 27.3% | Spectral CNN | Default params | 369-band hyperspectral data |
| LiDAR | 11.5% | Structural CNN | Default params | Canopy height model |

**Important Notes:**
- RGB performance achieved through config: lr=5e-5, weight_decay=5e-4, batch_size=256, AdamW optimizer
- HSI and LiDAR results are preliminary with default parameters - significant improvement expected with optimization
- Multi-modal fusion is expected to significantly improve performance

## Reproducing Baseline Results

### Prerequisites

First, download the dataset:
```python
from scripts.get_dataloaders import get_dataloaders
# This downloads the dataset to _neon_tree_classification_dataset_files/
train_loader, test_loader = get_dataloaders(config='combined')
```

### With Comet ML (Exact Reproduction)

The original experiments used Comet ML for logging and early stopping:

```bash
# RGB baseline
uv run python examples/train.py \
    --csv_path _neon_tree_classification_dataset_files/metadata/combined_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5 \
    --modality rgb --model_type resnet --batch_size 1024 --seed 42 \
    --logger comet --early_stop_patience 15

# HSI baseline
uv run python examples/train.py \
    --csv_path _neon_tree_classification_dataset_files/metadata/combined_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5 \
    --modality hsi --model_type spectral_cnn --batch_size 128 --seed 42 \
    --logger comet --early_stop_patience 15

# LiDAR baseline
uv run python examples/train.py \
    --csv_path _neon_tree_classification_dataset_files/metadata/combined_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5 \
    --modality lidar --model_type structural --batch_size 1024 --seed 42 \
    --logger comet --early_stop_patience 15
```

### Without Comet ML (Approximate Reproduction)

Without early stopping, results may vary:

```bash
# RGB baseline (fixed epochs)
uv run python examples/train.py \
    --csv_path _neon_tree_classification_dataset_files/metadata/combined_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5 \
    --modality rgb --model_type resnet --batch_size 1024 --seed 42 --epochs 100

# HSI baseline (fixed epochs)
uv run python examples/train.py \
    --csv_path _neon_tree_classification_dataset_files/metadata/combined_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5 \
    --modality hsi --model_type spectral_cnn --batch_size 128 --seed 42 --epochs 100

# LiDAR baseline (fixed epochs)
uv run python examples/train.py \
    --csv_path _neon_tree_classification_dataset_files/metadata/combined_dataset.csv \
    --hdf5_path _neon_tree_classification_dataset_files/neon_dataset.h5 \
    --modality lidar --model_type structural --batch_size 1024 --seed 42 --epochs 100
```

## Custom Model Architectures

### Creating Custom Models

Add new model architectures in `neon_tree_classification/models/` and reference them with the `--model_type` flag.

Example custom model:

```python
# neon_tree_classification/models/my_custom_model.py
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Add more layers...
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x
```

## Training Best Practices

### 1. Start with RGB

RGB data is easiest to work with and provides good baseline performance:
- Standard computer vision techniques apply
- Pre-trained ImageNet models can be fine-tuned
- Faster training times

### 2. Dataset Configuration Selection

Choose based on your goals:
- `combined`: Maximum data, all species
- `large`: Good balance of data quantity and quality
- `high_quality`: Cleanest data, fewer species

### 3. Hyperparameter Tuning

Key hyperparameters to tune:
- Learning rate (start with 1e-3 to 1e-4)
- Batch size (larger is usually better, up to memory limits)
- Weight decay (0 to 1e-4)
- Augmentation parameters

### 4. Data Augmentation

For RGB:
```python
import torchvision.transforms as transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```

### 5. Learning Rate Scheduling

Use learning rate scheduling for better convergence:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=100
)
```

## Multi-Modal Training

Combining multiple modalities typically improves performance:

### Early Fusion
```python
# Concatenate features from different modalities
rgb_features = rgb_encoder(rgb_data)
hsi_features = hsi_encoder(hsi_data)
lidar_features = lidar_encoder(lidar_data)

combined = torch.cat([rgb_features, hsi_features, lidar_features], dim=1)
output = classifier(combined)
```

### Late Fusion
```python
# Average predictions from different modalities
rgb_pred = rgb_model(rgb_data)
hsi_pred = hsi_model(hsi_data)
lidar_pred = lidar_model(lidar_data)

final_pred = (rgb_pred + hsi_pred + lidar_pred) / 3
```

## Experiment Tracking

### Using Comet ML

```bash
# Set up Comet ML
export COMET_API_KEY="your_api_key"

# Train with Comet logging
uv run python examples/train.py \
    --modality rgb \
    --logger comet \
    --csv_path path/to/dataset.csv \
    --hdf5_path path/to/dataset.h5
```

### Using Weights & Biases

```bash
# Set up W&B
wandb login

# Train with W&B logging
uv run python examples/train.py \
    --modality rgb \
    --logger wandb \
    --csv_path path/to/dataset.csv \
    --hdf5_path path/to/dataset.h5
```

## Common Issues and Solutions

### Issue: Out of Memory
**Solution:** Reduce batch size or image resolution
```bash
python examples/train.py --batch_size 16 --modality rgb
```

### Issue: Slow Training
**Solution:** Increase num_workers and use larger batches
```bash
python examples/train.py --batch_size 256 --num_workers 16
```

### Issue: Poor Convergence
**Solution:** 
1. Check learning rate (try 1e-4 or 1e-5)
2. Use learning rate warmup
3. Add data augmentation
4. Try different normalization methods

### Issue: Overfitting
**Solution:**
1. Add dropout
2. Use weight decay
3. Add more data augmentation
4. Use early stopping

## Performance Benchmarks

Training times on NVIDIA A100 (40GB):

| Modality | Batch Size | Epochs | Time per Epoch | Total Time |
|----------|------------|--------|----------------|------------|
| RGB | 1024 | 100 | ~2 min | ~3.5 hours |
| HSI | 128 | 100 | ~5 min | ~8 hours |
| LiDAR | 1024 | 100 | ~1 min | ~2 hours |

Memory requirements:
- RGB: ~8GB GPU memory (batch_size=1024)
- HSI: ~12GB GPU memory (batch_size=128)
- LiDAR: ~4GB GPU memory (batch_size=1024)
