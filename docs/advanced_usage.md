# Advanced Usage

This guide covers advanced features for experienced users who need custom data filtering, specialized training configurations, or want to use the PyTorch Lightning DataModule directly.

## Custom Data Filtering with Lightning DataModule

The `NeonCrownDataModule` provides flexible filtering and splitting options for advanced use cases.

### Basic Configuration

```python
from neon_tree_classification.core.datamodule import NeonCrownDataModule

# Basic configuration with species/site filtering
datamodule = NeonCrownDataModule(
    csv_path="_neon_tree_classification_dataset_files/metadata/combined_dataset.csv",
    hdf5_path="_neon_tree_classification_dataset_files/neon_dataset.h5",
    modalities=["rgb"],  # Single modality training
    batch_size=32,
    # Filtering options
    species_filter=["PSMEM", "TSHE"],  # Train on specific species
    site_filter=["HARV", "OSBS"],      # Train on specific sites
    year_filter=[2018, 2019],          # Train on specific years
    # Split method options
    split_method="random",  # Options: "random", "site", "year"
    val_ratio=0.15,
    test_ratio=0.15
)

datamodule.setup("fit")
```

### Split Methods

The DataModule supports three splitting strategies:

**1. Random Split** (default)
```python
datamodule = NeonCrownDataModule(
    csv_path="path/to/dataset.csv",
    hdf5_path="path/to/dataset.h5",
    split_method="random",
    val_ratio=0.15,
    test_ratio=0.15
)
```

**2. Site-Based Split**

Useful for testing generalization across geographic locations:
```python
datamodule = NeonCrownDataModule(
    csv_path="path/to/dataset.csv",
    hdf5_path="path/to/dataset.h5",
    split_method="site",
    val_ratio=0.15,
    test_ratio=0.15
)
```

**3. Year-Based Split**

Useful for testing temporal generalization:
```python
datamodule = NeonCrownDataModule(
    csv_path="path/to/dataset.csv",
    hdf5_path="path/to/dataset.h5",
    split_method="year",
    val_ratio=0.15,
    test_ratio=0.15
)
```

### External Test Sets

For domain adaptation or cross-site validation:

```python
datamodule = NeonCrownDataModule(
    csv_path="_neon_tree_classification_dataset_files/metadata/combined_dataset.csv",
    hdf5_path="_neon_tree_classification_dataset_files/neon_dataset.h5",
    external_test_csv_path="path/to/external_test.csv",
    external_test_hdf5_path="path/to/external_test.h5",  # Optional, uses main HDF5 if not provided
    modalities=["rgb"]
)

datamodule.setup("fit")  # Auto-filters species for compatibility
```

## Advanced DataLoader Configuration

### Custom Normalization

Each modality supports different normalization methods:

**RGB Normalization:**
- `"0_1"`: Scale to [0, 1] range (default)
- `"imagenet"`: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- `"per_sample"`: Normalize each sample independently

**HSI Normalization:**
- `"per_sample"`: Normalize each sample independently (default)
- `"global"`: Use global dataset statistics
- `"none"`: No normalization

**LiDAR Normalization:**
- `"height"`: Normalize by maximum canopy height (default)
- `"per_sample"`: Normalize each sample independently
- `"none"`: No normalization

Example:
```python
train_loader, test_loader = get_dataloaders(
    config='large',
    modalities=['rgb', 'hsi', 'lidar'],
    batch_size=32,
    rgb_norm_method='imagenet',
    hsi_norm_method='global',
    lidar_norm_method='height'
)
```

### Custom Image Sizes

Adjust the spatial resolution for each modality:

```python
train_loader, test_loader = get_dataloaders(
    config='large',
    modalities=['rgb', 'hsi', 'lidar'],
    batch_size=32,
    rgb_size=(224, 224),    # Larger RGB for fine-grained features
    hsi_size=(16, 16),      # Higher HSI resolution
    lidar_size=(16, 16)     # Higher LiDAR resolution
)
```

## Direct Dataset Usage

For maximum control, use the `NeonCrownDataset` class directly:

```python
from neon_tree_classification.core.dataset import NeonCrownDataset
from torch.utils.data import DataLoader

# Create dataset with custom parameters
dataset = NeonCrownDataset(
    csv_path="_neon_tree_classification_dataset_files/metadata/large_dataset.csv",
    hdf5_path="_neon_tree_classification_dataset_files/neon_dataset.h5",
    modalities=['rgb', 'hsi'],
    species_filter=['ACRU', 'TSCA'],  # Limit to specific species
    site_filter=['HARV', 'MLBS'],     # Limit to specific sites
    year_filter=[2018, 2019, 2020],   # Limit to specific years
    include_metadata=True,             # Include crown_id, species names, etc.
    rgb_size=(128, 128),
    hsi_size=(12, 12),
    rgb_norm_method='imagenet',
    hsi_norm_method='per_sample'
)

# Create custom DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
```

## Accessing Metadata

Enable metadata in batches to access crown IDs, species names, and site information:

```python
from scripts.get_dataloaders import get_dataloaders

# Note: get_dataloaders doesn't support include_metadata yet
# Use NeonCrownDataset directly:
from neon_tree_classification.core.dataset import NeonCrownDataset

dataset = NeonCrownDataset(
    csv_path="path/to/dataset.csv",
    hdf5_path="path/to/dataset.h5",
    modalities=['rgb'],
    include_metadata=True
)

# Access metadata in batches
for batch in DataLoader(dataset, batch_size=32):
    rgb = batch['rgb']
    labels = batch['species_idx']
    crown_ids = batch['crown_id']
    species_names = batch['species']
    sites = batch['site']
```

## Multi-GPU Training

For distributed training with PyTorch Lightning:

```python
import pytorch_lightning as pl
from neon_tree_classification.core.datamodule import NeonCrownDataModule

# Configure DataModule
datamodule = NeonCrownDataModule(
    csv_path="path/to/dataset.csv",
    hdf5_path="path/to/dataset.h5",
    modalities=["rgb"],
    batch_size=32  # Per-GPU batch size
)

# Create trainer with multi-GPU support
trainer = pl.Trainer(
    devices=4,           # Number of GPUs
    strategy='ddp',      # Distributed Data Parallel
    precision=16,        # Mixed precision training
    max_epochs=100
)

# Your Lightning module
trainer.fit(model, datamodule=datamodule)
```

## Custom Training Loop

Example of a custom training loop without PyTorch Lightning:

```python
import torch
from scripts.get_dataloaders import get_dataloaders

# Get dataloaders
train_loader, test_loader = get_dataloaders(
    config='large',
    modalities=['rgb'],
    batch_size=64
)

# Your model
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    model.train()
    for batch in train_loader:
        rgb = batch['rgb'].cuda()
        labels = batch['species_idx'].cuda()
        
        optimizer.zero_grad()
        outputs = model(rgb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            rgb = batch['rgb'].cuda()
            labels = batch['species_idx'].cuda()
            outputs = model(rgb)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Epoch {epoch}: Accuracy = {accuracy:.2f}%')
```

## Performance Tips

1. **Use larger batch sizes**: The dataset fits in memory efficiently due to HDF5 compression
2. **Increase num_workers**: More workers can significantly speed up data loading
3. **Enable pin_memory**: Speeds up CPU-to-GPU transfer
4. **Use persistent_workers**: Reduces worker initialization overhead

```python
train_loader, test_loader = get_dataloaders(
    config='large',
    modalities=['rgb'],
    batch_size=256,      # Larger batch size
    num_workers=16,      # More workers (adjust based on CPU cores)
)
```
