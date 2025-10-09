# NEON Multi-Modal Tree Species Classification Dataset

A comprehensive dataset of 167 tree species with 47,971 individual tree crowns from 30 NEON sites across North America. Each sample includes RGB imagery, 369-band hyperspectral data, and LiDAR canopy height models.

## Dataset Overview

**Dataset Statistics:**
- 47,971 individual tree crowns
- 167 unique species
- 30 NEON sites across North America
- 10 years of data (2014-2023)
- 3 modalities: RGB (3 bands), Hyperspectral (369 bands), LiDAR CHM (1 band)
- Ecological metadata: Height (95.4% available), stem diameter (99.4% available), canopy position (81.4% available)

**Dataset Configurations:**

The dataset comes with 3 pre-configured subsets for different use cases:

| Configuration | Samples | Species | Description |
|---------------|---------|---------|-------------|
| `combined` | 47,971 | 167 | Complete dataset with all available samples |
| `large` | ~42,000 | ~162 | Main training set |
| `high_quality` | ~5,500 | ~96 | Curated subset with highest data quality |

**Data Format:**
- HDF5 storage for efficient compressed format and fast loading
- CSV metadata files with crown IDs, species labels, site information, and ecological measurements

## Installation

**Requirements:**
- Python 3.9+ (recommended: Python 3.11)
- CUDA-capable GPU (optional, recommended for training)

**Installation Steps:**

```bash
# Clone the repository
git clone https://github.com/Ritesh313/NeonTreeClassification.git
cd NeonTreeClassification

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Getting Started

### Quick Start Example

Run the quickstart script to verify installation and download the dataset:

```bash
# Using uv
uv run python quickstart.py

# Or after activating environment
source .venv/bin/activate
python quickstart.py
```

The dataset (590 MB) will automatically download on first use.

### Using Dataloaders

**Basic Usage:**

```python
from scripts.get_dataloaders import get_dataloaders

# Get dataloaders (dataset downloads automatically on first use)
train_loader, test_loader = get_dataloaders(
    config='large',
    modalities=['rgb', 'hsi', 'lidar'],
    batch_size=32
)

# Each batch contains:
for batch in train_loader:
    rgb_data = batch['rgb']        # torch.Tensor [batch_size, 3, 128, 128]
    hsi_data = batch['hsi']        # torch.Tensor [batch_size, 369, 12, 12]
    lidar_data = batch['lidar']    # torch.Tensor [batch_size, 1, 12, 12]
    labels = batch['species_idx']  # torch.Tensor [batch_size]
```

**Training Scenarios:**

```python
# Standard training on large dataset
train_loader, test_loader = get_dataloaders(config='large', test_ratio=0.2)

# Maximum data training
train_loader, test_loader = get_dataloaders(config='combined', test_ratio=0.15)

# High-quality subset only
train_loader, test_loader = get_dataloaders(config='high_quality', test_ratio=0.2)

# Domain transfer (train on large, test on high_quality)
train_loader, test_loader = get_dataloaders(
    train_config='large',
    test_config='high_quality'
)
```

## Dataset Details

**Top Species:**

| Rank | Species | Count | Percentage |
|------|---------|-------|------------|
| 1 | Acer rubrum L. | 5,684 | 11.8% |
| 2 | Tsuga canadensis (L.) Carrière | 3,303 | 6.9% |
| 3 | Pseudotsuga menziesii (Mirb.) Franco var. menziesii | 2,978 | 6.2% |
| 4 | Pinus palustris Mill. | 2,207 | 4.6% |
| 5 | Quercus rubra L. | 2,086 | 4.3% |

**Geographic Distribution:**

Data collected from 30 NEON sites across North America. Top 5 sites:
- HARV: 7,162 samples (14.9%)
- MLBS: 5,424 samples (11.3%)
- GRSM: 4,822 samples (10.1%)
- DELA: 4,539 samples (9.5%)
- RMNP: 3,931 samples (8.2%)

**NEON Data Products:**
- RGB: DP3.30010.001 - High-resolution orthorectified imagery
- Hyperspectral: DP3.30006.002 - 426-band spectrometer reflectance
- LiDAR: DP3.30015.001 - Canopy Height Model

**Data Structure:**

The metadata CSV contains:
- `crown_id`: Unique identifier for each tree crown
- `species`: Species code
- `species_name`: Full species name
- `site`: NEON site code
- `year`: Data collection year
- `height`: Tree height in meters
- `stemDiameter`: Stem diameter in cm
- `canopyPosition`: Light exposure level
- `rgb_path`, `hsi_path`, `lidar_path`: Paths to data in HDF5 file

## Documentation

For detailed documentation, see the `docs/` directory:
- [Advanced Usage](docs/advanced_usage.md) - Custom filtering, Lightning DataModule, and advanced features
- [Training Guide](docs/training.md) - Model training examples and baseline results
- [Visualization Guide](docs/visualization.md) - Data visualization tools and examples
- [Processing Pipeline](docs/processing.md) - NEON data processing workflow

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## Acknowledgments

- National Ecological Observatory Network (NEON)
- Dataset statistics generated on 2025-08-28
