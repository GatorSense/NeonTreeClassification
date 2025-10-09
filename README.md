# NEON Multi-Modal Tree Species Classification

A comprehensive toolkit for multi-modal tree species classification using NEON ecological data. This project combines RGB imagery, hyperspectral data, and LiDAR to enable accurate tree species identification across diverse North American ecosystems.

## Project Vision

This repository aims to provide an end-to-end solution for tree species classification:

- [x] **Dataset**: Ready-to-use multi-modal tree crown dataset with 167 species
- [ ] **Data Processing**: Tools for downloading and processing raw NEON data products
- [ ] **Classification Models**: Pre-trained models and training pipelines
- [ ] **DeepForest Integration**: Automated crown detection and classification workflow

## What's Available Now

### Multi-Modal Dataset

A curated dataset of 47,971 individual tree crowns from 30 NEON sites, ready for immediate use:

- **167 tree species** from diverse North American ecosystems
- **3 modalities**: RGB (3 bands), Hyperspectral (369 bands), LiDAR CHM (1 band)
- **10 years of data** (2014-2023) with ecological metadata
- **3 configurations**: `combined` (47,971 samples), `large` (~42,000 samples), `high_quality` (~5,500 samples)
- **HDF5 format**: Efficient storage with automatic download (590 MB)

## Quick Start

### Installation

```bash
git clone https://github.com/Ritesh313/NeonTreeClassification.git
cd NeonTreeClassification
uv sync  # or: pip install -e .
```

### Get the Dataset

```python
from scripts.get_dataloaders import get_dataloaders

# Dataset downloads automatically (590 MB)
train_loader, test_loader = get_dataloaders(
    config='large',
    modalities=['rgb', 'hsi', 'lidar'],
    batch_size=32
)

# Use in your training loop
for batch in train_loader:
    rgb = batch['rgb']          # [batch_size, 3, 128, 128]
    hsi = batch['hsi']          # [batch_size, 369, 12, 12]
    lidar = batch['lidar']      # [batch_size, 1, 12, 12]
    labels = batch['species_idx']  # [batch_size]
```

Or run the quickstart example:
```bash
uv run python quickstart.py
```

## Coming Soon

**Data Processing Pipeline**: Tools for processing raw NEON data products are being finalized and will be released for public use. This will enable users to:
- Download NEON tiles for all three modalities
- Crop individual tree crowns from shapefiles
- Create custom datasets with their own crown annotations

**Classification Models**: Pre-trained models and training scripts for tree species classification will be added to the repository.

**DeepForest Integration**: Planned integration with [DeepForest](https://github.com/weecology/DeepForest) to enable:
- Automatic crown detection from aerial imagery
- Seamless multi-modal data extraction for detected crowns
- Direct classification using pre-trained models from this repository

## Dataset Details

**Top 5 Species:**
1. Acer rubrum L. (5,684 samples, 11.8%)
2. Tsuga canadensis (L.) Carrière (3,303 samples, 6.9%)
3. Pseudotsuga menziesii (Mirb.) Franco var. menziesii (2,978 samples, 6.2%)
4. Pinus palustris Mill. (2,207 samples, 4.6%)
5. Quercus rubra L. (2,086 samples, 4.3%)

**Top 5 Sites:**
- HARV: 7,162 samples (14.9%)
- MLBS: 5,424 samples (11.3%)
- GRSM: 4,822 samples (10.1%)
- DELA: 4,539 samples (9.5%)
- RMNP: 3,931 samples (8.2%)

**NEON Data Products:**
- RGB: DP3.30010.001 (High-resolution orthorectified imagery)
- Hyperspectral: DP3.30006.002 (426-band spectrometer reflectance)
- LiDAR: DP3.30015.001 (Canopy Height Model)

For complete dataset documentation, training guides, and advanced usage, see the [docs/](docs/) directory.

## Acknowledgments

National Ecological Observatory Network (NEON)

