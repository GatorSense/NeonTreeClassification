# Processing Pipeline

This guide documents the NEON data processing pipeline for advanced users who want to process raw NEON data products or understand how the dataset was created.

## Overview

The `processing/` folder contains a comprehensive pipeline for converting raw NEON data products into the final training-ready dataset. This is useful for:
- Processing new NEON data
- Customizing the dataset creation process
- Understanding data quality control steps
- Creating similar datasets from other sources

## Pipeline Steps

### 1. Download NEON Tiles

Download RGB, HSI, and LiDAR data from the NEON API:

```bash
python processing/neon_data_processing/neon_downloader.py \
    --site HARV \
    --year 2018 \
    --output_dir /path/to/output
```

**What it does:**
- Downloads NEON data products from the API
- Organizes files by site, year, and modality
- Validates downloads and checks file integrity

**Key parameters:**
- `--site`: NEON site code (e.g., HARV, MLBS, GRSM)
- `--year`: Data collection year
- `--products`: Which products to download (rgb, hsi, lidar)

### 2. Curate Tiles

Quality control and tile selection:

```bash
python processing/neon_data_processing/curate_tiles.py \
    --input_dir /path/to/downloaded/tiles \
    --output_dir /path/to/curated/tiles \
    --quality_threshold 0.8
```

**What it does:**
- Checks for data completeness (all three modalities present)
- Validates spatial alignment between modalities
- Filters out low-quality or corrupted tiles
- Creates metadata about tile quality

**Quality checks:**
- Spatial overlap between RGB, HSI, and LiDAR
- Missing data percentage
- Coordinate system consistency
- File format validation

### 3. Process Shapefiles

Extract crown metadata and validate annotations:

```bash
cd processing/shapefile_processing
python process_shapefiles.py \
    --shapefile_dir /path/to/shapefiles \
    --output_csv crowns_metadata.csv
```

**What it does:**
- Extracts tree crown polygons from shapefiles
- Links crowns to individual tree measurements
- Validates crown annotations
- Merges ecological metadata (height, diameter, etc.)

**Output format:**
- CSV with crown ID, species, location, measurements
- Validated polygon geometries
- Quality flags for each annotation

### 4. Crop Tree Crowns

Extract individual tree crowns from tiles:

```bash
python processing/neon_data_processing/crop_crowns_multimodal.py \
    --tiles_dir /path/to/curated/tiles \
    --crowns_csv crowns_metadata.csv \
    --output_dir /path/to/cropped/crowns \
    --rgb_size 128 \
    --hsi_size 12 \
    --lidar_size 12
```

**What it does:**
- Extracts bounding boxes around each crown
- Crops corresponding regions from RGB, HSI, and LiDAR tiles
- Resamples to target resolutions
- Handles coordinate transformations between modalities

**Parameters:**
- `--rgb_size`: Target RGB resolution (default: 128x128)
- `--hsi_size`: Target HSI resolution (default: 12x12)
- `--lidar_size`: Target LiDAR resolution (default: 12x12)
- `--padding`: Additional padding around crowns (default: 0)

### 5. Convert Formats

Optimize data storage:

```bash
# Convert TIF to NumPy
python processing/neon_data_processing/convert_tif_to_npy.py \
    --input_dir /path/to/cropped/crowns \
    --output_dir /path/to/numpy/arrays

# Convert HSI H5 to TIF
python processing/neon_data_processing/hsi_convert_h5_to_tif.py \
    --input_dir /path/to/hsi/h5 \
    --output_dir /path/to/hsi/tif
```

**What it does:**
- Converts various formats to efficient storage
- Applies compression where appropriate
- Validates converted data

### 6. Generate Training CSV

Create final training/test CSVs:

```bash
python processing/neon_data_processing/create_training_csv.py \
    --crowns_dir /path/to/cropped/crowns \
    --metadata_csv crowns_metadata.csv \
    --output_csv training_dataset.csv
```

**What it does:**
- Combines all metadata
- Validates data availability for each sample
- Adds file paths to HDF5 dataset
- Creates train/val/test splits

### 7. Filter and Combine

Dataset refinement:

```bash
# Filter rare species
python processing/misc/filter_rare_species.py \
    --input_csv training_dataset.csv \
    --output_csv filtered_dataset.csv \
    --min_samples 50

# Combine multiple datasets
python processing/misc/dataset_combiner.py \
    --input_csvs dataset1.csv dataset2.csv dataset3.csv \
    --output_csv combined_dataset.csv
```

**What it does:**
- Removes species with insufficient samples
- Combines datasets from different sites/years
- Ensures species compatibility across datasets
- Creates configuration subsets (large, high_quality, combined)

## Repository Structure

```
processing/
├── neon_data_processing/       # Main processing scripts
│   ├── neon_downloader.py      # Download NEON data
│   ├── curate_tiles.py         # Quality control
│   ├── crop_crowns_multimodal.py  # Extract crowns
│   ├── convert_tif_to_npy.py   # Format conversion
│   ├── hsi_convert_h5_to_tif.py   # HSI format conversion
│   └── create_training_csv.py  # Generate training CSVs
├── shapefile_processing/       # Shapefile tools
│   ├── process_shapefiles.py   # Extract crown metadata
│   └── README.md              # Shapefile processing guide
└── misc/                      # Utility scripts
    ├── filter_rare_species.py  # Species filtering
    └── dataset_combiner.py     # Combine datasets
```

## NEON Data Products

### RGB (DP3.30010.001)
**High-Resolution Orthorectified Camera Imagery**
- Resolution: 10cm
- Format: GeoTIFF
- Bands: RGB (3 channels)
- Coverage: Full site mosaics

### Hyperspectral (DP3.30006.002)
**Surface Directional Reflectance**
- Resolution: 1m
- Format: HDF5
- Bands: 426 spectral bands (380-2510 nm)
- Processing: Atmospheric correction applied
- Note: Reduced to 369 bands in dataset (removed noisy bands)

### LiDAR (DP3.30015.001)
**Ecosystem Structure**
- Resolution: 1m
- Format: GeoTIFF
- Data: Canopy Height Model (CHM)
- Derived from: Point cloud classification

## Data Quality Control

### Spatial Alignment

Ensure all modalities are properly aligned:

```python
from processing.utils import check_spatial_alignment

# Verify alignment
aligned = check_spatial_alignment(
    rgb_path='path/to/rgb.tif',
    hsi_path='path/to/hsi.h5',
    lidar_path='path/to/lidar.tif',
    tolerance=0.5  # meters
)

if not aligned:
    print("Warning: Modalities not aligned!")
```

### Missing Data

Handle missing or corrupted data:

```python
from processing.utils import validate_data

# Check data quality
quality = validate_data(
    crown_id='HARV_123',
    rgb_path='path/to/rgb.npy',
    hsi_path='path/to/hsi.npy',
    lidar_path='path/to/lidar.npy'
)

print(f"Quality score: {quality['score']:.2f}")
print(f"Issues: {quality['issues']}")
```

## HDF5 Dataset Creation

Convert processed crowns to HDF5:

```python
import h5py
import numpy as np

# Create HDF5 dataset
with h5py.File('neon_dataset.h5', 'w') as f:
    # Create groups
    rgb_group = f.create_group('rgb')
    hsi_group = f.create_group('hsi')
    lidar_group = f.create_group('lidar')
    
    # Add crown data
    for crown_id, data in processed_crowns.items():
        rgb_group.create_dataset(
            crown_id, 
            data=data['rgb'], 
            compression='gzip',
            compression_opts=9
        )
        hsi_group.create_dataset(
            crown_id, 
            data=data['hsi'], 
            compression='gzip',
            compression_opts=9
        )
        lidar_group.create_dataset(
            crown_id, 
            data=data['lidar'], 
            compression='gzip',
            compression_opts=9
        )
```

## Configuration Subsets

Create different dataset configurations:

### Combined Dataset
All available data (47,971 samples, 167 species)

```bash
python processing/misc/create_config.py \
    --input_csv all_crowns.csv \
    --output_csv combined_dataset.csv \
    --config combined
```

### Large Dataset
Main training set (~42,000 samples, ~162 species)

```bash
python processing/misc/create_config.py \
    --input_csv all_crowns.csv \
    --output_csv large_dataset.csv \
    --config large \
    --min_samples_per_species 50
```

### High Quality Dataset
Curated subset (~5,500 samples, ~96 species)

```bash
python processing/misc/create_config.py \
    --input_csv all_crowns.csv \
    --output_csv high_quality_dataset.csv \
    --config high_quality \
    --quality_threshold 0.9 \
    --min_samples_per_species 100
```

## Processing Best Practices

### 1. Start Small
Process one site first to validate the pipeline:

```bash
# Test with single site
python process_site.sh HARV 2018
```

### 2. Parallel Processing
Use parallel processing for large-scale operations:

```bash
# Process multiple sites in parallel
parallel -j 4 python process_site.sh ::: HARV MLBS GRSM DELA
```

### 3. Disk Space
Monitor disk usage - raw and processed data can be large:
- Raw tiles: ~100GB per site
- Processed crowns: ~50GB per site
- Final HDF5 dataset: ~600MB (compressed)

### 4. Validation
Always validate processed data:

```bash
python processing/utils/validate_dataset.py \
    --csv_path training_dataset.csv \
    --hdf5_path neon_dataset.h5
```

## Troubleshooting

### Issue: Spatial misalignment
**Solution:** Check coordinate reference systems (CRS) and reproject if needed

### Issue: Missing HSI bands
**Solution:** Verify HSI data download and band extraction

### Issue: Corrupted crowns
**Solution:** Increase quality thresholds in curation step

### Issue: Memory errors
**Solution:** Process in batches or use more efficient data types

## Custom Processing

For custom processing workflows:

```python
from processing.pipeline import ProcessingPipeline

# Create custom pipeline
pipeline = ProcessingPipeline(
    sites=['HARV', 'MLBS'],
    years=[2018, 2019, 2020],
    output_dir='custom_dataset'
)

# Configure processing
pipeline.set_quality_threshold(0.85)
pipeline.set_crown_sizes(rgb=224, hsi=16, lidar=16)

# Run pipeline
pipeline.run()
```

## Additional Resources

- [NEON Data Portal](https://data.neonscience.org/)
- [NEON Data Products Catalog](https://data.neonscience.org/data-products/explore)
- [NEON API Documentation](https://data.neonscience.org/data-api)
- [Processing README](../processing/neon_data_processing/README.md)
