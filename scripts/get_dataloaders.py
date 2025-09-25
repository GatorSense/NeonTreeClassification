"""
Simple DataLoader factory for NEON Tree Classification Dataset.

Provides direct PyTorch DataLoaders without requiring Lightning knowledge.
Supports all 4 use cases with the downloaded dataset.

Usage:
    from scripts.get_dataloaders import get_dataloaders

    # Train on large, test on high-quality
    train_loader, test_loader = get_dataloaders(
        train_config='large',
        test_config='high_quality',
        modalities=['rgb']
    )

    # Train and test on same config
    train_loader, test_loader = get_dataloaders(
        config='combined',
        modalities=['rgb', 'hsi']
    )
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import sys
import zipfile
from urllib.request import urlretrieve
import h5py

# Add parent directory to path to import dataset
sys.path.append(str(Path(__file__).parent.parent))
from neon_tree_classification.core.dataset import NeonCrownDataset


DATASET_URL = "https://www.dropbox.com/scl/fi/v49xi6d7wtetctqphebx0/neon_tree_classification_dataset.zip?rlkey=fb7bz6kd0ckip4u0qd5xdor58&st=dvjyd5ry&dl=1"


def _dataset_exists(dataset_dir: Path) -> bool:
    """Check if dataset files exist and are valid."""
    hdf5_path = dataset_dir / "neon_dataset.h5"
    metadata_dir = dataset_dir / "metadata"

    if not (hdf5_path.exists() and metadata_dir.exists()):
        return False

    # Check if HDF5 file can be opened
    try:
        with h5py.File(hdf5_path, "r") as f:
            pass
        return True
    except:
        return False


def _auto_download_if_needed(dataset_dir: Path):
    """Download and extract dataset."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_dir / "dataset.zip"

    try:

        def progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size / total_size) * 100)
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)

        print(f"Downloading from: {DATASET_URL}")
        urlretrieve(DATASET_URL, zip_path, progress)
        print(f"\n📦 Downloaded {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

        print("Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dataset_dir)

        zip_path.unlink()  # Clean up

        # Validate download
        if not _dataset_exists(dataset_dir):
            raise RuntimeError("Downloaded dataset appears corrupted")

        print("✅ Dataset ready!")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(
            f"Could not download dataset. Please download manually from {DATASET_URL}"
        )


def get_dataloaders(
    # Configuration options
    config: Optional[str] = None,  # Use same config for train/test
    train_config: Optional[str] = None,  # Separate train config
    test_config: Optional[str] = None,  # Separate test config
    # Data parameters
    modalities: List[str] = ["rgb"],
    batch_size: int = 32,
    test_ratio: float = 0.2,
    # Dataset location (auto-detect by default)
    dataset_dir: Optional[str] = None,
    auto_download: bool = True,
    # DataLoader parameters
    num_workers: int = 4,
    shuffle_train: bool = True,
    # Dataset parameters
    rgb_size: Tuple[int, int] = (128, 128),
    hsi_size: Tuple[int, int] = (12, 12),
    lidar_size: Tuple[int, int] = (12, 12),
    rgb_norm_method: str = "0_1",
    hsi_norm_method: str = "per_sample",
    lidar_norm_method: str = "height",
) -> Tuple[DataLoader, DataLoader]:
    """
    Get PyTorch DataLoaders for NEON tree classification.

    Args:
        config: Use same dataset config for both train/test ('large', 'high_quality', 'combined')
        train_config: Training dataset config (if different from test)
        test_config: Test dataset config (if different from train)
        modalities: List of modalities ['rgb', 'hsi', 'lidar']
        batch_size: Batch size for DataLoaders
        test_ratio: Fraction of data to use for testing (0.0-1.0)
        dataset_dir: Path to dataset directory (auto-detect if None)
        auto_download: Automatically download dataset if missing
        num_workers: Number of DataLoader workers
        shuffle_train: Whether to shuffle training data
        rgb_size: Target size for RGB images
        hsi_size: Target size for HSI images
        lidar_size: Target size for LiDAR images
        *_norm_method: Normalization methods for each modality

    Returns:
        Tuple of (train_loader, test_loader)

    Examples:
        # Use case 1: Train on large, test on high-quality
        train_loader, test_loader = get_dataloaders(
            train_config='large',
            test_config='high_quality'
        )

        # Use case 2: Train and test on large only
        train_loader, test_loader = get_dataloaders(config='large')

        # Use case 3: Train and test on high-quality only
        train_loader, test_loader = get_dataloaders(config='high_quality')

        # Use case 4: Train and test on combined
        train_loader, test_loader = get_dataloaders(config='combined')
    """

    # Validate config parameters
    if config is not None and (train_config is not None or test_config is not None):
        raise ValueError(
            "Use either 'config' OR 'train_config'/'test_config', not both"
        )

    if config is None and (train_config is None or test_config is None):
        raise ValueError(
            "Must specify either 'config' OR both 'train_config' and 'test_config'"
        )

    # Set train/test configs
    if config is not None:
        train_config = config
        test_config = config

    # Auto-detect dataset directory
    if dataset_dir is None:
        dataset_dir = _find_dataset_dir()

    dataset_dir = Path(dataset_dir)

    # Auto-download if enabled and data missing
    if auto_download and not _dataset_exists(dataset_dir):
        print("📦 Dataset not found. Downloading...")
        _auto_download_if_needed(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Get file paths
    hdf5_path = dataset_dir / "neon_dataset.h5"
    train_csv_path = dataset_dir / "metadata" / f"{train_config}_dataset.csv"
    test_csv_path = dataset_dir / "metadata" / f"{test_config}_dataset.csv"

    # Validate files exist
    for path in [hdf5_path, train_csv_path, test_csv_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"🔥 Setting up DataLoaders:")
    print(f"   Train config: {train_config}")
    print(f"   Test config: {test_config}")
    print(f"   Modalities: {modalities}")
    print(f"   Batch size: {batch_size}")

    # Create datasets
    if train_config == test_config:
        # Same config: create one dataset and split it
        dataset = NeonCrownDataset(
            csv_path=str(train_csv_path),
            hdf5_path=str(hdf5_path),
            modalities=modalities,
            rgb_size=rgb_size,
            hsi_size=hsi_size,
            lidar_size=lidar_size,
            rgb_norm_method=rgb_norm_method,
            hsi_norm_method=hsi_norm_method,
            lidar_norm_method=lidar_norm_method,
        )

        # Split dataset
        test_size = int(len(dataset) * test_ratio)
        train_size = len(dataset) - test_size

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),  # Reproducible splits
        )

        print(
            f"   Dataset split: {len(train_dataset):,} train, {len(test_dataset):,} test"
        )

    else:
        # Different configs: create separate datasets
        train_dataset = NeonCrownDataset(
            csv_path=str(train_csv_path),
            hdf5_path=str(hdf5_path),
            modalities=modalities,
            rgb_size=rgb_size,
            hsi_size=hsi_size,
            lidar_size=lidar_size,
            rgb_norm_method=rgb_norm_method,
            hsi_norm_method=hsi_norm_method,
            lidar_norm_method=lidar_norm_method,
        )

        # Get compatible species between train and test datasets
        compatible_species = _get_compatible_species(train_csv_path, test_csv_path)
        print(f"   Compatible species: {len(compatible_species)}")

        # Create test dataset with compatible species filter
        test_dataset = NeonCrownDataset(
            csv_path=str(test_csv_path),
            hdf5_path=str(hdf5_path),
            modalities=modalities,
            species_filter=compatible_species,  # Filter to compatible species only
            rgb_size=rgb_size,
            hsi_size=hsi_size,
            lidar_size=lidar_size,
            rgb_norm_method=rgb_norm_method,
            hsi_norm_method=hsi_norm_method,
            lidar_norm_method=lidar_norm_method,
            # Use training dataset's mappings for consistency
            label_to_idx=train_dataset.label_to_idx,
            normalization_stats=train_dataset.normalization_stats,
        )

        print(f"   Train dataset: {len(train_dataset):,} samples")
        print(f"   Test dataset: {len(test_dataset):,} samples")  # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print(f"✅ DataLoaders ready!")

    return train_loader, test_loader


def _get_compatible_species(train_csv_path: str, test_csv_path: str) -> List[str]:
    """
    Get species for test dataset filtering (intersection approach).

    Logic matches datamodule.py:
    - Keep ALL species from training (no filtering on train)
    - Filter test dataset to only species that exist in training
    - Return intersection (overlapping species only)
    """
    # Read species columns only (fast CSV I/O)
    train_species = set(
        pd.read_csv(train_csv_path, usecols=["species"])["species"].dropna().unique()
    )
    test_df = pd.read_csv(test_csv_path, usecols=["species", "crown_id"])
    test_species = set(test_df["species"].dropna().unique())

    # Find overlapping species (intersection)
    overlapping_species = train_species.intersection(test_species)

    # Count test samples that would remain after filtering
    test_samples_original = len(test_df)
    test_samples_filtered = len(test_df[test_df["species"].isin(overlapping_species)])
    overlap_percentage = test_samples_filtered / test_samples_original * 100

    # Report species filtering (same as datamodule)
    train_only = train_species - test_species
    test_only = test_species - train_species

    print(f"   Train species: {len(train_species)}")
    print(f"   Test species: {len(test_species)}")
    print(f"   Overlapping species: {len(overlapping_species)}")
    print(
        f"   Test samples: {test_samples_original:,} → {test_samples_filtered:,} ({overlap_percentage:.1f}%)"
    )

    if test_only:
        print(
            f"   ⚠️  Filtering {len(test_only)} species from test set not in training: {sorted(list(test_only))[:5]}{'...' if len(test_only) > 5 else ''}"
        )

    if train_only:
        print(
            f"   ℹ️  Training has {len(train_only)} species not in test set (keeping in training)"
        )

    # Warn if low overlap but don't fail (unlike datamodule which fails at 50%)
    if overlap_percentage < 50.0:
        print(
            f"   ⚠️  Low species overlap: Only {overlap_percentage:.1f}% of test samples retained"
        )
    else:
        print(
            f"   ✅ Species compatibility: {overlap_percentage:.1f}% test samples retained"
        )

    return sorted(overlapping_species)


def _find_dataset_dir() -> str:
    """Auto-detect dataset directory."""
    script_dir = Path(__file__).parent

    # Check common locations
    candidates = [
        script_dir.parent
        / "_neon_tree_classification_dataset_files",  # ../_neon_tree_classification_dataset_files
        script_dir
        / "_neon_tree_classification_dataset_files",  # ./_neon_tree_classification_dataset_files
        Path(
            "_neon_tree_classification_dataset_files"
        ),  # ./_neon_tree_classification_dataset_files from current dir
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "neon_dataset.h5").exists():
            return str(candidate)

    # If auto_download will be used, return default location
    return str(script_dir.parent / "_neon_tree_classification_dataset_files")


def get_dataset_info(dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    """Get information about available dataset configurations."""
    if dataset_dir is None:
        dataset_dir = _find_dataset_dir()

    dataset_dir = Path(dataset_dir)
    metadata_dir = dataset_dir / "metadata"

    info = {
        "dataset_dir": str(dataset_dir),
        "configurations": {},
        "hdf5_size_mb": 0,
    }

    # Get HDF5 file size
    hdf5_path = dataset_dir / "neon_dataset.h5"
    if hdf5_path.exists():
        info["hdf5_size_mb"] = hdf5_path.stat().st_size / 1024 / 1024

    # Get info for each CSV
    if metadata_dir.exists():
        for csv_file in metadata_dir.glob("*_dataset.csv"):
            config_name = csv_file.stem.replace("_dataset", "")

            # Quick line count
            with open(csv_file) as f:
                line_count = sum(1 for _ in f) - 1  # Subtract header

            info["configurations"][config_name] = {
                "samples": line_count,
                "csv_path": str(csv_file),
            }

    return info


if __name__ == "__main__":
    # Example usage and testing
    print("NEON Tree Classification DataLoader Factory")
    print("=" * 50)

    # Show dataset info
    try:
        info = get_dataset_info()
        print(f"Dataset directory: {info['dataset_dir']}")
        print(f"HDF5 file size: {info['hdf5_size_mb']:.1f} MB")
        print(f"\nAvailable configurations:")
        for name, config in info["configurations"].items():
            print(f"  {name}: {config['samples']:,} samples")
        print()

        # Test case 1: Train on large, test on high_quality
        print("Test case 1: Train on large, test on high_quality")
        train_loader, test_loader = get_dataloaders(
            train_config="large", test_config="high_quality", batch_size=16
        )
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        print()

    except Exception as e:
        print(f"Error: {e}")
        print(
            "\nDataset will download automatically when you first call get_dataloaders()"
        )
