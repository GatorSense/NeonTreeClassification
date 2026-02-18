"""
NEON Tree Classification DataModule - Simple Version

Simple PyTorch Lightning DataModule for NEON tree crown classification.
Splits everything from one dataset - no external test set complexity.

Author: Ritesh Chowdhry
"""

import os
import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import List, Optional, Dict, Any, Callable, Tuple
import warnings

from .dataset import NeonCrownDataset


class NeonCrownDataModule(LightningDataModule):
    """
    Simple Lightning DataModule for NEON tree crown classification.

    Features:
    - Everything splits from one dataset (no external test complexity)
    - Efficient data splitting using Subset (no dataset duplication)
    - Optimized DataLoader configuration for maximum performance
    - Class weight calculation for imbalanced datasets
    - Support for site/year-based splitting strategies
    - Clean parameter passing to underlying dataset

    Usage:
        >>> # Train/Val/Test from one dataset
        >>> datamodule = NeonCrownDataModule(
        ...     csv_path="data.csv",
        ...     npy_base_dir="data/npy/",
        ...     modalities=["rgb"],
        ...     batch_size=32,
        ...     use_validation=True,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15
        ... )
        >>> trainer.fit(model, datamodule)
        >>> trainer.test(model, datamodule)
    """

    def __init__(
        self,
        csv_path: str,
        hdf5_path: str,
        modalities: List[str] = ["rgb"],
        # External test dataset parameters (optional)
        external_test_csv_path: Optional[str] = None,
        external_test_hdf5_path: Optional[str] = None,
        # Dataset parameters (passed through to NeonCrownDataset)
        species_filter: Optional[List[str]] = None,
        site_filter: Optional[List[str]] = None,
        year_filter: Optional[List[int]] = None,
        rgb_size: Tuple[int, int] = (224, 224),  # Matches ImageNet pretraining
        hsi_size: Tuple[int, int] = (12, 12),
        lidar_size: Tuple[int, int] = (12, 12),
        rgb_resize_mode: str = "nearest",
        hsi_resize_mode: str = "nearest",
        lidar_resize_mode: str = "nearest",
        rgb_norm_method: str = "imagenet",  # ImageNet normalization for pretrained models
        hsi_norm_method: str = "per_sample",
        lidar_norm_method: str = "height",
        custom_transforms: Optional[Dict[str, Callable]] = None,
        include_metadata: bool = False,
        validate_hdf5: bool = True,
        # DataModule-specific parameters
        taxonomic_level: str = "species",  # "species" or "genus"
        use_validation: bool = True,  # Whether to split validation from training
        val_ratio: float = 0.15,  # Validation split ratio
        test_ratio: float = 0.15,  # Test split ratio
        split_method: str = "random",  # "random", "site", "year"
        split_seed: int = 42,
        use_balanced_sampler: bool = False,  # Use WeightedRandomSampler for class balance
        # DataLoader parameters
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        drop_last: bool = True,  # Consistent batch sizes for training
        worker_init_fn: Optional[
            Callable
        ] = None,  # For reproducible DataLoader workers
    ):
        """
        Initialize DataModule.

        Args:
            csv_path: Path to crown metadata CSV
            npy_base_dir: Base directory containing NPY files
            modalities: List of modalities to load ['rgb', 'hsi', 'lidar']

            # Dataset parameters - see NeonCrownDataset for details
            species_filter: Optional species filter
            site_filter: Optional site filter
            year_filter: Optional year filter
            rgb_size: RGB target size
            hsi_size: HSI target size
            lidar_size: LiDAR target size
            *_resize_mode: Resize interpolation methods
            *_norm_method: Normalization methods
            custom_transforms: Custom transform functions
            include_metadata: Include debugging metadata
            validate_hdf5: Validate HDF5 file structure during init

            # DataModule parameters
            taxonomic_level: Classification level - "species" (167 classes) or "genus" (60 classes)
            use_validation: Whether to create validation split
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            split_method: How to split data ("random", "site", "year")
            split_seed: Random seed for reproducible splits

            # DataLoader parameters
            batch_size: Batch size for all DataLoaders
            num_workers: Number of DataLoader workers
            pin_memory: Enable memory pinning for GPU transfer
            persistent_workers: Keep workers alive between epochs (faster)
            prefetch_factor: Batches to prefetch per worker
            drop_last: Drop incomplete final batch (training only)
            worker_init_fn: Function to initialize DataLoader workers (for reproducibility)
        """
        super().__init__()

        # Store parameters
        self.csv_path = csv_path
        self.hdf5_path = hdf5_path
        self.modalities = modalities

        # External test dataset parameters
        self.external_test_csv_path = external_test_csv_path
        self.external_test_hdf5_path = external_test_hdf5_path

        # Taxonomic level
        if taxonomic_level not in ["species", "genus"]:
            raise ValueError(
                f"taxonomic_level must be 'species' or 'genus', got '{taxonomic_level}'"
            )
        self.taxonomic_level = taxonomic_level

        # Dataset parameters
        self.dataset_params = {
            "csv_path": csv_path,
            "hdf5_path": hdf5_path,
            "modalities": modalities,
            "species_filter": species_filter,
            "site_filter": site_filter,
            "year_filter": year_filter,
            "rgb_size": rgb_size,
            "hsi_size": hsi_size,
            "lidar_size": lidar_size,
            "rgb_resize_mode": rgb_resize_mode,
            "hsi_resize_mode": hsi_resize_mode,
            "lidar_resize_mode": lidar_resize_mode,
            "rgb_norm_method": rgb_norm_method,
            "hsi_norm_method": hsi_norm_method,
            "lidar_norm_method": lidar_norm_method,
            "custom_transforms": custom_transforms,
            "include_metadata": include_metadata,
            "validate_hdf5": validate_hdf5,
        }

        # Split parameters
        self.use_validation = use_validation
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_method = split_method
        self.split_seed = split_seed
        self.use_balanced_sampler = use_balanced_sampler

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn

        # Validate split ratios
        total_ratio = (val_ratio if use_validation else 0.0) + test_ratio
        if total_ratio >= 1.0:
            raise ValueError(
                f"val_ratio + test_ratio must be < 1.0, got {total_ratio:.4f}"
            )

        # Will be set in setup()
        self.full_dataset = None  # Full dataset
        self.train_dataset = None  # Training split
        self.val_dataset = None  # Validation split (if enabled)
        self.test_dataset = None  # Test split
        self._setup_done = False

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for Lightning stages.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if self._setup_done:
            print("⚡ DataModule already set up, skipping...")
            return

        print("🔄 Setting up DataModule...")

        # Detect and route to appropriate setup mode
        if self.external_test_csv_path is not None:
            # External test mode: separate datasets for train and test
            self._setup_external_test_mode()
        else:
            # Single dataset mode: split everything from one dataset
            self._setup_single_dataset_mode()

        # Print summary
        print(f"✅ DataModule setup complete:")
        print(f"   Train: {len(self.train_dataset):,} samples")
        print(f"   Val: {len(self.val_dataset) if self.val_dataset else 0:,} samples")
        print(f"   Test: {len(self.test_dataset):,} samples")
        print(f"   Classes: {self.full_dataset.num_classes}")

        self._setup_done = True

    def _split_train_val_test(self) -> Tuple[List[int], List[int], List[int]]:
        """Split data into train/val/test."""
        n_samples = len(self.full_dataset)
        indices = np.arange(n_samples)

        # Set random seed
        np.random.seed(self.split_seed)

        # Get stratification labels if available
        stratify_labels = self._get_stratification_labels(indices)

        if self.split_method == "random":
            # First split: training vs (val + test)
            train_size = 1.0 - self.val_ratio - self.test_ratio
            train_indices, temp_indices = train_test_split(
                indices,
                train_size=train_size,
                random_state=self.split_seed,
                stratify=stratify_labels,
            )

            # Second split: val vs test from temp_indices
            val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
            temp_stratify = None
            if stratify_labels is not None:
                temp_stratify = [
                    stratify_labels[np.where(indices == i)[0][0]] for i in temp_indices
                ]

            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_size,
                random_state=self.split_seed + 1,
                stratify=temp_stratify,
            )

        elif self.split_method == "site":
            train_indices, val_indices, test_indices = self._split_by_site_three_way()
        elif self.split_method == "year":
            train_indices, val_indices, test_indices = self._split_by_year_three_way()
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

        return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()

    def _split_train_test(self) -> Tuple[List[int], List[int]]:
        """Split data into train/test."""
        n_samples = len(self.full_dataset)
        indices = np.arange(n_samples)

        np.random.seed(self.split_seed)
        stratify_labels = self._get_stratification_labels(indices)

        if self.split_method == "random":
            train_indices, test_indices = train_test_split(
                indices,
                test_size=self.test_ratio,
                random_state=self.split_seed,
                stratify=stratify_labels,
            )
        elif self.split_method == "site":
            train_indices, test_indices = self._split_by_site_two_way()
        elif self.split_method == "year":
            train_indices, test_indices = self._split_by_year_two_way()
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

        return train_indices.tolist(), test_indices.tolist()

    def _get_stratification_labels(self, indices: np.ndarray) -> Optional[List[str]]:
        """Get species labels for stratification."""
        try:
            species_labels = [
                self.full_dataset.data.iloc[i]["species"] for i in indices
            ]
            return species_labels if len(set(species_labels)) > 1 else None
        except:
            return None

    def _split_by_site_three_way(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split by site into train/val/test."""
        sites = self.full_dataset.get_sites()
        n_sites = len(sites)

        n_val_sites = max(1, int(n_sites * self.val_ratio))
        n_test_sites = max(1, int(n_sites * self.test_ratio))

        sites_shuffled = np.array(sites)
        np.random.shuffle(sites_shuffled)

        val_sites = sites_shuffled[:n_val_sites]
        test_sites = sites_shuffled[n_val_sites : n_val_sites + n_test_sites]
        train_sites = sites_shuffled[n_val_sites + n_test_sites :]

        df = self.full_dataset.data
        train_indices = df[df["site"].isin(train_sites)].index.values
        val_indices = df[df["site"].isin(val_sites)].index.values
        test_indices = df[df["site"].isin(test_sites)].index.values

        return train_indices, val_indices, test_indices

    def _split_by_year_three_way(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split by year into train/val/test."""
        years = self.full_dataset.get_years()
        n_years = len(years)

        n_val_years = max(1, int(n_years * self.val_ratio))
        n_test_years = max(1, int(n_years * self.test_ratio))

        years_shuffled = np.array(years)
        np.random.shuffle(years_shuffled)

        val_years = years_shuffled[:n_val_years]
        test_years = years_shuffled[n_val_years : n_val_years + n_test_years]
        train_years = years_shuffled[n_val_years + n_test_years :]

        df = self.full_dataset.data
        train_indices = df[df["year"].isin(train_years)].index.values
        val_indices = df[df["year"].isin(val_years)].index.values
        test_indices = df[df["year"].isin(test_years)].index.values

        return train_indices, val_indices, test_indices

    def _split_by_site_two_way(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split by site into train/test."""
        sites = self.full_dataset.get_sites()
        n_sites = len(sites)
        n_test_sites = max(1, int(n_sites * self.test_ratio))

        sites_shuffled = np.array(sites)
        np.random.shuffle(sites_shuffled)

        test_sites = sites_shuffled[:n_test_sites]
        train_sites = sites_shuffled[n_test_sites:]

        df = self.full_dataset.data
        train_indices = df[df["site"].isin(train_sites)].index.values
        test_indices = df[df["site"].isin(test_sites)].index.values

        return train_indices, test_indices

    def _split_by_year_two_way(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split by year into train/test."""
        years = self.full_dataset.get_years()
        n_years = len(years)
        n_test_years = max(1, int(n_years * self.test_ratio))

        years_shuffled = np.array(years)
        np.random.shuffle(years_shuffled)

        test_years = years_shuffled[:n_test_years]
        train_years = years_shuffled[n_test_years:]

        df = self.full_dataset.data
        train_indices = df[df["year"].isin(train_years)].index.values
        test_indices = df[df["year"].isin(test_years)].index.values

        return train_indices, test_indices

    def _check_species_compatibility(
        self, train_csv_path: str, test_csv_path: str
    ) -> List[str]:
        """
        Check species compatibility between train and test datasets.

        Returns list of species that should be used for test dataset filtering.
        Fails if less than 50% of test samples would remain after filtering.

        Args:
            train_csv_path: Path to training CSV
            test_csv_path: Path to test CSV

        Returns:
            List of species to use for test dataset filtering

        Raises:
            ValueError: If less than 50% overlap in test samples
        """
        print("🔍 Checking species compatibility between train and test datasets...")

        # Read species columns only (fast CSV I/O)
        train_species = set(
            pd.read_csv(train_csv_path, usecols=["species"])["species"]
            .dropna()
            .unique()
        )
        test_df = pd.read_csv(test_csv_path, usecols=["species", "crown_id"])
        test_species = set(test_df["species"].dropna().unique())

        # Find overlapping species
        overlapping_species = train_species.intersection(test_species)

        # Count test samples that would remain after filtering
        test_samples_original = len(test_df)
        test_samples_filtered = len(
            test_df[test_df["species"].isin(overlapping_species)]
        )
        overlap_percentage = test_samples_filtered / test_samples_original * 100

        # Report species filtering
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

        # Check overlap threshold
        if overlap_percentage < 50.0:
            raise ValueError(
                f"❌ Species compatibility check failed: Only {overlap_percentage:.1f}% of test samples "
                f"would remain after filtering ({test_samples_filtered:,}/{test_samples_original:,}). "
                f"Need at least 50% overlap."
            )

        print(
            f"✅ Species compatibility check passed: {overlap_percentage:.1f}% test samples retained"
        )
        return sorted(overlapping_species)

    def _setup_external_test_mode(self) -> None:
        """Setup DataModule with external test dataset."""
        print("🔧 Setting up external test mode...")

        # Check species compatibility and get filtered species list
        compatible_species = self._check_species_compatibility(
            self.csv_path, self.external_test_csv_path
        )

        # Extract label mapping based on taxonomic level
        if self.taxonomic_level == "genus":
            print(f"📊 Extracting genus-level labels from training data...")
            label_to_idx = self._create_genus_label_mapping()
            self.dataset_params["label_to_idx"] = label_to_idx
            print(f"   Found {len(label_to_idx)} unique genera")

        # Create training dataset (full species set from training CSV)
        print("Creating training dataset...")
        train_dataset_params = self.dataset_params.copy()
        self.full_dataset = NeonCrownDataset(**train_dataset_params)

        # Create test dataset with compatible species only and same label mapping
        print("Creating external test dataset...")
        test_dataset_params = self.dataset_params.copy()
        test_dataset_params.update(
            {
                "csv_path": self.external_test_csv_path,
                "hdf5_path": self.external_test_hdf5_path or self.hdf5_path,
                "species_filter": compatible_species,  # Filter to compatible species
                "label_to_idx": self.full_dataset.label_to_idx,  # Use training label mapping
                "normalization_stats": self.full_dataset.normalization_stats,  # Use training normalization
            }
        )
        self.test_dataset = NeonCrownDataset(**test_dataset_params)

        # Split training dataset into train/val (no test from training data)
        if self.use_validation:
            print("Setting up train/val splits from training dataset...")
            # Temporarily use val_ratio in place of test_ratio for the split
            temp_test_ratio = self.test_ratio
            self.test_ratio = self.val_ratio  # Use val_ratio for the split

            train_indices, val_indices = self._split_train_test()

            self.test_ratio = temp_test_ratio  # Restore original test_ratio

            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = Subset(self.full_dataset, val_indices)
        else:
            # Use all training data for training (no validation split)
            train_indices = list(range(len(self.full_dataset)))
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = None

    def _setup_single_dataset_mode(self) -> None:
        """Setup DataModule with single dataset (current behavior)."""
        print("🔧 Setting up single dataset mode...")

        # Extract label mapping based on taxonomic level
        if self.taxonomic_level == "genus":
            print(f"📊 Extracting genus-level labels from species names...")
            label_to_idx = self._create_genus_label_mapping()
            self.dataset_params["label_to_idx"] = label_to_idx
            print(f"   Found {len(label_to_idx)} unique genera")

        # Create full dataset
        print("Creating full dataset...")
        self.full_dataset = NeonCrownDataset(**self.dataset_params)

        # Split the data
        if self.use_validation:
            # Three-way split: train/val/test
            print("Setting up train/val/test splits...")
            train_indices, val_indices, test_indices = self._split_train_val_test()
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = Subset(self.full_dataset, val_indices)
            self.test_dataset = Subset(self.full_dataset, test_indices)
        else:
            # Two-way split: train/test
            print("Setting up train/test splits...")
            train_indices, test_indices = self._split_train_test()
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = None
            self.test_dataset = Subset(self.full_dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        """Create optimized training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not available. Call setup() first.")

        # Compute sampler if balanced sampling is enabled
        sampler = None
        shuffle = True

        if self.use_balanced_sampler:
            print("⚖️  Using WeightedRandomSampler for balanced class sampling")
            sampler = self._create_weighted_sampler()
            shuffle = False  # Can't use shuffle with sampler

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=self.drop_last,  # Consistent batch sizes
            worker_init_fn=self.worker_init_fn if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation DataLoader (if validation enabled)."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=False,  # Keep all validation samples
            worker_init_fn=self.worker_init_fn if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test DataLoader (if test data available)."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for testing
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=False,  # Keep all test samples
            worker_init_fn=self.worker_init_fn if self.num_workers > 0 else None,
        )

    def _create_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Create WeightedRandomSampler for balanced class sampling.

        Computes sample weights inversely proportional to class frequency,
        so rare classes are sampled more often and common classes less often.

        Returns:
            WeightedRandomSampler for training dataset
        """
        # Get training indices and corresponding labels
        if hasattr(self.train_dataset, "indices"):
            # Subset dataset
            train_indices = self.train_dataset.indices
            train_df = self.full_dataset.data.iloc[train_indices]
            full_dataset = self.full_dataset
        else:
            # Full dataset used for training
            train_df = self.train_dataset.data
            full_dataset = self.train_dataset

        # Get label for each sample (handle both species and genus level)
        if self.taxonomic_level == "genus":
            # Extract genus from species_name
            sample_labels = train_df["species_name"].apply(lambda x: str(x).split()[0])
        else:
            # Use species codes directly
            sample_labels = train_df["species"]

        # Count class frequencies
        class_counts = sample_labels.value_counts().to_dict()

        # Compute weight for each class (inverse frequency)
        num_samples = len(sample_labels)
        class_weights = {
            cls: num_samples / count for cls, count in class_counts.items()
        }

        # Assign weight to each sample based on its class
        sample_weights = [class_weights[label] for label in sample_labels]
        sample_weights = torch.DoubleTensor(sample_weights)

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,  # Sample with replacement to oversample rare classes
        )

        print(f"   Created sampler for {len(sample_weights)} samples")
        print(
            f"   Sample weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}"
        )

        return sampler

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of class weights for loss function
        """
        if not self._setup_done:
            raise RuntimeError("Must call setup() before getting class weights")

        if self.train_dataset is None:
            raise RuntimeError("Training dataset not available")

        print("⚖️  Calculating class weights...")

        # Get training indices and corresponding DataFrame rows
        if hasattr(self.train_dataset, "indices"):
            # Subset dataset
            train_indices = self.train_dataset.indices
            train_df = self.full_dataset.data.iloc[train_indices]
            full_dataset = self.full_dataset
        else:
            # Full dataset used for training (shouldn't happen in this simple version)
            train_df = self.train_dataset.data
            full_dataset = self.train_dataset

        # Count labels in training set (handle both species and genus level)
        if self.taxonomic_level == "genus":
            # Extract genus from species_name for each sample
            sample_labels = train_df["species_name"].apply(lambda x: str(x).split()[0])
            label_counts = sample_labels.value_counts()
        else:
            # Use species codes directly
            label_counts = train_df["species"].value_counts()

        # Calculate inverse frequency weights
        total_samples = len(train_df)
        weights = []

        # Ensure weights are in same order as label_to_idx mapping
        for label_idx in range(full_dataset.num_classes):
            label_name = full_dataset.idx_to_label[label_idx]
            count = label_counts.get(label_name, 0)
            if count > 0:
                weight = total_samples / (full_dataset.num_classes * count)
            else:
                weight = 0.0  # No samples of this class in training
            weights.append(weight)

        class_weights = torch.tensor(weights, dtype=torch.float32)

        print(f"📊 Class weights computed for {len(weights)} classes")
        print(f"   Weight range: {class_weights.min():.3f} - {class_weights.max():.3f}")

        return class_weights

    def _create_genus_label_mapping(self) -> Dict[str, int]:
        """
        Create genus-level label mapping from species names in the CSV.

        Extracts genus (first word) from species_name column.

        Returns:
            Dictionary mapping genus name to integer index
        """
        import warnings

        # Load CSV to extract species names
        df = pd.read_csv(self.csv_path)

        # Apply any filters that were specified
        if self.dataset_params.get("species_filter"):
            df = df[df["species"].isin(self.dataset_params["species_filter"])]
        if self.dataset_params.get("site_filter"):
            df = df[df["site"].isin(self.dataset_params["site_filter"])]
        if self.dataset_params.get("year_filter"):
            df = df[df["year"].isin(self.dataset_params["year_filter"])]

        # Extract genus from species_name (first word)
        df["genus"] = df["species_name"].apply(lambda x: str(x).split()[0])

        # Get unique genera and create mapping
        unique_genera = sorted(df["genus"].unique())
        label_to_idx = {genus: idx for idx, genus in enumerate(unique_genera)}

        # Validate genus names and warn about edge cases
        non_alpha_genera = [g for g in unique_genera if not g.isalpha()]
        if non_alpha_genera:
            warnings.warn(
                f"Found non-alphabetic genus names: {non_alpha_genera}. "
                f"These may be unidentified species or family names. "
                f"Run 'python processing/misc/inspect_labels.py' to review. "
                f"To exclude, use: species_filter=[...]"
            )

        # Check for known family names
        known_families = {"Pinaceae", "Rosaceae", "Fabaceae", "Asteraceae"}
        found_families = set(unique_genera) & known_families
        if found_families:
            sample_counts = df[df["genus"].isin(found_families)].groupby("genus").size()
            warnings.warn(
                f"Found family names treated as genera: {dict(sample_counts)}. "
                f"These represent unidentified species within that family. "
                f"See docs/taxonomic_levels.md for more information."
            )

        return label_to_idx

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset and splits."""
        if not self._setup_done:
            raise RuntimeError("Must call setup() first")

        return {
            "total_samples": len(self.full_dataset),
            "num_classes": self.full_dataset.num_classes,
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
            "modalities": self.modalities,
            "split_method": self.split_method,
            "use_validation": self.use_validation,
            "species_list": self.full_dataset.get_species_list(),
            "sites": self.full_dataset.get_sites(),
            "years": self.full_dataset.get_years(),
        }
