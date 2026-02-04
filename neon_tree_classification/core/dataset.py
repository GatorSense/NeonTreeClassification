"""
NEON Tree Classification Dataset - Optimized Version

Fast, efficient dataset for NEON tree crown classification using NPY files.
Designed for maximum training performance with minimal CPU bottlenecks.

Author: Ritesh Chowdhry
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any, Callable, Union, Tuple


class NeonCrownDataset(Dataset):
    """
    Dataset for NEON tree crown classification.

    Features:
    - Fast NPY file loading (3-5x faster than TIF)
    - Built-in resizing with optimal interpolation methods
    - Configurable normalization with performance-first defaults
    - Efficient filtering applied during initialization
    - Training-ready tensors with proper dtypes
    - Optional custom transforms per modality
    - Metadata support for debugging

    Expected CSV format:
    crown_id,site,year,plot,species,easting,northing,rgb_path,hsi_path,lidar_path
    """

    def __init__(
        self,
        csv_path: str,
        hdf5_path: str,
        modalities: List[str] = ["rgb"],
        # Filtering (applied once during init for efficiency)
        species_filter: Optional[List[str]] = None,
        site_filter: Optional[List[str]] = None,
        year_filter: Optional[List[int]] = None,
        # Target sizes for training (required for consistent batching)
        rgb_size: Tuple[int, int] = (128, 128),
        hsi_size: Tuple[int, int] = (12, 12),
        lidar_size: Tuple[int, int] = (12, 12),
        # Resize methods (optimized for speed)
        rgb_resize_mode: str = "nearest",  # Fastest for RGB images
        hsi_resize_mode: str = "nearest",  # Changed to nearest for speed
        lidar_resize_mode: str = "nearest",  # Changed to nearest for speed
        # Normalization methods (performance-first defaults)
        rgb_norm_method: str = "0_1",  # Simple division, fastest
        hsi_norm_method: str = "per_sample",  # Per-sample z-score, faster than per_pixel
        lidar_norm_method: str = "height",  # Simple max scaling, fastest
        # Custom transforms (optional, per-modality)
        custom_transforms: Optional[Dict[str, Callable]] = None,
        # External consistency (for test/validation datasets)
        label_to_idx: Optional[Dict[str, int]] = None,  # Use external label mapping
        normalization_stats: Optional[
            Dict[str, Any]
        ] = None,  # Use external normalization stats
        # Return format options
        include_metadata: bool = False,  # Add crown_id, species, site for debugging
        # Internal options
        validate_hdf5: bool = True,  # Check HDF5 file exists and validate structure
    ):
        """
        Initialize optimized NEON dataset with HDF5 backend.

        Args:
            csv_path: Path to CSV with crown metadata (must contain 'crown_id' column)
            hdf5_path: Path to HDF5 file containing data organized as {modality}/{crown_id}
                      Expected structure: h5_file[modality][crown_id] -> numpy array
            modalities: List of modalities ['rgb', 'hsi', 'lidar']
            species_filter: Optional list of species codes to include
            site_filter: Optional list of site codes to include
            year_filter: Optional list of years to include
            rgb_size: Target (H, W) for RGB images
            hsi_size: Target (H, W) for HSI images
            lidar_size: Target (H, W) for LiDAR images
            rgb_resize_mode: Interpolation mode for RGB ('nearest', 'bilinear')
            hsi_resize_mode: Interpolation mode for HSI ('nearest', 'bilinear')
            lidar_resize_mode: Interpolation mode for LiDAR ('nearest', 'bilinear')
            rgb_norm_method: RGB normalization ('none', '0_1', 'imagenet')
            hsi_norm_method: HSI normalization ('none', 'per_sample', 'per_band', 'per_pixel', 'global_minmax', 'global_zscore')
            lidar_norm_method: LiDAR normalization ('none', 'height', 'zscore')
            custom_transforms: Optional per-modality transform functions
            label_to_idx: Optional external label mapping for consistency with training data
            normalization_stats: Optional external normalization stats for consistency with training data
            include_metadata: Whether to include debugging metadata in samples
            validate_hdf5: Whether to validate HDF5 file exists and structure
        """
        self.csv_path = csv_path
        self.hdf5_path = hdf5_path
        self.modalities = modalities
        self.include_metadata = include_metadata

        # Store size and method configurations
        self.sizes = {"rgb": rgb_size, "hsi": hsi_size, "lidar": lidar_size}

        self.resize_modes = {
            "rgb": rgb_resize_mode,
            "hsi": hsi_resize_mode,
            "lidar": lidar_resize_mode,
        }

        self.norm_methods = {
            "rgb": rgb_norm_method,
            "hsi": hsi_norm_method,
            "lidar": lidar_norm_method,
        }

        self.custom_transforms = custom_transforms or {}

        # Store external consistency parameters
        self.external_label_mapping = label_to_idx
        self.external_norm_stats = normalization_stats

        # Validate modalities
        valid_modalities = {"rgb", "hsi", "lidar"}
        for mod in modalities:
            if mod not in valid_modalities:
                raise ValueError(f"Unknown modality: {mod}. Valid: {valid_modalities}")

        # Load and filter data (done once for efficiency)
        print(f"Loading dataset from {csv_path}...")
        self.data = self._load_and_filter_data(species_filter, site_filter, year_filter)

        # Create or use external label mapping
        if self.external_label_mapping is not None:
            # Use provided label mapping for consistency
            self.label_to_idx = self.external_label_mapping
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            self.num_classes = len(self.label_to_idx)
            print(f"Using external label mapping with {self.num_classes} classes")

            # Validate that all species in data exist in external mapping
            self._validate_species_consistency()
        else:
            # Create new label mapping from data
            self.label_to_idx, self.idx_to_label = self._create_label_mapping()
            self.num_classes = len(self.label_to_idx)

        # Pre-compute normalization statistics or use external ones
        if self.external_norm_stats is not None:
            # Use provided normalization stats for consistency
            self.norm_stats = self.external_norm_stats
            print("Using external normalization statistics")
        else:
            # Compute new normalization stats from data
            self.norm_stats = self._precompute_normalization_stats()

        # Open HDF5 file for data access (keep open during dataset lifetime)
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, "r")
            print(f"✅ Opened HDF5 file: {self.hdf5_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to open HDF5 file {self.hdf5_path}: {e}")

        # Validate HDF5 structure if requested
        if validate_hdf5:
            print("Validating HDF5 structure...")
            self._validate_hdf5_structure()

        print(f"✅ NeonCrownDataset initialized:")
        print(f"   Modalities: {modalities}")
        print(f"   Samples: {len(self.data):,}")
        print(f"   Classes: {self.num_classes}")
        print(
            f"   Target sizes: RGB{self.sizes['rgb']} HSI{self.sizes['hsi']} LiDAR{self.sizes['lidar']}"
        )

    def __del__(self):
        """Clean up: close HDF5 file when dataset is destroyed."""
        if hasattr(self, "hdf5_file") and self.hdf5_file is not None:
            try:
                self.hdf5_file.close()
            except:
                pass  # Ignore errors during cleanup

    def _load_and_filter_data(
        self,
        species_filter: Optional[List[str]],
        site_filter: Optional[List[str]],
        year_filter: Optional[List[int]],
    ) -> pd.DataFrame:
        """Load CSV and apply all filters efficiently."""
        # Load CSV
        df = pd.read_csv(self.csv_path)
        print(f"   Loaded {len(df):,} total samples")

        # Apply species filter
        if species_filter is not None:
            df = df[df["species"].isin(species_filter)]
            print(f"   After species filter: {len(df):,} samples")

        # Apply site filter
        if site_filter is not None:
            df = df[df["site"].isin(site_filter)]
            print(f"   After site filter: {len(df):,} samples")

        # Apply year filter
        if year_filter is not None:
            df = df[df["year"].isin(year_filter)]
            print(f"   After year filter: {len(df):,} samples")

        # Filter for samples with all requested modalities available
        df = self._filter_complete_samples(df)
        print(f"   After completeness filter: {len(df):,} samples")

        # Reset index for fast iloc access
        df = df.reset_index(drop=True)

        return df

    def _filter_complete_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only samples that have all requested modalities."""
        modality_columns = {"rgb": "rgb_path", "hsi": "hsi_path", "lidar": "lidar_path"}

        for modality in self.modalities:
            col = modality_columns[modality]
            if col in df.columns:
                # Keep rows where path is not empty/NaN
                df = df[df[col].notna() & (df[col] != "")]
        return df

    def _create_label_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create bidirectional species label mapping."""
        if "species" not in self.data.columns:
            return {}, {}

        unique_species = sorted(self.data["species"].dropna().unique())
        label_to_idx = {species: idx for idx, species in enumerate(unique_species)}
        idx_to_label = {idx: species for species, idx in label_to_idx.items()}

        return label_to_idx, idx_to_label

    def _validate_species_consistency(self) -> None:
        """Validate that all species in dataset exist in the external label mapping."""
        if "species" not in self.data.columns:
            return

        data_species = set(self.data["species"].dropna().unique())
        mapping_labels = set(self.label_to_idx.keys())

        # Check if mapping contains species codes or genus names
        # If the first mapping key is a species code (all uppercase, short), it's species-level
        # If it's a genus name (capitalized, longer), it's genus-level
        sample_label = next(iter(mapping_labels)) if mapping_labels else ""
        is_genus_mapping = sample_label and sample_label[0].isupper() and sample_label[1:].islower()
        
        if is_genus_mapping:
            # Genus-level mapping: validate that all species have extractable genus
            if "species_name" not in self.data.columns:
                raise ValueError(
                    "Genus-level mapping detected but 'species_name' column not found in data. "
                    "Cannot extract genus from species names."
                )
            
            # Extract genera from species names and check they're all in mapping
            data_genera = set(self.data["species_name"].apply(lambda x: str(x).split()[0]).unique())
            missing_genera = data_genera - mapping_labels
            if missing_genera:
                raise ValueError(
                    f"Genera extracted from dataset not found in external label mapping: {sorted(missing_genera)}. "
                    f"External mapping has: {sorted(mapping_labels)}"
                )
            
            print(f"✓ Genus-level validation passed: All {len(data_genera)} genera found in mapping")
        else:
            # Species-level mapping: check species codes
            missing_in_mapping = data_species - mapping_labels
            if missing_in_mapping:
                raise ValueError(
                    f"Species in dataset not found in external label mapping: {sorted(missing_in_mapping)}. "
                    f"External mapping has: {sorted(mapping_labels)}"
                )

            # Check for species in mapping that are not in data (warning only)
            missing_in_data = mapping_labels - data_species
            if missing_in_data:
                print(
                    f"⚠️  Species in external mapping not found in dataset: {sorted(missing_in_data)}"
                )

    def _precompute_normalization_stats(self) -> Dict[str, Any]:
        """Pre-compute dataset-wide normalization statistics for global methods."""
        stats = {}

        # Only compute stats for modalities that need global normalization
        for modality in self.modalities:
            method = self.norm_methods[modality]

            if method in ["global_minmax", "global_zscore"]:
                print(f"   Computing global {method} statistics for {modality}...")

                # OPTIMIZED: Use much smaller sample for speed (100 instead of 1000)
                # For global stats, 100 samples is usually sufficient for good estimates
                n_samples = min(100, len(self.data))
                sample_indices = np.random.choice(
                    len(self.data), n_samples, replace=False
                )

                # OPTIMIZED: Use online/streaming statistics to avoid memory issues
                if method == "global_minmax":
                    global_min = float("inf")
                    global_max = float("-inf")
                    total_values = 0

                    for idx in sample_indices:
                        crown_id = str(self.data.iloc[idx]["crown_id"])

                        try:
                            data = self.hdf5_file[modality][crown_id][:]
                            data_min = float(np.min(data))
                            data_max = float(np.max(data))
                            global_min = min(global_min, data_min)
                            global_max = max(global_max, data_max)
                            total_values += data.size
                        except:
                            continue  # Skip missing data

                    if total_values > 0:
                        stats[f"{modality}_global_min"] = global_min
                        stats[f"{modality}_global_max"] = global_max
                        print(
                            f"      {modality} global min/max computed from {total_values:,} values"
                        )

                elif method == "global_zscore":
                    # OPTIMIZED: Use Welford's online algorithm for mean/std
                    count = 0
                    mean = 0.0
                    M2 = 0.0  # Sum of squares of deviations

                    for idx in sample_indices:
                        crown_id = str(self.data.iloc[idx]["crown_id"])

                        try:
                            data = self.hdf5_file[modality][crown_id][:].flatten()

                            # Welford's online algorithm
                            for value in data:
                                count += 1
                                delta = value - mean
                                mean += delta / count
                                delta2 = value - mean
                                M2 += delta * delta2

                        except:
                            continue  # Skip missing files

                    if count > 1:
                        variance = M2 / count  # Population variance
                        std = np.sqrt(variance)
                        stats[f"{modality}_global_mean"] = float(mean)
                        stats[f"{modality}_global_std"] = float(std)
                        print(
                            f"      {modality} global mean/std computed from {count:,} values"
                        )

        return stats

    def _validate_hdf5_structure(self) -> None:
        """Validate that HDF5 file has the expected structure and data."""
        missing_data = []

        # Check that all required modalities exist
        for modality in self.modalities:
            if modality not in self.hdf5_file:
                raise KeyError(f"Modality '{modality}' not found in HDF5 file")

        # Sample check: verify some crown_ids exist
        sample_size = min(10, len(self.data))
        for idx in range(sample_size):
            row = self.data.iloc[idx]
            crown_id = str(row["crown_id"])

            for modality in self.modalities:
                if crown_id not in self.hdf5_file[modality]:
                    missing_data.append(f"{modality}/{crown_id}")

                if len(missing_data) > 10:  # Don't spam too many errors
                    break

        if missing_data:
            print(f"❌ Missing {len(missing_data)} HDF5 datasets (showing first 10):")
            for path in missing_data[:10]:
                print(f"   {path}")
            raise KeyError(
                f"Missing {len(missing_data)} required datasets in HDF5 file"
            )

        print(f"✅ HDF5 structure validated ({sample_size} samples checked)")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training-ready sample.

        Returns:
            Dictionary containing:
            - {modality}: torch.Tensor for each requested modality
            - species_idx: int (species class index)
            - crown_id: str (if include_metadata=True)
            - species: str (if include_metadata=True)
            - site: str (if include_metadata=True)
        """
        row = self.data.iloc[idx]
        crown_id = row["crown_id"]
        sample = {}

        # Load each modality
        for modality in self.modalities:
            # Fast HDF5 loading from pre-opened file
            try:
                data = self.hdf5_file[modality][str(crown_id)][:]
                tensor = torch.from_numpy(data)
            except Exception as e:
                raise KeyError(f"Failed to load {modality}/{crown_id} from HDF5: {e}")

            # Apply built-in resize (optimized)
            target_size = self.sizes[modality]
            resize_mode = self.resize_modes[modality]
            tensor = self._resize_tensor(tensor, target_size, resize_mode)

            # Apply normalization (performance-optimized)
            norm_method = self.norm_methods[modality]
            tensor = self._normalize_tensor(
                tensor, modality, norm_method
            )  # Apply custom transform if provided
            if modality in self.custom_transforms:
                tensor = self.custom_transforms[modality](tensor)

            sample[modality] = tensor

        # Add label (using "species_idx" to match Lightning module expectations)
        if "species" in row and pd.notna(row["species"]):
            # Check if we're using genus-level labels
            # (external label mapping contains genus names, not species codes)
            if self.external_label_mapping is not None:
                # Using external mapping - need to determine if it's genus or species level
                # Check by seeing if species code exists in mapping
                if row["species"] in self.label_to_idx:
                    # Species-level mapping
                    sample["species_idx"] = self.label_to_idx[row["species"]]
                else:
                    # Genus-level mapping - extract genus from species_name
                    genus = str(row["species_name"]).split()[0]
                    if genus in self.label_to_idx:
                        sample["species_idx"] = self.label_to_idx[genus]
                    else:
                        raise KeyError(
                            f"Genus '{genus}' (from species_name '{row['species_name']}') "
                            f"not found in label mapping. Available genera: {list(self.label_to_idx.keys())[:10]}..."
                        )
            else:
                # Using internal mapping (species codes)
                sample["species_idx"] = self.label_to_idx[row["species"]]

        # Add metadata if requested
        if self.include_metadata:
            sample.update(
                {
                    "crown_id": crown_id,
                    "species": row.get("species", ""),
                    "site": row.get("site", ""),
                }
            )

        return sample

    def _resize_tensor(
        self, tensor: torch.Tensor, target_size: Tuple[int, int], mode: str
    ) -> torch.Tensor:
        """Resize tensor using F.interpolate."""
        # Handle different tensor shapes
        if tensor.ndim == 2:  # Add channel dimension
            tensor = tensor.unsqueeze(0)

        current_size = tensor.shape[-2:]  # (H, W)

        if current_size == target_size:
            return tensor  # Already correct size

        # Use F.interpolate for GPU-friendly resizing
        resized = F.interpolate(
            tensor.unsqueeze(0).float(),  # Add batch dim, ensure float
            size=target_size,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        ).squeeze(
            0
        )  # Remove batch dim

        return resized

    def _normalize_tensor(
        self, tensor: torch.Tensor, modality: str, method: str
    ) -> torch.Tensor:
        """Normalize tensor with minimal CPU overhead."""
        if method == "none":
            return tensor

        # RGB normalization
        if modality == "rgb":
            if method == "0_1":
                return tensor.float() / 255.0  # Simple division, fastest
            elif method == "imagenet":
                # ImageNet normalization
                tensor = tensor.float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                return (tensor - mean) / std

        # HSI normalization
        elif modality == "hsi":
            tensor = tensor.float()
            if method == "per_sample":
                # Per-sample z-score normalization
                mean = tensor.mean()
                std = tensor.std()
                std = torch.clamp(std, min=1e-8)
                return (tensor - mean) / std
            elif method == "per_band":
                # Per-band z-score
                mean = tensor.mean(dim=(-2, -1), keepdim=True)
                std = tensor.std(dim=(-2, -1), keepdim=True)
                std = torch.clamp(std, min=1e-8)
                return (tensor - mean) / std
            elif method == "per_pixel":
                # Per-pixel L2 normalization
                bands, height, width = tensor.shape
                reshaped = tensor.view(bands, -1)
                norms = torch.norm(reshaped, dim=0, keepdim=True)
                norms = torch.clamp(norms, min=1e-8)
                normalized = reshaped / norms
                return normalized.view(bands, height, width)
            elif method == "global_minmax":
                # Global min-max normalization using dataset-wide statistics
                global_min = self.norm_stats[f"{modality}_global_min"]
                global_max = self.norm_stats[f"{modality}_global_max"]
                if global_max > global_min:
                    return (tensor - global_min) / (global_max - global_min)
                else:
                    return tensor
            elif method == "global_zscore":
                # Global z-score normalization using dataset-wide statistics
                global_mean = self.norm_stats[f"{modality}_global_mean"]
                global_std = self.norm_stats[f"{modality}_global_std"]
                if global_std > 1e-8:
                    return (tensor - global_mean) / global_std
                else:
                    return tensor - global_mean

        # LiDAR normalization
        elif modality == "lidar":
            tensor = tensor.float()
            if method == "height":
                # Simple height normalization (fastest)
                valid_mask = tensor > 0  # Exclude NoData/ground
                if valid_mask.any():
                    max_height = tensor[valid_mask].max()
                    return torch.clamp(tensor / max_height, 0, 1)
                else:
                    return tensor
            elif method == "zscore":
                # Z-score for valid data only
                valid_mask = tensor > 0
                if valid_mask.any():
                    valid_data = tensor[valid_mask]
                    mean = valid_data.mean()
                    std = valid_data.std()
                    std = torch.clamp(std, min=1e-8)
                    normalized = tensor.clone()
                    normalized[valid_mask] = (tensor[valid_mask] - mean) / std
                    return normalized
                else:
                    return tensor

        # If method not recognized, raise error (fail fast as requested)
        raise ValueError(
            f"Unknown normalization method '{method}' for modality '{modality}'"
        )

    # Utility methods for dataset exploration and debugging

    def get_species_list(self) -> List[str]:
        """Get list of unique species in dataset."""
        return list(self.idx_to_label.values())

    def get_sites(self) -> List[str]:
        """Get list of unique sites in dataset."""
        return sorted(self.data["site"].unique().tolist())

    def get_years(self) -> List[int]:
        """Get list of unique years in dataset."""
        return sorted(self.data["year"].unique().tolist())

    @property
    def normalization_stats(self) -> Dict[str, Any]:
        """Get normalization statistics for sharing with other datasets."""
        return self.norm_stats.copy()

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample."""
        row = self.data.iloc[idx]
        return {
            "crown_id": row["crown_id"],
            "site": row["site"],
            "year": row["year"],
            "species": row.get("species", ""),
            "species_idx": self.label_to_idx.get(row.get("species", ""), -1),
            "coordinates": (row.get("easting", 0), row.get("northing", 0)),
        }

    def summary(self) -> None:
        """Print dataset summary."""
        print("=" * 60)
        print("NEON Tree Crown Dataset Summary")
        print("=" * 60)
        print(f"Total samples: {len(self):,}")
        print(f"Species: {self.num_classes}")
        print(f"Sites: {len(self.get_sites())}")
        print(f"Years: {min(self.get_years())}-{max(self.get_years())}")
        print(f"Modalities: {self.modalities}")

        # Size configuration
        print(f"\nTarget sizes:")
        for mod in self.modalities:
            print(
                f"  {mod.upper()}: {self.sizes[mod]} ({self.resize_modes[mod]} interpolation)"
            )

        # Normalization configuration
        print(f"\nNormalization:")
        for mod in self.modalities:
            print(f"  {mod.upper()}: {self.norm_methods[mod]}")

        # Top species
        if self.num_classes > 0:
            species_counts = self.data["species"].value_counts()
            print(f"\nTop 5 species:")
            for i, (species, count) in enumerate(species_counts.head(5).items()):
                percentage = count / len(self) * 100
                print(f"  {i+1}. {species}: {count:,} ({percentage:.1f}%)")

        print("=" * 60)

    @classmethod
    def load(
        cls,
        csv_path: str = "training_data_clean.csv",
        npy_base_dir: str = "cropped_crowns_npy/",  # Relative to downloaded data directory
        modalities: List[str] = ["rgb"],
        **kwargs,
    ) -> "NeonCrownDataset":
        """
        Load dataset with sensible defaults for quick exploration.

        This method assumes the standard NEON dataset structure:
        - CSV with crown metadata and crown_id column
        - NPY directory structure: {npy_base_dir}/{modality}/{crown_id}.npy

        Args:
            csv_path: Path to crown metadata CSV
            npy_base_dir: Path to NPY data directory (relative or absolute)
            modalities: List of modalities to load
            **kwargs: Additional arguments passed to constructor

        Returns:
            NeonCrownDataset instance ready for use

        Example:
            >>> # Quick RGB dataset (assumes standard structure)
            >>> dataset = NeonCrownDataset.load()
            >>> dataset.summary()
            >>>
            >>> # Custom paths for downloaded dataset
            >>> dataset = NeonCrownDataset.load(
            ...     csv_path="downloaded_data/metadata.csv",
            ...     npy_base_dir="downloaded_data/npy_files/",
            ...     modalities=['rgb', 'hsi']
            ... )
        """
        return cls(
            csv_path=csv_path,
            npy_base_dir=npy_base_dir,
            modalities=modalities,
            **kwargs,
        )
