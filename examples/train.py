#!/usr/bin/env python3
"""
Training script for NEON tree species classification.

Usage examples:
  # Train RGB classifier
  python train.py --modality rgb --model_type resnet --csv_path /path/to/metadata.csv --hdf5_path /path/to/data.h5

  # Train at genus level (60 classes instead of 167 species)
  python train.py --modality rgb --model_type resnet --taxonomic_level genus --csv_path /path/to/metadata.csv --hdf5_path /path/to/data.h5

  # Train HSI classifier with custom params
  python train.py --modality hsi --model_type spectral_cnn --lr 5e-4 --batch_size 16 --csv_path /path/to/metadata.csv --hdf5_path /path/to/data.h5

  # Train LiDAR classifier
  python train.py --modality lidar --model_type height_cnn --epochs 30 --csv_path /path/to/metadata.csv --hdf5_path /path/to/data.h5

  # Train with Comet logging and tags for experiment organization
  python train.py --modality rgb --model_type resnet --logger comet --tags "baseline,v1,rgb" --csv_path /path/to/metadata.csv --hdf5_path /path/to/data.h5

  # Train with multiple tags (useful for SLURM job arrays or hyperparameter sweeps)
  python train.py --modality hsi --model_type spectral_cnn --logger comet --tags "experiment_1,hyperparams_sweep,hsi" --csv_path /path/to/metadata.csv --hdf5_path /path/to/data.h5

  # Reproducible training with custom seed
  python train.py --modality rgb --model_type resnet --seed 12345 --csv_path /path/to/metadata.csv --hdf5_path /path/to/data.h5
"""

import argparse
import os
import sys
from datetime import datetime
import random
import numpy as np
import lightning as L

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import torch

# Optimize CUDA performance for Tensor Cores
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

try:
    from lightning.pytorch.loggers import CometLogger

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

from neon_tree_classification.core.datamodule import NeonCrownDataModule
from neon_tree_classification.models.lightning_modules import (
    RGBClassifier,
    HSIClassifier,
    LiDARClassifier,
)


def set_seed_everything(seed: int):
    """
    Set all random seeds for reproducible training.

    Uses Lightning's seed_everything() plus additional CUDA determinism settings
    for complete reproducibility across PyTorch, NumPy, Python, and CUDA operations.
    """
    print(f"🌱 Setting global seed: {seed}")

    # Lightning handles: torch, numpy, python, cuda seeds
    L.seed_everything(seed)

    # Additional CUDA determinism (not handled by Lightning)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("   ✓ CUDA determinism enabled")

    return seed


def worker_init_fn(worker_id):
    """Initialize DataLoader worker with unique but reproducible seed."""
    # Get the base seed from the global state or pass it somehow
    # For now, we'll use a global variable approach
    base_seed = getattr(worker_init_fn, "base_seed", 42)
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_callbacks(args, results_path):
    """Create improved callbacks for training with better checkpointing and early stopping."""
    callbacks = []

    class CleanModelCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
        def _save_checkpoint(self, trainer, filepath):
            # Save the checkpoint
            super()._save_checkpoint(trainer, filepath)

            # Get the current val_loss
            val_loss = trainer.callback_metrics.get("val_loss", "N/A")
            epoch = trainer.current_epoch
            print(f"\n✓ Best checkpoint saved: epoch {epoch}, val_loss {val_loss:.4f}")

    # Best validation loss checkpoint
    callbacks.append(
        CleanModelCheckpoint(
            dirpath=os.path.join(results_path, "checkpoints"),
            filename="best_val_loss-{epoch:02d}-{val_loss:.2f}",
            verbose=False,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    )

    # Last epoch checkpoint
    callbacks.append(
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=os.path.join(results_path, "checkpoints"),
            filename="last_epoch-{epoch:02d}",
            verbose=False,
            save_top_k=1,  # Will save the last one
            monitor=None,
            mode="min",
        )
    )

    # Early stopping
    callbacks.append(
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=args.early_stop_patience,
            verbose=False,
            mode="min",
        )
    )

    # Learning rate monitoring
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return callbacks


def main():

    parser = argparse.ArgumentParser(description="Train NEON tree species classifier")

    # Data arguments
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to crown CSV file"
    )
    parser.add_argument(
        "--hdf5_path", type=str, required=True, help="Path to HDF5 data file"
    )
    parser.add_argument(
        "--modality", type=str, choices=["rgb", "hsi", "lidar"], required=True
    )

    # External test dataset arguments (optional)
    parser.add_argument(
        "--external_test_csv",
        type=str,
        default=None,
        help="External test CSV (if not provided, splits test from main dataset)",
    )
    parser.add_argument(
        "--external_test_hdf5",
        type=str,
        default=None,
        help="External test HDF5 (uses main HDF5 if not provided)",
    )

    # Model arguments
    parser.add_argument(
        "--model_type", type=str, default="simple", help="Model architecture type"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default=None,
        help="Model variant (e.g., 'vit_b_16', 'vit_l_16' for ViT models)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of tree species classes (auto-detected if not provided)",
    )
    parser.add_argument(
        "--num_bands",
        type=int,
        default=None,
        help="Number of HSI bands (HSI only, auto-detected if not specified)",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"]
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "step"],
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--early_stop_patience", type=int, default=15, help="Early stopping patience"
    )

    # Data split arguments
    parser.add_argument(
        "--split_method", type=str, default="random", choices=["random", "site", "year"]
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Training split ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Validation split ratio"
    )
    parser.add_argument(
        "--split_seed", type=int, default=42, help="Random seed for splits"
    )
    parser.add_argument(
        "--taxonomic_level",
        type=str,
        default="species",
        choices=["species", "genus"],
        help="Taxonomic level for classification: 'species' (167 classes) or 'genus' (60 classes)",
    )
    parser.add_argument(
        "--use_balanced_sampler",
        action="store_true",
        help="Use WeightedRandomSampler for balanced class sampling (recommended for imbalanced datasets)",
    )

    # Image size arguments
    parser.add_argument(
        "--rgb_size",
        type=int,
        default=224,
        help="RGB image size (single value for square images, e.g., 224 for 224x224). Default matches ImageNet pretraining.",
    )

    # Normalization arguments
    parser.add_argument(
        "--rgb_norm_method",
        type=str,
        default="imagenet",
        choices=["none", "0_1", "imagenet"],
        help="RGB normalization method: 'imagenet' (recommended for pretrained models), '0_1' (simple [0,1] range), 'none'",
    )

    # Reproducibility arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducible training (affects model init, optimizers, CUDA ops, etc.)",
    )

    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loader workers",  # Increased default
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training (PyTorch 2.0+)",
    )

    # Logging arguments
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "comet"]
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save logs, checkpoints, and results (auto-generated if not provided)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags for Comet experiment (e.g., 'baseline,v1,test'). Only used with --logger comet",
    )

    args = parser.parse_args()

    # Set up reproducible training (must be done early, before any model/data operations)
    set_seed_everything(args.seed)

    # Use main seed for split_seed if not explicitly changed from default
    if args.split_seed == 42 and args.seed != 42:
        args.split_seed = args.seed
        print(f"   ✓ Using main seed for data splits: {args.split_seed}")

    # Store seed for DataLoader worker initialization
    worker_init_fn.base_seed = args.seed

    # Set up experiment name (auto-generate)
    # Include model_variant and taxonomic_level to avoid collisions in array jobs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_variant if args.model_variant else args.model_type
    experiment_name = f"{args.modality}_{model_name}_{args.taxonomic_level}_{timestamp}"

    # Set up output directory with dynamic naming within provided path
    if args.output_dir is None:
        base_output_dir = "./outputs"
    else:
        base_output_dir = args.output_dir

    # Create dynamic experiment directory within the base path
    experiment_output_dir = os.path.join(base_output_dir, experiment_name)
    args.output_dir = experiment_output_dir

    print(f"🌲 Training {args.modality.upper()} classifier: {args.model_type}")
    print(f"📁 HDF5 Data: {args.hdf5_path}")
    print(f"🧪 Experiment: {experiment_name}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🌱 Global seed: {args.seed}")
    if args.split_seed != args.seed:
        print(f"🔀 Split seed: {args.split_seed} (different from global seed)")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data module
    datamodule = NeonCrownDataModule(
        csv_path=args.csv_path,
        hdf5_path=args.hdf5_path,  # Updated parameter name
        external_test_csv_path=args.external_test_csv,  # External test support
        external_test_hdf5_path=args.external_test_hdf5,  # External test support
        modalities=[args.modality],
        rgb_size=(args.rgb_size, args.rgb_size),  # Image size for RGB
        rgb_norm_method=args.rgb_norm_method,  # Normalization for RGB (imagenet for pretrained models)
        taxonomic_level=args.taxonomic_level,  # Species or genus level
        use_balanced_sampler=args.use_balanced_sampler,  # Balanced sampling
        split_method=args.split_method,
        use_validation=True,  # Always use validation in this script
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        # Performance optimizations
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Prefetch more batches
        drop_last=True,  # Consistent batch sizes
        worker_init_fn=worker_init_fn,  # For reproducible DataLoader workers
    )

    # Setup data to get number of classes
    datamodule.setup()

    # Auto-detect or validate number of classes
    actual_num_classes = (
        datamodule.full_dataset.num_classes
    )  # Changed from datamodule.num_classes
    if args.num_classes is None:
        args.num_classes = actual_num_classes
        print(f"🔍 Auto-detected {args.num_classes} classes")
    else:
        if args.num_classes != actual_num_classes:
            raise ValueError(
                f"Mismatch: You specified {args.num_classes} classes, but dataset has {actual_num_classes} classes"
            )
        print(f"✅ Verified {args.num_classes} classes match dataset")

    # Auto-detect number of bands for HSI data
    if args.modality == "hsi" and args.num_bands is None:
        # Load a sample to get the number of bands
        sample_idx = 0  # First sample
        sample_data = datamodule.full_dataset[sample_idx]
        sample_tensor = sample_data[args.modality]  # Get HSI tensor
        actual_num_bands = sample_tensor.shape[0]  # First dimension is bands
        args.num_bands = actual_num_bands
        print(f"🔍 Auto-detected {args.num_bands} HSI bands from data")
    elif args.modality == "hsi":
        # Verify specified bands match data
        sample_idx = 0
        sample_data = datamodule.full_dataset[sample_idx]
        sample_tensor = sample_data[args.modality]
        actual_num_bands = sample_tensor.shape[0]
        if args.num_bands != actual_num_bands:
            raise ValueError(
                f"Mismatch: You specified {args.num_bands} bands, but data has {actual_num_bands} bands"
            )
        print(f"✅ Verified {args.num_bands} bands match dataset")

    # Create classifier based on modality
    if args.modality == "rgb":
        # Prepare model kwargs
        model_kwargs = {}
        if args.model_variant is not None:
            model_kwargs["model_variant"] = args.model_variant

        classifier = RGBClassifier(
            model_type=args.model_type,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
            log_images=True,  # Enable image logging for RGB
            idx_to_label=datamodule.full_dataset.idx_to_label,  # For DeepForest CropModel compatibility
            **model_kwargs,  # Pass model variant for ViT and other models
        )
    elif args.modality == "hsi":
        classifier = HSIClassifier(
            model_type=args.model_type,
            num_bands=args.num_bands,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
        )
    elif args.modality == "lidar":
        classifier = LiDARClassifier(
            model_type=args.model_type,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
        )

    # NOTE: torch.compile disabled due to compatibility issues with Lightning metrics
    # See: https://github.com/Lightning-AI/lightning/issues/17122
    if args.compile and hasattr(torch, "compile"):
        print("⚠️  torch.compile disabled - causes issues with Lightning metrics")
        print("   Training will proceed without compilation")
        # classifier = torch.compile(classifier)  # Disabled

    # Set up logger
    if args.logger == "comet":
        if not COMET_AVAILABLE:
            raise ImportError(
                "CometML not available. Install with: pip install comet-ml"
            )

        # Parse tags if provided
        comet_tags = []
        if args.tags:
            comet_tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]

        # Always add seed to tags for reproducibility tracking
        comet_tags.append(f"seed_{args.seed}")

        print(f"🏷️  Comet tags: {comet_tags}")

        logger = CometLogger(
            save_dir=args.output_dir,
            experiment_name=experiment_name,
        )

        # Add tags to the experiment after creation
        logger.experiment.add_tags(comet_tags)
    else:
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name="",  # Empty name since experiment_name is already in the path
            version="",  # No version subfolder
        )

    # Log dataset info
    dataset_info = datamodule.get_dataset_info()
    print(
        f"📈 Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val, {dataset_info['test_samples']} test"
    )

    # Calculate smart logging interval to avoid warnings
    train_batches = len(datamodule.train_dataloader())
    log_every_n_steps = min(
        50, max(1, train_batches // 4)
    )  # Log 4 times per epoch, max 50
    print(
        f"📊 Training batches per epoch: {train_batches}, logging every {log_every_n_steps} steps"
    )

    # Set up callbacks with improved checkpointing and early stopping
    callbacks = create_callbacks(args, args.output_dir)

    # Create trainer with performance optimizations
    trainer = L.Trainer(
        default_root_dir=args.output_dir,  # Set Lightning logs directory
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() and args.gpus > 0 else "cpu",
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else 1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,  # Dynamic logging interval
        enable_progress_bar=False,  # Disabled for cleaner SLURM logs
        precision="32",  # Use 32-bit for better reproducibility (was "16-mixed")
    )

    # Get class weights for imbalanced datasets
    class_weights = datamodule.get_class_weights()
    if class_weights is not None:
        classifier.class_weights = class_weights
        print(f"📊 Using class weights for {len(class_weights)} classes")

    # Train
    print("🚀 Starting training...")
    trainer.fit(classifier, datamodule)

    # Test
    print("🧪 Testing best model...")
    trainer.test(classifier, datamodule, ckpt_path="best")

    print(
        f"✅ Training complete! Best model saved to: {trainer.checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    main()
