#!/usr/bin/env python3
"""
Upload trained NeonTreeClassification models to HuggingFace Hub.

Converts Lightning checkpoints to safetensors format for DeepForest CropModel integration.

Usage:
    python scripts/upload_to_huggingface.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --repo_name "Ritesh313/neon-tree-resnet18-species" \
        --model_type resnet \
        --taxonomic_level species

Requirements:
    pip install huggingface_hub safetensors
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_lightning_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a Lightning checkpoint and extract relevant data."""
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return {
        "state_dict": checkpoint["state_dict"],
        "hyper_parameters": checkpoint.get("hyper_parameters", {}),
        "label_dict": checkpoint.get("label_dict"),
        "numeric_to_label_dict": checkpoint.get("numeric_to_label_dict"),
        "epoch": checkpoint.get("epoch"),
    }


def extract_model_state_dict(
    lightning_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Extract just the model weights from Lightning's state_dict.

    Lightning prefixes model weights with 'model.' - we need to remove this
    for compatibility with standard PyTorch loading.
    """
    model_state_dict = {}
    for key, value in lightning_state_dict.items():
        if key.startswith("model."):
            new_key = key[6:]  # Remove "model." prefix
            model_state_dict[new_key] = value
        else:
            # Keep non-model keys (metrics, etc.) - but typically we skip these
            pass

    return model_state_dict


def create_config(
    checkpoint_data: Dict[str, Any],
    model_type: str,
    model_variant: Optional[str],
    taxonomic_level: str,
    num_classes: int,
) -> Dict[str, Any]:
    """Create config.json for the HuggingFace model."""
    config = {
        "model_type": model_type,
        "model_variant": model_variant,
        "taxonomic_level": taxonomic_level,
        "num_classes": num_classes,
        "label_dict": checkpoint_data["label_dict"],
        "numeric_to_label_dict": {
            str(k): v for k, v in checkpoint_data["numeric_to_label_dict"].items()
        },  # JSON requires string keys
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "input_size": [224, 224],
        "training_info": {
            "epoch": checkpoint_data.get("epoch"),
            "framework": "pytorch-lightning",
            "dataset": "NEON Tree Crown Dataset",
            "dataset_size": 47971,
        },
    }
    return config


def create_model_card(
    model_type: str,
    model_variant: Optional[str],
    taxonomic_level: str,
    num_classes: int,
    repo_name: str,
) -> str:
    """Create README.md model card for HuggingFace."""

    model_name = model_variant if model_variant else model_type

    card = f"""---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - tree-species-classification
  - ecology
  - neon
  - deepforest
  - crop-model
---

# NEON Tree {taxonomic_level.capitalize()} Classification - {model_name.upper()}

A {model_name} model trained for tree {taxonomic_level} classification on the NEON Tree Crown Dataset.
This model is designed for integration with [DeepForest](https://github.com/weecology/DeepForest) as a CropModel.

## Model Details

- **Architecture**: {model_name}
- **Task**: Tree {taxonomic_level} classification
- **Classes**: {num_classes} {taxonomic_level} classes
- **Input size**: 224x224 RGB images
- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Dataset**: NEON Tree Crown Dataset (~48,000 tree crowns from 30 NEON sites)

## Usage with DeepForest

```python
from deepforest import CropModel

# Load model
model = CropModel.load_model("{repo_name}")

# Use with DeepForest predictions
# (after running detection with main DeepForest model)
results = model.predict(image_crops)
```

## Direct PyTorch Usage

```python
import torch
from safetensors.torch import load_file
from torchvision import transforms

# Load model weights
state_dict = load_file("model.safetensors")

# Load config for label mapping
import json
with open("config.json") as f:
    config = json.load(f)

# Create your model architecture and load weights
# model.load_state_dict(state_dict)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Training Details

- **Framework**: PyTorch Lightning
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3
- **Scheduler**: ReduceLROnPlateau
- **Data Split**: 70/15/15 (train/val/test)
- **Seed**: 42

## Dataset

The model was trained on the NEON Tree Crown Dataset, which includes:
- 47,971 individual tree crowns
- 167 species / 60 genera
- 30 NEON sites across North America
- Multi-modal data: RGB, Hyperspectral, LiDAR (this model uses RGB only)

## Citation

If you use this model, please cite:

```bibtex
@software{{neontreeclassification,
  author = {{Chowdhry, Ritesh}},
  title = {{NeonTreeClassification: Multi-modal Tree Species Classification}},
  url = {{https://github.com/Ritesh313/NeonTreeClassification}},
  year = {{2026}}
}}
```

## License

MIT License
"""
    return card


def upload_to_huggingface(
    checkpoint_path: str,
    repo_name: str,
    model_type: str,
    model_variant: Optional[str],
    taxonomic_level: str,
    private: bool = False,
    dry_run: bool = False,
):
    """Upload model to HuggingFace Hub."""

    try:
        from huggingface_hub import HfApi, create_repo
        from safetensors.torch import save_file
    except ImportError:
        print("❌ Please install: pip install huggingface_hub safetensors")
        sys.exit(1)

    # Load checkpoint
    checkpoint_data = load_lightning_checkpoint(checkpoint_path)

    # Validate label_dict exists
    if not checkpoint_data["label_dict"]:
        print(
            "❌ Checkpoint missing label_dict! Was the model trained with idx_to_label?"
        )
        sys.exit(1)

    if not checkpoint_data["numeric_to_label_dict"]:
        print(
            "❌ Checkpoint missing numeric_to_label_dict! "
            "Was the model trained with idx_to_label?"
        )
        sys.exit(1)

    num_classes = len(checkpoint_data["label_dict"])
    print(f"✅ Found {num_classes} classes in label_dict")

    # Extract model weights
    model_state_dict = extract_model_state_dict(checkpoint_data["state_dict"])
    print(f"✅ Extracted {len(model_state_dict)} model parameters")

    # Create config
    config = create_config(
        checkpoint_data, model_type, model_variant, taxonomic_level, num_classes
    )

    # Create model card
    model_card = create_model_card(
        model_type, model_variant, taxonomic_level, num_classes, repo_name
    )

    if dry_run:
        print("\n🔍 DRY RUN - Would upload:")
        print(f"   Repository: {repo_name}")
        print(f"   Model type: {model_type}")
        print(f"   Model variant: {model_variant}")
        print(f"   Taxonomic level: {taxonomic_level}")
        print(f"   Num classes: {num_classes}")
        print(f"   Parameters: {sum(p.numel() for p in model_state_dict.values()):,}")
        print(f"\n   Config preview:")
        print(f"   - label_dict sample: {dict(list(config['label_dict'].items())[:3])}")
        print(f"   - normalize: {config['normalize']}")
        return

    # Create temp directory for files
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save safetensors
        safetensors_path = tmpdir / "model.safetensors"
        save_file(model_state_dict, str(safetensors_path))
        print(f"✅ Saved safetensors: {safetensors_path.stat().st_size / 1e6:.1f} MB")

        # Save config
        config_path = tmpdir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved config.json")

        # Save model card
        readme_path = tmpdir / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card)
        print(f"✅ Saved README.md")

        # Upload to HuggingFace
        api = HfApi()

        # Create repo
        print(f"\n🚀 Creating/updating repo: {repo_name}")
        create_repo(repo_name, exist_ok=True, private=private)

        # Upload files
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_name,
            commit_message=f"Upload {model_type} {taxonomic_level} model",
        )

        print(f"\n✅ Successfully uploaded to: https://huggingface.co/{repo_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload NeonTreeClassification models to HuggingFace Hub"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="HuggingFace repo name (e.g., 'Ritesh313/neon-tree-resnet18-species')",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["resnet", "vit"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default=None,
        help="Model variant (e.g., 'resnet18', 'vit_b_16')",
    )
    parser.add_argument(
        "--taxonomic_level",
        type=str,
        required=True,
        choices=["species", "genus"],
        help="Taxonomic level the model was trained for",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't actually upload, just show what would be uploaded",
    )

    args = parser.parse_args()

    upload_to_huggingface(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo_name,
        model_type=args.model_type,
        model_variant=args.model_variant,
        taxonomic_level=args.taxonomic_level,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
