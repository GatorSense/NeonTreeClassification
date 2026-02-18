"""
Model registry for NEON tree classification models.

Maintains catalog of available pretrained models and their configurations.
"""

from pathlib import Path
from typing import Dict, Optional, List
import warnings


# Model catalog - will be populated with HuggingFace URLs later
AVAILABLE_MODELS = {
    "resnet_species": {
        "description": "ResNet RGB model for species-level classification (167 classes)",
        "taxonomic_level": "species",
        "num_classes": 167,
        "architecture": "resnet",
        "modality": "rgb",
        "input_size": (128, 128),
        "accuracy": 75.88,  # Test accuracy percentage
        "parameters": "11.2M",
        "url": None,  # To be added when uploaded to HuggingFace
        "local_path_template": "checkpoints/resnet_species_best.ckpt",
    },
    "resnet_genus": {
        "description": "ResNet RGB model for genus-level classification (60 classes)",
        "taxonomic_level": "genus",
        "num_classes": 60,
        "architecture": "resnet",
        "modality": "rgb",
        "input_size": (128, 128),
        "accuracy": 72.24,  # Test accuracy percentage
        "parameters": "11.2M",
        "url": None,  # To be added when uploaded to HuggingFace
        "local_path_template": "checkpoints/resnet_genus_best.ckpt",
    },
}


def get_model_info(model_name: str) -> Dict:
    """
    Get information about a registered model.

    Args:
        model_name: Name of the model (e.g., 'resnet_species')

    Returns:
        Dictionary with model configuration and metadata

    Raises:
        ValueError: If model name is not registered
    """
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")

    return AVAILABLE_MODELS[model_name].copy()


def list_available_models() -> List[str]:
    """
    Get list of all available model names.

    Returns:
        List of model names
    """
    return list(AVAILABLE_MODELS.keys())


def validate_model_name(model_name: str) -> bool:
    """
    Check if model name is valid.

    Args:
        model_name: Name to validate

    Returns:
        True if valid, False otherwise
    """
    return model_name in AVAILABLE_MODELS


def get_models_by_level(taxonomic_level: str) -> List[str]:
    """
    Get all models for a specific taxonomic level.

    Args:
        taxonomic_level: 'species' or 'genus'

    Returns:
        List of model names matching the taxonomic level
    """
    return [
        name
        for name, info in AVAILABLE_MODELS.items()
        if info["taxonomic_level"] == taxonomic_level
    ]


def get_model_checkpoint_path(
    model_name: str, checkpoint_dir: Optional[Path] = None
) -> Path:
    """
    Get the checkpoint path for a model.

    Args:
        model_name: Name of the model
        checkpoint_dir: Directory containing checkpoints (optional)

    Returns:
        Path to checkpoint file

    Raises:
        ValueError: If model not found
        FileNotFoundError: If checkpoint doesn't exist at expected location
    """
    model_info = get_model_info(model_name)

    if checkpoint_dir is None:
        # Use default location relative to project root
        project_root = Path(__file__).parent.parent.parent
        checkpoint_dir = project_root

    checkpoint_path = checkpoint_dir / model_info["local_path_template"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Please download or provide the correct checkpoint_dir."
        )

    return checkpoint_path


def register_model(
    name: str,
    description: str,
    taxonomic_level: str,
    num_classes: int,
    architecture: str,
    **kwargs,
) -> None:
    """
    Register a new model in the catalog.

    Args:
        name: Unique model identifier
        description: Human-readable description
        taxonomic_level: 'species' or 'genus'
        num_classes: Number of output classes
        architecture: Model architecture name
        **kwargs: Additional model metadata

    Raises:
        ValueError: If model name already exists
    """
    if name in AVAILABLE_MODELS:
        raise ValueError(f"Model '{name}' already registered")

    AVAILABLE_MODELS[name] = {
        "description": description,
        "taxonomic_level": taxonomic_level,
        "num_classes": num_classes,
        "architecture": architecture,
        **kwargs,
    }


def print_model_catalog() -> None:
    """Print formatted catalog of available models."""
    print("\n" + "=" * 80)
    print("NEON TREE CLASSIFICATION - AVAILABLE MODELS")
    print("=" * 80)

    for name, info in AVAILABLE_MODELS.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Level: {info['taxonomic_level']} ({info['num_classes']} classes)")
        print(
            f"  Architecture: {info['architecture']} ({info.get('parameters', 'N/A')})"
        )
        print(f"  Input size: {info.get('input_size', 'N/A')}")
        if info.get("accuracy"):
            print(f"  Test accuracy: {info['accuracy']:.2f}%")
        print(
            f"  Status: {'✓ Available online' if info.get('url') else '⚠ Local only'}"
        )

    print("\n" + "=" * 80)


def download_model(
    model_name: str, cache_dir: Optional[Path] = None, force_download: bool = False
) -> Path:
    """
    Download model from HuggingFace Hub (placeholder for future implementation).

    Args:
        model_name: Name of the model to download
        cache_dir: Directory to cache downloaded models
        force_download: Force re-download even if cached

    Returns:
        Path to downloaded checkpoint

    Raises:
        NotImplementedError: Feature not yet implemented
        ValueError: If model doesn't have download URL
    """
    model_info = get_model_info(model_name)

    if model_info["url"] is None:
        raise ValueError(
            f"Model '{model_name}' does not have a download URL yet. "
            f"Please use a local checkpoint file."
        )

    # TODO: Implement HuggingFace Hub download
    raise NotImplementedError(
        "Automatic model download from HuggingFace Hub will be implemented "
        "after models are uploaded. For now, please use local checkpoint files."
    )


def get_label_mapping_path(
    taxonomic_level: str, custom_path: Optional[Path] = None
) -> Path:
    """
    Get path to label mapping JSON file.

    Args:
        taxonomic_level: 'species' or 'genus'
        custom_path: Custom path to label file (optional)

    Returns:
        Path to label mapping JSON

    Raises:
        FileNotFoundError: If label file doesn't exist
    """
    if custom_path is not None:
        path = Path(custom_path)
    else:
        # Default location
        inference_dir = Path(__file__).parent
        filename = f"{taxonomic_level}_labels.json"
        path = inference_dir / "label_mappings" / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Label mapping file not found: {path}. "
            f"Run 'python scripts/create_label_mappings.py --csv_path <path>' to create it."
        )

    return path
