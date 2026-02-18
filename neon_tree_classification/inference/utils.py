"""
Utility functions for inference module.

Handles label loading, prediction formatting, and model extraction.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def load_label_mapping(
    json_path: Union[str, Path], taxonomic_level: str = "species"
) -> Dict:
    """
    Load label mapping from JSON file.

    Args:
        json_path: Path to label JSON file
        taxonomic_level: 'species' or 'genus' (for validation)

    Returns:
        Dictionary with label mappings and metadata

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If taxonomic level doesn't match file
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Validate taxonomic level
    if "metadata" in data:
        file_level = data["metadata"].get("taxonomic_level", "").lower()
        if file_level and file_level != taxonomic_level.lower():
            raise ValueError(
                f"Label file is for {file_level} level, but requested {taxonomic_level} level"
            )

    # Convert string keys to integers for idx_to_* mappings
    if "idx_to_code" in data:
        data["idx_to_code"] = {int(k): v for k, v in data["idx_to_code"].items()}
    if "idx_to_name" in data:
        data["idx_to_name"] = {int(k): v for k, v in data["idx_to_name"].items()}
    if "idx_to_genus" in data:
        data["idx_to_genus"] = {int(k): v for k, v in data["idx_to_genus"].items()}
    if "idx_to_count" in data:
        data["idx_to_count"] = {int(k): v for k, v in data["idx_to_count"].items()}

    return data


def format_predictions(
    logits: torch.Tensor, label_mapping: Dict, top_k: int = 5, temperature: float = 1.0
) -> List[Dict]:
    """
    Format model predictions into human-readable results.

    Args:
        logits: Model output logits (batch_size, num_classes) or (num_classes,)
        label_mapping: Label mapping dictionary from load_label_mapping()
        top_k: Number of top predictions to return per sample
        temperature: Temperature for softmax (default 1.0, higher = more uniform)

    Returns:
        List of prediction dictionaries, one per batch sample.
        Each dict contains:
            - 'predictions': List of top-k predictions with prob, class_idx, label info
            - 'top_class_idx': Index of most confident class
            - 'top_probability': Probability of top class
            - 'entropy': Prediction entropy (uncertainty measure)
    """
    # Handle single sample (add batch dimension)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    batch_size = logits.shape[0]

    # Apply temperature scaling and softmax
    probs = torch.softmax(logits / temperature, dim=1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)

    # Calculate entropy for uncertainty
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

    # Format results
    results = []
    for i in range(batch_size):
        predictions = []
        for j in range(len(top_indices[i])):
            class_idx = top_indices[i][j].item()
            prob = top_probs[i][j].item()

            # Get label information based on taxonomic level
            if "idx_to_code" in label_mapping:
                # Species level
                pred_info = {
                    "probability": prob,
                    "class_idx": class_idx,
                    "species_code": label_mapping["idx_to_code"][class_idx],
                    "species_name": label_mapping["idx_to_name"][class_idx],
                }
            elif "idx_to_genus" in label_mapping:
                # Genus level
                genus = label_mapping["idx_to_genus"][class_idx]
                pred_info = {
                    "probability": prob,
                    "class_idx": class_idx,
                    "genus": genus,
                    "species_in_genus": label_mapping.get("genus_to_species", {}).get(
                        genus, []
                    ),
                }
            else:
                # Fallback
                pred_info = {
                    "probability": prob,
                    "class_idx": class_idx,
                }

            predictions.append(pred_info)

        result = {
            "predictions": predictions,
            "top_class_idx": top_indices[i][0].item(),
            "top_probability": top_probs[i][0].item(),
            "entropy": entropy[i].item(),
        }
        results.append(result)

    return results


def extract_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_class,
    num_classes: int,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Extract pure PyTorch model from Lightning checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file
        model_class: Model class to instantiate (e.g., ResNetRGB)
        num_classes: Number of output classes
        device: Device to load model on

    Returns:
        Loaded PyTorch model in eval mode

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint format is invalid
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Create model
    model = model_class(num_classes=num_classes)

    # Extract state dict (remove 'model.' prefix from Lightning wrapper)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key.replace("model.", "", 1)
                model_state_dict[new_key] = value
    else:
        raise RuntimeError("No 'state_dict' found in checkpoint")

    # Load weights
    try:
        model.load_state_dict(model_state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict: {e}")

    # Set to eval mode
    model.eval()
    model.to(device)

    return model


def calculate_confidence_threshold(
    probabilities: torch.Tensor, method: str = "entropy", threshold: float = 0.5
) -> torch.Tensor:
    """
    Calculate confidence mask based on prediction probabilities.

    Args:
        probabilities: Softmax probabilities (batch_size, num_classes)
        method: 'max_prob' or 'entropy'
        threshold: Threshold value
            - For 'max_prob': minimum probability to accept (0-1)
            - For 'entropy': maximum entropy to accept (higher = more uncertain)

    Returns:
        Boolean tensor (batch_size,) indicating confident predictions
    """
    if method == "max_prob":
        max_probs = probabilities.max(dim=1)[0]
        return max_probs >= threshold
    elif method == "entropy":
        entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=1)
        max_entropy = np.log(probabilities.shape[1])  # Maximum possible entropy
        return entropy <= (threshold * max_entropy)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'max_prob' or 'entropy'")


def get_model_info(checkpoint_path: Union[str, Path]) -> Dict:
    """
    Extract metadata from checkpoint without loading the full model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint metadata
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    info = {
        "epoch": checkpoint.get("epoch", None),
        "global_step": checkpoint.get("global_step", None),
        "hyperparameters": checkpoint.get("hyper_parameters", {}),
        "checkpoint_path": str(path),
        "checkpoint_size_mb": path.stat().st_size / (1024 * 1024),
    }

    # Extract useful hyperparameters
    hparams = info["hyperparameters"]
    if hparams:
        info["num_classes"] = hparams.get("num_classes", None)
        info["model_type"] = hparams.get("model_type", None)
        info["learning_rate"] = hparams.get("learning_rate", None)
        info["optimizer"] = hparams.get("optimizer", None)

    return info


def print_prediction_summary(results: List[Dict], detailed: bool = False) -> None:
    """
    Print formatted prediction results to console.

    Args:
        results: List of prediction dictionaries from format_predictions()
        detailed: Whether to print detailed info for all top-k predictions
    """
    for i, result in enumerate(results):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}")
        print(f"{'='*70}")

        top_pred = result["predictions"][0]
        print(f"Top Prediction:")
        if "species_code" in top_pred:
            print(f"  Species: {top_pred['species_code']} - {top_pred['species_name']}")
        elif "genus" in top_pred:
            print(f"  Genus: {top_pred['genus']}")
        print(f"  Confidence: {result['top_probability']:.2%}")
        print(f"  Entropy: {result['entropy']:.3f}")

        if detailed and len(result["predictions"]) > 1:
            print(f"\nTop {len(result['predictions'])} Predictions:")
            for j, pred in enumerate(result["predictions"], 1):
                if "species_code" in pred:
                    label = f"{pred['species_code']} - {pred['species_name'][:40]}"
                elif "genus" in pred:
                    label = pred["genus"]
                else:
                    label = f"Class {pred['class_idx']}"
                print(f"  {j}. {label:45s} {pred['probability']:6.2%}")
