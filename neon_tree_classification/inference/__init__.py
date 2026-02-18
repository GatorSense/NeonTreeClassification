"""
NEON Tree Classification Inference Module

Provides inference capabilities for pretrained tree species classification models.

Usage:
    from neon_tree_classification.inference import TreeClassifier

    # Load from checkpoint
    classifier = TreeClassifier.from_checkpoint(
        checkpoint_path='path/to/model.ckpt',
        taxonomic_level='species'
    )

    # Predict single image
    result = classifier.predict('path/to/image.jpg', top_k=5)

    # Batch prediction
    results = classifier.predict_batch(['img1.jpg', 'img2.jpg'])
"""

from .predictor import TreeClassifier
from .preprocessing import preprocess_image, prepare_tensor
from .utils import load_label_mapping, format_predictions

__all__ = [
    "TreeClassifier",
    "preprocess_image",
    "prepare_tensor",
    "load_label_mapping",
    "format_predictions",
]

__version__ = "1.0.0"
