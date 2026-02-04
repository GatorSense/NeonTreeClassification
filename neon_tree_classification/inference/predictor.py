"""
Main inference predictor for NEON tree species classification.

Provides high-level API for loading models and making predictions.
"""

import torch
import warnings
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from neon_tree_classification.models.rgb_models import create_rgb_model
from .preprocessing import preprocess_image, preprocess_image_batch
from .utils import (
    load_label_mapping,
    format_predictions,
    extract_model_from_checkpoint,
    print_prediction_summary,
)
from .model_registry import (
    get_model_info,
    get_label_mapping_path,
    list_available_models,
)


class TreeClassifier:
    """
    High-level interface for tree species classification inference.
    
    Supports both species-level (167 classes) and genus-level (60 classes) classification
    using pretrained RGB ResNet models.
    
    Examples:
        >>> # Load from checkpoint
        >>> classifier = TreeClassifier.from_checkpoint(
        ...     checkpoint_path='path/to/best.ckpt',
        ...     taxonomic_level='species'
        ... )
        >>> 
        >>> # Single image prediction
        >>> result = classifier.predict('tree_image.jpg', top_k=5)
        >>> print(f"Top prediction: {result['predictions'][0]['species_name']}")
        >>> 
        >>> # Batch prediction
        >>> results = classifier.predict_batch(['img1.jpg', 'img2.jpg'])
        >>> 
        >>> # Get class probabilities
        >>> probs = classifier.get_class_probabilities('tree_image.jpg')
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        label_mapping: Dict,
        taxonomic_level: str,
        device: str = None,
        input_size: Tuple[int, int] = (128, 128),
    ):
        """
        Initialize tree classifier.
        
        Args:
            model: PyTorch model for inference
            label_mapping: Label mapping dictionary
            taxonomic_level: 'species' or 'genus'
            device: Device for inference ('cpu', 'cuda', 'mps'). Auto-detected if None.
            input_size: Input image size (width, height)
        """
        self.model = model
        self.label_mapping = label_mapping
        self.taxonomic_level = taxonomic_level
        self.input_size = input_size
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        # Get number of classes from label mapping
        if 'idx_to_code' in label_mapping:
            self.num_classes = len(label_mapping['idx_to_code'])
        elif 'idx_to_genus' in label_mapping:
            self.num_classes = len(label_mapping['idx_to_genus'])
        else:
            raise ValueError("Invalid label mapping format")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        taxonomic_level: str = 'species',
        label_mapping_path: Optional[Union[str, Path]] = None,
        model_type: str = 'resnet',
        device: str = None,
    ) -> 'TreeClassifier':
        """
        Load classifier from Lightning checkpoint file.
        
        Args:
            checkpoint_path: Path to .ckpt file
            taxonomic_level: 'species' (167 classes) or 'genus' (60 classes)
            label_mapping_path: Custom path to label JSON (optional, auto-detected otherwise)
            model_type: Model architecture ('resnet', 'simple')
            device: Device for inference
        
        Returns:
            Initialized TreeClassifier
        
        Examples:
            >>> # Species-level classification
            >>> classifier = TreeClassifier.from_checkpoint(
            ...     'checkpoints/resnet_species_best.ckpt',
            ...     taxonomic_level='species'
            ... )
            >>> 
            >>> # Genus-level classification
            >>> classifier = TreeClassifier.from_checkpoint(
            ...     'checkpoints/resnet_genus_best.ckpt',
            ...     taxonomic_level='genus'
            ... )
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load label mapping
        if label_mapping_path is None:
            label_path = get_label_mapping_path(taxonomic_level)
        else:
            label_path = Path(label_mapping_path)
        
        print(f"Loading label mapping from: {label_path}")
        label_mapping = load_label_mapping(label_path, taxonomic_level)
        num_classes = label_mapping['metadata']['num_classes']
        
        print(f"Creating {model_type} model with {num_classes} classes")
        model_class = create_rgb_model
        
        # Create model architecture
        model = model_class(model_type=model_type, num_classes=num_classes)
        
        # Load weights from checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model state dict (remove 'model.' prefix)
        state_dict = checkpoint['state_dict']
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key.replace('model.', '', 1)
                model_state_dict[new_key] = value
        
        model.load_state_dict(model_state_dict)
        
        print(f"✅ Model loaded successfully")
        print(f"   Architecture: {model_type}")
        print(f"   Classes: {num_classes} ({taxonomic_level} level)")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return cls(
            model=model,
            label_mapping=label_mapping,
            taxonomic_level=taxonomic_level,
            device=device,
            input_size=(128, 128),
        )
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: Optional[Path] = None,
        device: str = None,
    ) -> 'TreeClassifier':
        """
        Load pretrained model from registry (placeholder for HuggingFace integration).
        
        Args:
            model_name: Name of pretrained model (e.g., 'resnet_species')
            cache_dir: Directory for cached models
            device: Device for inference
        
        Returns:
            Initialized TreeClassifier
        
        Raises:
            NotImplementedError: Feature pending HuggingFace upload
        """
        available = ', '.join(list_available_models())
        raise NotImplementedError(
            f"from_pretrained() will be available after HuggingFace upload. "
            f"Available models: {available}. "
            f"For now, use from_checkpoint() with a local .ckpt file."
        )
    
    def predict(
        self,
        image_input: Union[str, Path],
        top_k: int = 5,
        return_dict: bool = True,
        temperature: float = 1.0,
    ) -> Union[Dict, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict tree species/genus for a single image.
        
        Args:
            image_input: Image path, PIL Image, or numpy array
            top_k: Number of top predictions to return
            return_dict: Return formatted dict (True) or raw tensors (False)
            temperature: Temperature for softmax (higher = more uniform probabilities)
        
        Returns:
            If return_dict=True: Dictionary with formatted predictions
            If return_dict=False: Tuple of (probabilities, class_indices)
        
        Examples:
            >>> result = classifier.predict('tree.jpg', top_k=3)
            >>> print(f"Top prediction: {result['predictions'][0]['species_name']}")
            >>> print(f"Confidence: {result['top_probability']:.2%}")
            >>> 
            >>> # Get raw tensors
            >>> probs, indices = classifier.predict('tree.jpg', return_dict=False)
        """
        # Preprocess image
        tensor = preprocess_image(
            image_input,
            target_size=self.input_size,
            normalize=True,
            norm_method='0_1',
            return_tensor=True,
            add_batch_dim=True,
            device=self.device
        )
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor)
        
        # Return format
        if return_dict:
            results = format_predictions(
                logits,
                self.label_mapping,
                top_k=top_k,
                temperature=temperature
            )
            return results[0]  # Return single result (not list)
        else:
            probs = torch.softmax(logits / temperature, dim=1)
            top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)
            return top_probs[0], top_indices[0]
    
    def predict_batch(
        self,
        image_inputs: List,
        top_k: int = 5,
        batch_size: int = 32,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """
        Predict tree species/genus for multiple images.
        
        Args:
            image_inputs: List of image paths, PIL Images, or numpy arrays
            top_k: Number of top predictions per image
            batch_size: Batch size for processing
            temperature: Temperature for softmax
        
        Returns:
            List of prediction dictionaries, one per input image
        
        Examples:
            >>> images = ['tree1.jpg', 'tree2.jpg', 'tree3.jpg']
            >>> results = classifier.predict_batch(images)
            >>> for i, result in enumerate(results):
            ...     print(f"Image {i+1}: {result['predictions'][0]['species_name']}")
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i + batch_size]
            
            # Preprocess batch
            tensor = preprocess_image_batch(
                batch,
                target_size=self.input_size,
                normalize=True,
                norm_method='0_1',
                device=self.device
            )
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(tensor)
            
            # Format predictions
            batch_results = format_predictions(
                logits,
                self.label_mapping,
                top_k=top_k,
                temperature=temperature
            )
            all_results.extend(batch_results)
        
        return all_results
    
    def get_class_probabilities(
        self,
        image_input: Union[str, Path],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Get probability distribution over all classes for an image.
        
        Args:
            image_input: Image path, PIL Image, or numpy array
            temperature: Temperature for softmax
        
        Returns:
            Tensor of probabilities (num_classes,)
        
        Examples:
            >>> probs = classifier.get_class_probabilities('tree.jpg')
            >>> print(f"Shape: {probs.shape}")  # (167,) for species level
            >>> print(f"Sum: {probs.sum()}")    # Should be 1.0
        """
        # Preprocess
        tensor = preprocess_image(
            image_input,
            target_size=self.input_size,
            normalize=True,
            device=self.device
        )
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits / temperature, dim=1)
        
        return probs[0]  # Remove batch dimension
    
    def print_prediction(
        self,
        image_input: Union[str, Path],
        top_k: int = 5,
    ) -> None:
        """
        Print formatted prediction for an image to console.
        
        Args:
            image_input: Image path, PIL Image, or numpy array
            top_k: Number of top predictions to display
        """
        result = self.predict(image_input, top_k=top_k)
        print_prediction_summary([result], detailed=True)
    
    def __repr__(self) -> str:
        return (
            f"TreeClassifier("
            f"taxonomic_level='{self.taxonomic_level}', "
            f"num_classes={self.num_classes}, "
            f"device='{self.device}')"
        )
