"""
Image preprocessing utilities for tree species classification inference.

Handles image loading, resizing, normalization, and tensor conversion.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union
from PIL import Image


def load_image(
    image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
) -> Image.Image:
    """
    Load image from various input formats and convert to PIL Image.

    Args:
        image_input: Can be:
            - str/Path: File path to image
            - PIL.Image: Already loaded PIL image
            - numpy.ndarray: Numpy array (H, W, 3) in 0-255 or 0-1 range
            - torch.Tensor: Torch tensor (C, H, W) or (H, W, C)

    Returns:
        PIL Image in RGB mode

    Raises:
        ValueError: If input format is not supported
        FileNotFoundError: If file path doesn't exist
    """
    # Already a PIL Image
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    # File path
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img = Image.open(path)
        return img.convert("RGB")

    # Numpy array
    if isinstance(image_input, np.ndarray):
        # Ensure RGB format (H, W, 3)
        if image_input.ndim == 2:
            # Grayscale to RGB
            image_input = np.stack([image_input] * 3, axis=-1)
        elif image_input.ndim == 3:
            # Check if channels are first or last
            if (
                image_input.shape[0] == 3
                and image_input.shape[0] < image_input.shape[2]
            ):
                # (3, H, W) -> (H, W, 3)
                image_input = np.transpose(image_input, (1, 2, 0))
            elif image_input.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got {image_input.shape[2]}")
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {image_input.shape}")

        # Convert to 0-255 range if needed
        if image_input.max() <= 1.0:
            image_input = (image_input * 255).astype(np.uint8)
        else:
            image_input = image_input.astype(np.uint8)

        return Image.fromarray(image_input, mode="RGB")

    # Torch tensor
    if isinstance(image_input, torch.Tensor):
        # Convert to numpy and recurse
        array = image_input.cpu().numpy()
        return load_image(array)

    raise ValueError(
        f"Unsupported image input type: {type(image_input)}. "
        f"Expected str, Path, PIL.Image, numpy.ndarray, or torch.Tensor"
    )


def resize_image(image: Image.Image, target_size: tuple = (128, 128)) -> Image.Image:
    """
    Resize image to target size.

    Args:
        image: PIL Image
        target_size: Target (width, height) - note PIL uses (W, H) not (H, W)

    Returns:
        Resized PIL Image
    """
    return image.resize(target_size, Image.Resampling.BILINEAR)


def normalize_rgb(
    image: Union[Image.Image, np.ndarray], method: str = "0_1"
) -> np.ndarray:
    """
    Normalize RGB image to 0-1 range.

    Args:
        image: PIL Image or numpy array (H, W, 3) in 0-255 range
        method: Normalization method ('0_1' or 'imagenet')

    Returns:
        Normalized numpy array (H, W, 3) as float32
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        array = np.array(image, dtype=np.float32)
    else:
        array = image.astype(np.float32)

    if method == "0_1":
        # Simple division by 255
        array = array / 255.0
    elif method == "imagenet":
        # ImageNet normalization
        array = array / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return array


def prepare_tensor(
    image: Union[Image.Image, np.ndarray], add_batch_dim: bool = True
) -> torch.Tensor:
    """
    Convert image to PyTorch tensor in model-ready format.

    Args:
        image: PIL Image or numpy array (H, W, 3)
        add_batch_dim: Whether to add batch dimension

    Returns:
        Torch tensor in (1, 3, H, W) if add_batch_dim else (3, H, W)
    """
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        array = np.array(image, dtype=np.float32)
    else:
        array = image.astype(np.float32)

    # Convert from (H, W, 3) to (3, H, W)
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    # Add batch dimension if requested
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)

    return tensor


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    target_size: tuple = (128, 128),
    normalize: bool = True,
    norm_method: str = "0_1",
    return_tensor: bool = True,
    add_batch_dim: bool = True,
    device: str = "cpu",
) -> Union[torch.Tensor, np.ndarray]:
    """
    Complete preprocessing pipeline for inference.

    This is the main function to use for preprocessing images before model inference.

    Args:
        image_input: Image in any supported format
        target_size: Target (width, height) for resizing
        normalize: Whether to normalize to 0-1 range
        norm_method: Normalization method ('0_1' or 'imagenet')
        return_tensor: Whether to return torch.Tensor (True) or numpy.ndarray (False)
        add_batch_dim: Whether to add batch dimension (only if return_tensor=True)
        device: Device to move tensor to ('cpu', 'cuda', 'mps')

    Returns:
        Preprocessed image as torch.Tensor (1, 3, H, W) or numpy.ndarray (H, W, 3)

    Examples:
        >>> # From file path
        >>> tensor = preprocess_image('tree.jpg')
        >>>
        >>> # From PIL Image, custom size
        >>> from PIL import Image
        >>> img = Image.open('tree.jpg')
        >>> tensor = preprocess_image(img, target_size=(256, 256))
        >>>
        >>> # Return numpy array instead
        >>> array = preprocess_image('tree.jpg', return_tensor=False)
    """
    # Step 1: Load image as PIL Image
    pil_image = load_image(image_input)

    # Step 2: Resize to target size
    resized = resize_image(pil_image, target_size)

    # Step 3: Normalize (converts to numpy array)
    if normalize:
        array = normalize_rgb(resized, method=norm_method)
    else:
        array = np.array(resized, dtype=np.float32)

    # Step 4: Return as requested format
    if return_tensor:
        tensor = prepare_tensor(array, add_batch_dim=add_batch_dim)
        tensor = tensor.to(device)
        return tensor
    else:
        return array


# Convenience functions for batch processing
def preprocess_image_batch(
    image_inputs: list,
    target_size: tuple = (128, 128),
    normalize: bool = True,
    norm_method: str = "0_1",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Preprocess a batch of images.

    Args:
        image_inputs: List of images in any supported format
        target_size: Target size for all images
        normalize: Whether to normalize
        norm_method: Normalization method
        device: Device for tensors

    Returns:
        Batched tensor (N, 3, H, W)
    """
    tensors = []
    for img_input in image_inputs:
        tensor = preprocess_image(
            img_input,
            target_size=target_size,
            normalize=normalize,
            norm_method=norm_method,
            return_tensor=True,
            add_batch_dim=False,  # We'll stack manually
            device=device,
        )
        tensors.append(tensor)

    # Stack into batch
    return torch.stack(tensors, dim=0)


def validate_image_input(image_input) -> bool:
    """
    Check if image input is valid without actually loading it.

    Args:
        image_input: Image in any format

    Returns:
        True if valid, False otherwise
    """
    try:
        load_image(image_input)
        return True
    except Exception:
        return False
