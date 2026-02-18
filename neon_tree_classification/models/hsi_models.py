"""
Hyperspectral imaging (HSI) neural network architectures for tree species classification.

All models expect HSI input as [batch_size, bands, height, width] tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class SimpleHSINet(nn.Module):
    """
    Simple neural network for hyperspectral tree crown classification.

    Treats HSI data as multi-channel images and uses 2D convolutions.
    """

    def __init__(
        self, num_bands: int = 369, num_classes: int = 10, input_size: int = 224
    ):
        """
        Initialize SimpleHSINet.

        Args:
            num_bands: Number of hyperspectral bands
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.input_size = input_size

        # Spectral dimension reduction first
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=1),  # 1x1 conv for spectral mixing
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Spatial feature extraction
        self.spatial_features = nn.Sequential(
            # First block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Reduce spectral dimensionality
        x = self.spectral_conv(x)

        # Extract spatial features
        features = self.spatial_features(x)

        # Classify
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Feature vector [batch_size, 256]
        """
        x = self.spectral_conv(x)
        features = self.spatial_features(x)
        return features.flatten(1)


class SpectralCNN(nn.Module):
    """
    1D CNN operating along the spectral dimension for pixel-wise classification.

    Processes each pixel's spectrum independently, then aggregates spatially.
    """

    def __init__(
        self, num_bands: int = 369, num_classes: int = 10, input_size: int = 144
    ):
        """
        Initialize SpectralCNN.

        Args:
            num_bands: Number of hyperspectral bands
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.input_size = input_size

        # 1D CNN for spectral features (applied per pixel)
        self.spectral_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Spatial aggregation
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        batch_size, bands, height, width = x.shape

        # Reshape for 1D convolution along spectral dimension
        # [batch_size, bands, height, width] -> [batch_size * height * width, 1, bands]
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, bands]
        x_reshaped = x_reshaped.view(-1, 1, bands)  # [B*H*W, 1, bands]

        # Apply 1D CNN along spectral dimension
        spectral_features = self.spectral_cnn(x_reshaped)  # [B*H*W, 128, 1]
        spectral_features = spectral_features.squeeze(-1)  # [B*H*W, 128]

        # Reshape back to spatial format
        spectral_features = spectral_features.view(batch_size, height, width, 128)
        spectral_features = spectral_features.permute(0, 3, 1, 2)  # [B, 128, H, W]

        # Spatial aggregation
        spatial_features = self.spatial_conv(spectral_features)  # [B, 256, 1, 1]

        # Classification
        logits = self.classifier(spatial_features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Feature vector [batch_size, 256]
        """
        batch_size, bands, height, width = x.shape

        # Process spectral features
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, 1, bands)
        spectral_features = self.spectral_cnn(x_reshaped).squeeze(-1)
        spectral_features = spectral_features.view(batch_size, height, width, 128)
        spectral_features = spectral_features.permute(0, 3, 1, 2)

        # Spatial aggregation
        features = self.spatial_conv(spectral_features)
        return features.flatten(1)


class HyperNet(nn.Module):
    """
    Advanced hyperspectral network with attention mechanisms.

    Uses both spectral and spatial attention for better feature learning.
    """

    def __init__(
        self, num_bands: int = 369, num_classes: int = 10, input_size: int = 224
    ):
        """
        Initialize HyperNet.

        Args:
            num_bands: Number of hyperspectral bands
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_bands = num_bands
        self.num_classes = num_classes

        # Spectral attention module
        self.spectral_attention = SpectralAttention(num_bands)

        # Feature extraction after attention
        self.features = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Apply spectral attention
        x = self.spectral_attention(x)

        # Extract features
        features = self.features(x)

        # Classify
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Feature vector [batch_size, 512]
        """
        x = self.spectral_attention(x)
        features = self.features(x)
        return features.flatten(1)


class SpectralAttention(nn.Module):
    """Spectral attention mechanism for hyperspectral data."""

    def __init__(self, num_bands: int):
        super().__init__()
        self.num_bands = num_bands

        # Global average pooling across spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(num_bands, num_bands // 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands // 4, num_bands),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral attention.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Attention-weighted HSI tensor [batch_size, bands, height, width]
        """
        batch_size, bands, height, width = x.shape

        # Global spatial pooling: [B, bands, H, W] -> [B, bands, 1, 1]
        pooled = self.gap(x)

        # Flatten for attention: [B, bands, 1, 1] -> [B, bands]
        pooled = pooled.view(batch_size, bands)

        # Compute attention weights: [B, bands] -> [B, bands]
        attention_weights = self.attention(pooled)

        # Reshape for broadcasting: [B, bands] -> [B, bands, 1, 1]
        attention_weights = attention_weights.view(batch_size, bands, 1, 1)

        # Apply attention weights
        return x * attention_weights


# Factory function for easy model creation
def create_hsi_model(
    model_type: str = "simple", num_bands: int = 369, num_classes: int = 10, **kwargs
) -> nn.Module:
    """
    Factory function to create HSI models.

    Args:
        model_type: Type of model ("simple", "spectral_cnn", "hypernet")
        num_bands: Number of hyperspectral bands
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        HSI classification model
    """
    if model_type == "simple":
        return SimpleHSINet(num_bands=num_bands, num_classes=num_classes, **kwargs)
    elif model_type == "spectral_cnn":
        return SpectralCNN(num_bands=num_bands, num_classes=num_classes, **kwargs)
    elif model_type == "hypernet":
        return HyperNet(num_bands=num_bands, num_classes=num_classes, **kwargs)
    elif model_type == "hang2020":
        return Hang2020(num_bands=num_bands, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown HSI model type: {model_type}")


# =============================================================================
# Hang et al. 2020 - Dual-Pathway Attention Architecture
# Paper: "Hyperspectral Image Classification with Attention Aided CNNs"
# https://arxiv.org/abs/2005.11977
#
# Implementation adapted from weecology/DeepTreeAttention for NEON tree classification
# =============================================================================


def global_spectral_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Global average pooling across spatial dimensions only.
    Maintains spectral/channel dimension.

    Args:
        x: [B, C, H, W] tensor

    Returns:
        [B, C, 1] tensor after spatial pooling
    """
    # Pool over H and W, keep channel dimension
    pooled = torch.mean(x, dim=[2, 3])  # [B, C]
    return pooled.unsqueeze(-1)  # [B, C, 1] for convolutions


class ConvModule(nn.Module):
    """
    Basic convolutional block with optional max pooling.
    Conv2d -> BatchNorm -> ReLU -> Optional MaxPool
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int = 3,
        maxpool_kernel: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, filters, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(filters)
        self.maxpool = (
            nn.MaxPool2d(maxpool_kernel) if maxpool_kernel is not None else None
        )

    def forward(self, x: torch.Tensor, pool: bool = False) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if pool and self.maxpool is not None:
            x = self.maxpool(x)
        return x


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Learns cross-band spatial features with convolutions and pooling attention.
    First reduces channels to 1, then applies 2D attention convolutions,
    multiplies attention map with input features.
    """

    def __init__(self, filters: int):
        super().__init__()

        # Channel pooling: reduce all filters to single spatial attention map
        self.channel_pool = nn.Conv2d(
            in_channels=filters, out_channels=1, kernel_size=1
        )

        # Adaptive kernel size based on feature map size
        if filters == 32:
            kernel_size = 7
        elif filters == 64:
            kernel_size = 5
        elif filters == 128:
            kernel_size = 3
        else:
            raise ValueError(f"Unknown filter size {filters} for spatial attention")

        # Spatial attention convolutions
        self.attention_conv1 = nn.Conv2d(1, 1, kernel_size=kernel_size, padding="same")
        self.attention_conv2 = nn.Conv2d(1, 1, kernel_size=kernel_size, padding="same")

        # Use adaptive pooling instead of fixed pooling
        self.class_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            attention_features: [B, C, H, W] attention-weighted features
            pooled_features: [B, C'] flattened features for classification
        """
        # Global spatial pooling via channel reduction
        pooled_features = self.channel_pool(x)  # [B, 1, H, W]
        pooled_features = F.relu(pooled_features)

        # Compute spatial attention map
        attention = self.attention_conv1(pooled_features)
        attention = F.relu(attention)
        attention = self.attention_conv2(attention)
        attention = torch.sigmoid(attention)  # [B, 1, H, W]

        # Apply attention to input features
        attention_features = torch.mul(x, attention)  # [B, C, H, W]

        # Classification head: pool and flatten
        pooled_attention = self.class_pool(attention_features)  # [B, C, H', W']
        pooled_attention_flat = torch.flatten(pooled_attention, start_dim=1)

        return attention_features, pooled_attention_flat


class SpectralAttention(nn.Module):
    """
    Spectral attention module.

    Learns cross-band spectral features. Applies global spatial pooling first,
    then 1D convolutions along spectral dimension to compute band attention weights.
    """

    def __init__(self, filters: int):
        super().__init__()

        # Adaptive kernel size based on feature depth
        if filters == 32:
            kernel_size = 3
        elif filters == 64:
            kernel_size = 5
        elif filters == 128:
            kernel_size = 7
        else:
            raise ValueError(f"Unknown filter size {filters} for spectral attention")

        # 1D spectral attention convolutions
        self.attention_conv1 = nn.Conv1d(
            filters, filters, kernel_size=kernel_size, padding="same"
        )
        self.attention_conv2 = nn.Conv1d(
            filters, filters, kernel_size=kernel_size, padding="same"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            attention_features: [B, C, H, W] spectral-attention-weighted features
            pooled_features: [B, C] flattened features for classification
        """
        # Global spatial pooling: [B, C, H, W] -> [B, C, 1]
        pooled_features = global_spectral_pool(x)

        # Compute spectral attention weights via 1D convolutions
        attention = self.attention_conv1(pooled_features)  # [B, C, 1]
        attention = F.relu(attention)
        attention = self.attention_conv2(attention)
        attention = torch.sigmoid(attention)  # [B, C, 1]

        # Broadcast attention to spatial dimensions: [B, C, 1] -> [B, C, 1, 1]
        attention = attention.unsqueeze(-1)

        # Apply spectral attention
        attention_features = torch.mul(x, attention)  # [B, C, H, W]

        # Classification head: global pool and flatten
        pooled_attention = global_spectral_pool(attention_features)  # [B, C, 1]
        pooled_attention_flat = torch.flatten(pooled_attention, start_dim=1)  # [B, C]

        return attention_features, pooled_attention_flat


class Classifier(nn.Module):
    """
    Simple linear classification head.
    Separates classifier from feature extractor for easier pretraining.
    """

    def __init__(self, in_features: int, classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SpatialNetwork(nn.Module):
    """
    Spatial pathway: learns spatial features with attention at multiple scales.

    Architecture:
        Conv(32) -> SpatialAttn -> Classifier(32)
        Conv(64) -> SpatialAttn -> Classifier(64)
        Conv(128) -> SpatialAttn -> Classifier(128)
    """

    def __init__(self, num_bands: int, num_classes: int):
        super().__init__()

        # Stage 1: 32 filters
        self.conv1 = ConvModule(num_bands, 32)
        self.attention_1 = SpatialAttention(32)
        self.classifier1 = Classifier(32, num_classes)

        # Stage 2: 64 filters
        self.conv2 = ConvModule(32, 64, maxpool_kernel=(2, 2))
        self.attention_2 = SpatialAttention(64)
        self.classifier2 = Classifier(64, num_classes)

        # Stage 3: 128 filters
        self.conv3 = ConvModule(64, 128, maxpool_kernel=(2, 2))
        self.attention_3 = SpatialAttention(128)
        self.classifier3 = Classifier(128, num_classes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through spatial pathway.

        Args:
            x: [B, C, H, W] input HSI

        Returns:
            List of 3 class score tensors [B, num_classes] from each stage
        """
        # Stage 1
        x = self.conv1(x)
        x, attention = self.attention_1(x)
        scores1 = self.classifier1(attention)

        # Stage 2
        x = self.conv2(x, pool=True)
        x, attention = self.attention_2(x)
        scores2 = self.classifier2(attention)

        # Stage 3
        x = self.conv3(x, pool=True)
        x, attention = self.attention_3(x)
        scores3 = self.classifier3(attention)

        return [scores1, scores2, scores3]


class SpectralNetwork(nn.Module):
    """
    Spectral pathway: learns spectral features with attention at multiple scales.

    Architecture:
        Conv(32) -> SpectralAttn -> Classifier(32)
        Conv(64) -> SpectralAttn -> Classifier(64)
        Conv(128) -> SpectralAttn -> Classifier(128)
    """

    def __init__(self, num_bands: int, num_classes: int):
        super().__init__()

        # Stage 1: 32 filters
        self.conv1 = ConvModule(num_bands, 32)
        self.attention_1 = SpectralAttention(32)
        self.classifier1 = Classifier(32, num_classes)

        # Stage 2: 64 filters
        self.conv2 = ConvModule(32, 64, maxpool_kernel=(2, 2))
        self.attention_2 = SpectralAttention(64)
        self.classifier2 = Classifier(64, num_classes)

        # Stage 3: 128 filters
        self.conv3 = ConvModule(64, 128, maxpool_kernel=(2, 2))
        self.attention_3 = SpectralAttention(128)
        self.classifier3 = Classifier(128, num_classes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through spectral pathway.

        Args:
            x: [B, C, H, W] input HSI

        Returns:
            List of 3 class score tensors [B, num_classes] from each stage
        """
        # Stage 1
        x = self.conv1(x)
        x, attention = self.attention_1(x)
        scores1 = self.classifier1(attention)

        # Stage 2
        x = self.conv2(x, pool=True)
        x, attention = self.attention_2(x)
        scores2 = self.classifier2(attention)

        # Stage 3
        x = self.conv3(x, pool=True)
        x, attention = self.attention_3(x)
        scores3 = self.classifier3(attention)

        return [scores1, scores2, scores3]


class Hang2020(nn.Module):
    """
    Dual-pathway attention architecture from Hang et al. 2020.
    Paper: "Hyperspectral Image Classification with Attention Aided CNNs"

    Features:
    - Separate spectral and spatial processing pathways
    - Multi-scale attention at 3 levels (32, 64, 128 filters)
    - Learnable weighted fusion of both pathways
    - Multi-output supervision during training

    This architecture is specifically designed for hyperspectral data and has shown
    strong performance on NEON tree species classification (DeepTreeAttention project).

    Args:
        num_bands: Number of HSI bands (default 369 for NEON)
        num_classes: Number of tree species classes
        input_size: Expected input spatial size (not used, kept for API compatibility)
    """

    def __init__(
        self,
        num_bands: int = 369,
        num_classes: int = 167,
        input_size: int = 128,
        **kwargs,
    ):
        super().__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes

        # Dual pathways
        self.spectral_network = SpectralNetwork(num_bands, num_classes)
        self.spatial_network = SpatialNetwork(num_bands, num_classes)

        # Learnable fusion weight (initialized to 0.5)
        self.alpha = nn.Parameter(
            torch.tensor(0.5, dtype=torch.float32), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual pathways with weighted fusion.

        Args:
            x: [B, num_bands, H, W] input HSI tensor

        Returns:
            [B, num_classes] final class scores (from stage 3 fusion)

        Note: During training, you can access intermediate scores via forward_with_aux()
        """
        # Get scores from both pathways (3 stages each)
        spectral_scores = self.spectral_network(x)
        spatial_scores = self.spatial_network(x)

        # Use final stage (index -1) for inference
        spectral_final = spectral_scores[-1]  # [B, num_classes]
        spatial_final = spatial_scores[-1]  # [B, num_classes]

        # Learnable weighted fusion (alpha in [0, 1] via sigmoid)
        weight = torch.sigmoid(self.alpha)
        joint_score = spectral_final * weight + spatial_final * (1 - weight)

        return joint_score

    def forward_with_aux(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning both final scores and auxiliary scores for multi-output training.

        Args:
            x: [B, num_bands, H, W] input HSI

        Returns:
            final_scores: [B, num_classes] fused predictions from stage 3
            aux_scores: List of 6 tensors [B, num_classes] - 3 spectral + 3 spatial
        """
        spectral_scores = self.spectral_network(x)
        spatial_scores = self.spatial_network(x)

        # Final fusion
        weight = torch.sigmoid(self.alpha)
        final_scores = spectral_scores[-1] * weight + spatial_scores[-1] * (1 - weight)

        # Return final + all auxiliary scores for deep supervision
        aux_scores = spectral_scores + spatial_scores

        return final_scores, aux_scores
