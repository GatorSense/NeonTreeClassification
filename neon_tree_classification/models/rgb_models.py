"""
RGB-based neural network architectures for tree species classification.

All models expect RGB input as [batch_size, 3, height, width] tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torchvision.models as models


class SimpleRGBNet(nn.Module):
    """
    Simple CNN for RGB tree crown classification.

    Basic architecture with conv layers followed by classification head.
    """

    def __init__(self, num_classes: int = 10, input_size: int = 224):
        """
        Initialize SimpleRGBNet.

        Args:
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth block
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
            x: RGB tensor [batch_size, 3, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: RGB tensor [batch_size, 3, height, width]

        Returns:
            Feature vector [batch_size, 256]
        """
        features = self.features(x)
        return features.flatten(1)


class ResNetRGB(nn.Module):
    """
    ResNet-inspired architecture for RGB tree crown classification.

    Uses residual connections for better gradient flow.
    """

    def __init__(self, num_classes: int = 10, num_blocks: list = [2, 2, 2, 2]):
        """
        Initialize ResNetRGB.

        Args:
            num_classes: Number of tree species classes
            num_blocks: Number of residual blocks per stage
        """
        super().__init__()
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int
    ):
        """Create a residual layer."""
        layers = []

        # First block (may have stride > 1)
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: RGB tensor [batch_size, 3, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: RGB tensor [batch_size, 3, height, width]

        Returns:
            Feature vector [batch_size, 512]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResidualBlock(nn.Module):
    """Basic residual block for ResNetRGB."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ViTRGB(nn.Module):
    """
    Vision Transformer (ViT) for RGB tree crown classification.
    
    Uses pretrained ViT models from torchvision with custom classification head.
    Supports ViT-B/16, ViT-B/32, ViT-L/16, and ViT-L/32 architectures.
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        model_variant: str = "vit_b_16",
        pretrained: bool = True
    ):
        """
        Initialize ViT model.
        
        Args:
            num_classes: Number of tree species classes
            model_variant: ViT variant - 'vit_b_16' (base/16), 'vit_b_32' (base/32),
                          'vit_l_16' (large/16), 'vit_l_32' (large/32)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_variant = model_variant
        
        # Load pretrained ViT model
        if model_variant == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_b_16(weights=weights)
            hidden_dim = 768
        elif model_variant == "vit_b_32":
            weights = models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_b_32(weights=weights)
            hidden_dim = 768
        elif model_variant == "vit_l_16":
            weights = models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_l_16(weights=weights)
            hidden_dim = 1024
        elif model_variant == "vit_l_32":
            weights = models.ViT_L_32_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_l_32(weights=weights)
            hidden_dim = 1024
        else:
            raise ValueError(
                f"Unknown ViT variant: {model_variant}. "
                f"Choose from: vit_b_16, vit_b_32, vit_l_16, vit_l_32"
            )
        
        # Replace classification head
        self.vit.heads = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: RGB tensor [batch_size, 3, height, width]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        return self.vit(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        
        Args:
            x: RGB tensor [batch_size, 3, height, width]
        
        Returns:
            Feature vector [batch_size, hidden_dim]
        """
        # Extract features (without classification head)
        x = self.vit._process_input(x)
        n = x.shape[0]
        
        # Expand class token to batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Pass through transformer encoder
        x = self.vit.encoder(x)
        
        # Use class token representation
        x = x[:, 0]
        return x


# Factory function for easy model creation
def create_rgb_model(
    model_type: str = "simple", num_classes: int = 10, **kwargs
) -> nn.Module:
    """
    Factory function to create RGB models.

    Args:
        model_type: Type of model ("simple", "resnet", "vit")
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments (e.g., model_variant for ViT)

    Returns:
        RGB classification model
    """
    if model_type == "simple":
        return SimpleRGBNet(num_classes=num_classes, **kwargs)
    elif model_type == "resnet":
        return ResNetRGB(num_classes=num_classes, **kwargs)
    elif model_type == "vit":
        return ViTRGB(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown RGB model type: {model_type}. "
            f"Choose from: simple, resnet, vit"
        )
