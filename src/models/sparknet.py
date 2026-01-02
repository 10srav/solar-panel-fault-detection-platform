"""SparkNet - Solar Panel Fault Detection CNN with Fire Modules.

Based on the paper: "SparkNet—A Solar Panel Fault Detection Deep Learning Model"
by Rohith G et al., IEEE Access 2025.

Architecture features:
- 4 independent branches with hierarchical CNN
- Fire Modules (Squeeze + Expand layers) for efficient feature extraction
- Multiple dropout layers for regularization
- Global Average Pooling
- Softmax classification
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FireModule(nn.Module):
    """Fire Module from SqueezeNet architecture.

    Consists of:
    - Squeeze layer: 1x1 convolution to reduce channels
    - Expand layer: combination of 1x1 and 3x3 convolutions

    Args:
        in_channels: Number of input channels.
        squeeze_planes: Number of channels in squeeze layer.
        expand1x1_planes: Number of channels in 1x1 expand.
        expand3x3_planes: Number of channels in 3x3 expand.
        extra_expand3x3: Whether to add an extra 3x3 expand branch.
    """

    def __init__(
        self,
        in_channels: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        extra_expand3x3: bool = True,
    ) -> None:
        super().__init__()

        self.squeeze = nn.Conv2d(in_channels, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

        self.extra_expand3x3 = extra_expand3x3
        if extra_expand3x3:
            self.expand3x3_extra = nn.Conv2d(
                squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
            )
            self.expand3x3_extra_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Fire Module."""
        x = self.squeeze_activation(self.squeeze(x))

        expand1x1 = self.expand1x1_activation(self.expand1x1(x))
        expand3x3 = self.expand3x3_activation(self.expand3x3(x))

        if self.extra_expand3x3:
            expand3x3_extra = self.expand3x3_extra_activation(self.expand3x3_extra(x))
            return torch.cat([expand1x1, expand3x3, expand3x3_extra], dim=1)

        return torch.cat([expand1x1, expand3x3], dim=1)

    def get_output_channels(self) -> int:
        """Get the number of output channels."""
        expand_channels = self.expand1x1.out_channels + self.expand3x3.out_channels
        if self.extra_expand3x3:
            expand_channels += self.expand3x3_extra.out_channels
        return expand_channels


class ConvBranch(nn.Module):
    """Single branch of the SparkNet hierarchical architecture.

    Each branch consists of:
    - Initial convolution layer
    - ReLU or LeakyReLU activation
    - MaxPooling

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        use_leaky_relu: Whether to use LeakyReLU instead of ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_leaky_relu: bool = False,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

        if use_leaky_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through branch."""
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class SparkNet(nn.Module):
    """SparkNet - Multi-branch CNN with Fire Modules for solar panel fault detection.

    Architecture:
    - Input layer (227x227x3)
    - 4 parallel branches with Conv + Pool layers
    - Concatenation of branch outputs
    - Series of Fire Modules (Fire2-Fire9)
    - 3 parallel dropout branches
    - Element-wise addition
    - Global Average Pooling
    - Softmax classification

    Args:
        num_classes: Number of output classes (default: 6 for solar panel faults).
        input_channels: Number of input image channels (default: 3 for RGB).
        dropout_rate: Dropout probability (default: 0.5).
    """

    def __init__(
        self,
        num_classes: int = 6,
        input_channels: int = 3,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate

        # ============ 4 Parallel Branches ============
        # Branch 1: Conv(64) -> ReLU -> MaxPool
        self.branch1 = ConvBranch(input_channels, 64, kernel_size=3, use_leaky_relu=False)

        # Branch 2: Conv(64) -> ReLU -> MaxPool
        self.branch2 = ConvBranch(input_channels, 64, kernel_size=3, use_leaky_relu=False)

        # Branch 3: Conv(64) -> ReLU -> MaxPool
        self.branch3 = ConvBranch(input_channels, 64, kernel_size=3, use_leaky_relu=False)

        # Branch 4: Conv(64) -> LeakyReLU -> MaxPool
        self.branch4 = ConvBranch(input_channels, 64, kernel_size=3, use_leaky_relu=True)

        # Branch 5 (additional): Conv(64) -> ReLU -> MaxPool
        self.branch5 = ConvBranch(input_channels, 64, kernel_size=3, use_leaky_relu=False)

        # ============ Fire Modules ============
        # After concatenation: 64*5 = 320 channels
        concat_channels = 320

        # Fire2: squeeze=16, expand=64 (output = 64 + 64 + 64 = 192)
        self.fire2 = FireModule(concat_channels, 16, 64, 64, extra_expand3x3=True)
        fire2_out = 192

        # Fire3: squeeze=16, expand=64
        self.fire3 = FireModule(fire2_out, 16, 64, 64, extra_expand3x3=True)
        fire3_out = 192
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fire4: squeeze=32, expand=128
        self.fire4 = FireModule(fire3_out, 32, 128, 128, extra_expand3x3=True)
        fire4_out = 384

        # Fire5: squeeze=32, expand=128
        self.fire5 = FireModule(fire4_out, 32, 128, 128, extra_expand3x3=True)
        fire5_out = 384
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fire6: squeeze=48, expand=192
        self.fire6 = FireModule(fire5_out, 48, 192, 192, extra_expand3x3=True)
        fire6_out = 576

        # Fire7: squeeze=48, expand=192
        self.fire7 = FireModule(fire6_out, 48, 192, 192, extra_expand3x3=True)
        fire7_out = 576

        # Fire8: squeeze=64, expand=256
        self.fire8 = FireModule(fire7_out, 64, 256, 256, extra_expand3x3=True)
        fire8_out = 768

        # Fire9: squeeze=64, expand=256
        self.fire9 = FireModule(fire8_out, 64, 256, 256, extra_expand3x3=True)
        fire9_out = 768

        # ============ Final Branches with Dropout ============
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        # Final convolution layers for each branch
        self.final_conv1 = nn.Conv2d(fire9_out, num_classes, kernel_size=1)
        self.final_conv2 = nn.Conv2d(fire9_out, num_classes, kernel_size=1)
        self.final_conv3 = nn.Conv2d(fire9_out, num_classes, kernel_size=1)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SparkNet.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Class probabilities of shape (batch, num_classes).
        """
        # Parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.branch5(x)

        # Concatenate branch outputs
        x = torch.cat([b1, b2, b3, b4, b5], dim=1)

        # Fire modules
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool3(x)

        x = self.fire4(x)
        x = self.fire5(x)
        x = self.pool5(x)

        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.fire9(x)

        # Three parallel dropout branches
        d1 = self.dropout1(x)
        d2 = self.dropout2(x)
        d3 = self.dropout3(x)

        # Final convolutions
        f1 = self.final_conv1(d1)
        f2 = self.final_conv2(d2)
        f3 = self.final_conv3(d3)

        # Element-wise addition
        x = f1 + f2 + f3

        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer.

        Used for ablation studies with classical ML classifiers.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Feature tensor of shape (batch, feature_dim).
        """
        # Parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.branch5(x)

        # Concatenate branch outputs
        x = torch.cat([b1, b2, b3, b4, b5], dim=1)

        # Fire modules
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool3(x)

        x = self.fire4(x)
        x = self.fire5(x)
        x = self.pool5(x)

        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.fire9(x)

        # Global Average Pooling to get features
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        return x

    def get_cam_target_layer(self) -> nn.Module:
        """Get the target layer for Grad-CAM visualization."""
        return self.fire9


def create_sparknet(
    num_classes: int = 6,
    input_channels: int = 3,
    dropout_rate: float = 0.5,
    pretrained_path: Optional[str] = None,
) -> SparkNet:
    """Create a SparkNet model.

    Args:
        num_classes: Number of output classes.
        input_channels: Number of input channels.
        dropout_rate: Dropout probability.
        pretrained_path: Path to pretrained weights.

    Returns:
        SparkNet model instance.
    """
    model = SparkNet(
        num_classes=num_classes,
        input_channels=input_channels,
        dropout_rate=dropout_rate,
    )

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    # Test the model
    model = SparkNet(num_classes=6)
    x = torch.randn(2, 3, 227, 227)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
