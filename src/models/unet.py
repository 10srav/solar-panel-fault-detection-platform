"""U-Net Segmentation Model for Thermal Image Fault Localization.

U-Net architecture for pixel-level fault segmentation in thermal images
of solar panels, identifying WHERE faults are located.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution Block.

    Consists of: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        mid_channels: Number of channels in the middle (default: same as out_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution block."""
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downsampling Block (Encoder).

    MaxPool -> DoubleConv

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through down block."""
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling Block (Decoder).

    Upsample/ConvTranspose -> Concatenate skip connection -> DoubleConv

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        bilinear: Use bilinear upsampling instead of transposed convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through up block.

        Args:
            x1: Input from previous layer (to be upsampled).
            x2: Skip connection from encoder.

        Returns:
            Output tensor after upsampling and convolution.
        """
        x1 = self.up(x1)

        # Handle size mismatch due to pooling
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output Convolution Layer.

    1x1 convolution to map to desired number of output channels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (segmentation classes).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output convolution."""
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for Thermal Image Segmentation.

    Standard U-Net architecture with encoder-decoder structure and skip connections.

    Args:
        in_channels: Number of input channels (1 for grayscale thermal, 3 for RGB thermal).
        out_channels: Number of output channels (segmentation classes).
        features: List of feature sizes for each encoder level.
        bilinear: Use bilinear upsampling instead of transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Optional[list[int]] = None,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])

        # Encoder (downsampling path)
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])

        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features[3], features[3] * 2 // factor)

        # Decoder (upsampling path)
        self.up1 = UpBlock(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = UpBlock(features[3], features[2] // factor, bilinear)
        self.up3 = UpBlock(features[2], features[1] // factor, bilinear)
        self.up4 = UpBlock(features[1], features[0], bilinear)

        # Output layer
        self.outc = OutConv(features[0], out_channels)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Segmentation mask of shape (batch, out_channels, height, width).
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)

        return logits

    def get_encoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get intermediate encoder features for analysis.

        Args:
            x: Input tensor.

        Returns:
            List of feature maps from each encoder level.
        """
        features = []
        x1 = self.inc(x)
        features.append(x1)

        x2 = self.down1(x1)
        features.append(x2)

        x3 = self.down2(x2)
        features.append(x3)

        x4 = self.down3(x3)
        features.append(x4)

        x5 = self.down4(x4)
        features.append(x5)

        return features


class UNetWithAttention(nn.Module):
    """U-Net with Attention Gates for improved segmentation.

    Enhanced version with attention mechanisms in skip connections.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        features: List of feature sizes for each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])

        # Encoder
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        self.down4 = DownBlock(features[3], features[3] * 2)

        # Attention gates
        self.att1 = AttentionGate(features[3], features[3] * 2, features[3])
        self.att2 = AttentionGate(features[2], features[3], features[2])
        self.att3 = AttentionGate(features[1], features[2], features[1])
        self.att4 = AttentionGate(features[0], features[1], features[0])

        # Decoder
        self.up1 = UpBlock(features[3] * 2 + features[3], features[3], bilinear=False)
        self.up2 = UpBlock(features[3] + features[2], features[2], bilinear=False)
        self.up3 = UpBlock(features[2] + features[1], features[1], bilinear=False)
        self.up4 = UpBlock(features[1] + features[0], features[0], bilinear=False)

        # Output
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with attention
        x4_att = self.att1(x4, x5)
        x = self.up1(x5, x4_att)

        x3_att = self.att2(x3, x)
        x = self.up2(x, x3_att)

        x2_att = self.att3(x2, x)
        x = self.up3(x, x2_att)

        x1_att = self.att4(x1, x)
        x = self.up4(x, x1_att)

        return self.outc(x)


class AttentionGate(nn.Module):
    """Attention Gate for U-Net.

    Applies attention to skip connection features based on gating signal.

    Args:
        F_g: Number of channels in gating signal.
        F_l: Number of channels in skip connection.
        F_int: Number of intermediate channels.
    """

    def __init__(self, F_l: int, F_g: int, F_int: int) -> None:
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Apply attention gate.

        Args:
            x: Skip connection features.
            g: Gating signal from decoder.

        Returns:
            Attention-weighted features.
        """
        # Resize gating signal to match skip connection
        g1 = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        g1 = self.W_g(g1)

        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def create_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    features: Optional[list[int]] = None,
    with_attention: bool = False,
    pretrained_path: Optional[str] = None,
) -> nn.Module:
    """Create a U-Net model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        features: List of feature sizes.
        with_attention: Use attention gates.
        pretrained_path: Path to pretrained weights.

    Returns:
        U-Net model instance.
    """
    if with_attention:
        model = UNetWithAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )
    else:
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


# Loss functions for segmentation
class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Dice loss."""
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE Loss."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate combined loss."""
        return self.bce_loss(predictions, targets) + self.dice_loss(predictions, targets)


if __name__ == "__main__":
    # Test the model
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test attention model
    model_att = UNetWithAttention(in_channels=1, out_channels=1)
    output_att = model_att(x)
    print(f"Attention U-Net output shape: {output_att.shape}")
    print(
        f"Attention U-Net parameters: {sum(p.numel() for p in model_att.parameters()):,}"
    )
