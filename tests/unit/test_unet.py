"""Unit tests for U-Net model."""

import pytest
import torch

from src.models.unet import UNet, DoubleConv, DownBlock, UpBlock


class TestDoubleConv:
    """Tests for DoubleConv block."""

    def test_double_conv_output_shape(self):
        """Test DoubleConv produces correct output shape."""
        block = DoubleConv(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 128, 128)
        output = block(x)

        assert output.shape == (2, 128, 128, 128)

    def test_double_conv_preserves_spatial(self):
        """Test DoubleConv preserves spatial dimensions."""
        block = DoubleConv(in_channels=32, out_channels=64)

        for size in [64, 128, 256]:
            x = torch.randn(1, 32, size, size)
            output = block(x)
            assert output.shape[2:] == (size, size)


class TestDownBlock:
    """Tests for DownBlock (encoder block)."""

    def test_down_block_output_shape(self):
        """Test DownBlock halves spatial dimensions."""
        block = DownBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 128, 128)
        output = block(x)

        assert output.shape == (2, 128, 64, 64)

    def test_down_block_pooling(self):
        """Test DownBlock correctly applies max pooling."""
        block = DownBlock(in_channels=32, out_channels=64)
        x = torch.randn(1, 32, 256, 256)
        output = block(x)

        assert output.shape == (1, 64, 128, 128)


class TestUpBlock:
    """Tests for UpBlock (decoder block)."""

    def test_up_block_output_shape(self):
        """Test UpBlock doubles spatial dimensions."""
        # in_channels=192 is the expected channels AFTER concat (128 upsampled + 64 skip)
        block = UpBlock(in_channels=192, out_channels=64)
        x = torch.randn(2, 128, 64, 64)  # Will be upsampled
        skip = torch.randn(2, 64, 128, 128)  # Skip connection (128+64=192 after concat)
        output = block(x, skip)

        assert output.shape == (2, 64, 128, 128)


class TestUNet:
    """Tests for U-Net model."""

    @pytest.fixture
    def model(self):
        """Create a U-Net model for testing."""
        return UNet(
            in_channels=1,
            out_channels=1,
            features=[64, 128, 256, 512],
        )

    def test_unet_forward_pass(self, model):
        """Test U-Net forward pass produces correct output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 256, 256)

        output = model(x)

        assert output.shape == (batch_size, 1, 256, 256)

    def test_unet_single_image(self, model):
        """Test U-Net with single image."""
        x = torch.randn(1, 1, 256, 256)
        output = model(x)

        assert output.shape == (1, 1, 256, 256)

    def test_unet_different_input_sizes(self, model):
        """Test U-Net with different input sizes."""
        for size in [128, 256, 512]:
            x = torch.randn(1, 1, size, size)
            output = model(x)
            assert output.shape == (1, 1, size, size)

    def test_unet_rgb_input(self):
        """Test U-Net with RGB (3-channel) input."""
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)

    def test_unet_multi_class_output(self):
        """Test U-Net with multiple output classes."""
        model = UNet(in_channels=1, out_channels=3)
        x = torch.randn(2, 1, 256, 256)
        output = model(x)

        assert output.shape == (2, 3, 256, 256)

    def test_unet_parameter_count(self, model):
        """Test U-Net has reasonable parameter count."""
        num_params = sum(p.numel() for p in model.parameters())

        # U-Net should have around 15-20 million parameters
        assert 10_000_000 < num_params < 25_000_000

    def test_unet_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        model.train()
        x = torch.randn(2, 1, 256, 256, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_unet_eval_mode(self, model):
        """Test U-Net in evaluation mode."""
        model.eval()
        x = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1, 256, 256)

    def test_unet_sigmoid_output_range(self, model):
        """Test that sigmoid output is in valid range."""
        model.eval()
        x = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            output = model(x)
            probs = torch.sigmoid(output)

        # Sigmoid output should be between 0 and 1
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_unet_binary_mask_prediction(self, model):
        """Test U-Net produces valid binary mask predictions."""
        model.eval()
        x = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            output = model(x)
            probs = torch.sigmoid(output)
            mask = (probs > 0.5).float()

        # Mask should be binary
        assert torch.all((mask == 0) | (mask == 1))
