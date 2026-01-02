"""Tests for neural network models."""

import pytest
import torch

from src.models.sparknet import SparkNet, FireModule
from src.models.unet import UNet, DiceLoss, DiceBCELoss


class TestFireModule:
    """Tests for Fire Module."""

    def test_fire_module_output_shape(self):
        """Test Fire Module produces correct output shape."""
        module = FireModule(
            in_channels=64,
            squeeze_planes=16,
            expand1x1_planes=64,
            expand3x3_planes=64,
            extra_expand3x3=True,
        )

        x = torch.randn(2, 64, 28, 28)
        output = module(x)

        # Output should be 64 + 64 + 64 = 192 channels
        assert output.shape == (2, 192, 28, 28)

    def test_fire_module_without_extra_expand(self):
        """Test Fire Module without extra 3x3 expand."""
        module = FireModule(
            in_channels=64,
            squeeze_planes=16,
            expand1x1_planes=64,
            expand3x3_planes=64,
            extra_expand3x3=False,
        )

        x = torch.randn(2, 64, 28, 28)
        output = module(x)

        # Output should be 64 + 64 = 128 channels
        assert output.shape == (2, 128, 28, 28)


class TestSparkNet:
    """Tests for SparkNet model."""

    @pytest.fixture
    def model(self):
        """Create SparkNet model."""
        return SparkNet(num_classes=6, input_channels=3, dropout_rate=0.5)

    def test_sparknet_forward_pass(self, model):
        """Test SparkNet forward pass produces correct output shape."""
        x = torch.randn(2, 3, 227, 227)
        output = model(x)

        assert output.shape == (2, 6)

    def test_sparknet_feature_extraction(self, model):
        """Test SparkNet feature extraction."""
        x = torch.randn(2, 3, 227, 227)
        features = model.get_features(x)

        # Features should be flattened from GAP
        assert len(features.shape) == 2
        assert features.shape[0] == 2

    def test_sparknet_parameter_count(self, model):
        """Test SparkNet has expected parameter count."""
        num_params = sum(p.numel() for p in model.parameters())
        # Should have around 1M parameters
        assert 500_000 < num_params < 2_000_000

    def test_sparknet_gradient_flow(self, model):
        """Test gradients flow through SparkNet."""
        x = torch.randn(2, 3, 227, 227, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestUNet:
    """Tests for U-Net model."""

    @pytest.fixture
    def model(self):
        """Create U-Net model."""
        return UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])

    def test_unet_forward_pass(self, model):
        """Test U-Net forward pass produces correct output shape."""
        x = torch.randn(2, 1, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)

    def test_unet_different_input_sizes(self, model):
        """Test U-Net works with different input sizes."""
        for size in [128, 256, 512]:
            x = torch.randn(1, 1, size, size)
            output = model(x)
            assert output.shape == (1, 1, size, size)

    def test_unet_encoder_features(self, model):
        """Test U-Net encoder feature extraction."""
        x = torch.randn(1, 1, 256, 256)
        features = model.get_encoder_features(x)

        assert len(features) == 5  # 5 levels
        # Each level should have increasing depth
        assert features[0].shape[1] == 64
        assert features[1].shape[1] == 128


class TestLossFunctions:
    """Tests for loss functions."""

    def test_dice_loss(self):
        """Test Dice loss computation."""
        loss_fn = DiceLoss()

        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert 0 <= loss <= 1

    def test_dice_bce_loss(self):
        """Test combined Dice-BCE loss."""
        loss_fn = DiceBCELoss()

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert loss >= 0


class TestModelSaving:
    """Tests for model saving and loading."""

    def test_sparknet_save_load(self, tmp_path):
        """Test saving and loading SparkNet."""
        model = SparkNet(num_classes=6)
        model.eval()  # Set to eval mode for deterministic output
        x = torch.randn(1, 3, 227, 227)
        with torch.no_grad():
            original_output = model(x)

        # Save
        path = tmp_path / "sparknet.pth"
        torch.save(model.state_dict(), path)

        # Load
        model2 = SparkNet(num_classes=6)
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()  # Set to eval mode for deterministic output
        with torch.no_grad():
            loaded_output = model2(x)

        assert torch.allclose(original_output, loaded_output)

    def test_unet_save_load(self, tmp_path):
        """Test saving and loading U-Net."""
        model = UNet()
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            original_output = model(x)

        path = tmp_path / "unet.pth"
        torch.save(model.state_dict(), path)

        model2 = UNet()
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            loaded_output = model2(x)

        assert torch.allclose(original_output, loaded_output)
