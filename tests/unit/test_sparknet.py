"""Unit tests for SparkNet model."""

import pytest
import torch

from src.models.sparknet import SparkNet, FireModule


class TestFireModule:
    """Tests for FireModule component."""

    def test_fire_module_output_shape_with_extra(self):
        """Test that FireModule produces correct output shape with extra expand."""
        batch_size = 4
        in_channels = 64
        squeeze_planes = 16
        expand1x1_planes = 64
        expand3x3_planes = 64

        # Default has extra_expand3x3=True
        module = FireModule(
            in_channels=in_channels,
            squeeze_planes=squeeze_planes,
            expand1x1_planes=expand1x1_planes,
            expand3x3_planes=expand3x3_planes,
            extra_expand3x3=True,
        )

        x = torch.randn(batch_size, in_channels, 56, 56)
        output = module(x)

        # With extra: expand1x1 + expand3x3 + expand3x3_extra
        expected_channels = expand1x1_planes + expand3x3_planes * 2
        assert output.shape == (batch_size, expected_channels, 56, 56)

    def test_fire_module_without_extra_expand(self):
        """Test FireModule without extra 3x3 expansion."""
        batch_size = 4
        in_channels = 64
        squeeze_planes = 16
        expand1x1_planes = 64
        expand3x3_planes = 64

        module = FireModule(
            in_channels=in_channels,
            squeeze_planes=squeeze_planes,
            expand1x1_planes=expand1x1_planes,
            expand3x3_planes=expand3x3_planes,
            extra_expand3x3=False,
        )

        x = torch.randn(batch_size, in_channels, 56, 56)
        output = module(x)

        # Without extra: expand1x1 + expand3x3
        expected_channels = expand1x1_planes + expand3x3_planes
        assert output.shape == (batch_size, expected_channels, 56, 56)


class TestSparkNet:
    """Tests for SparkNet model."""

    @pytest.fixture
    def model(self):
        """Create a SparkNet model for testing."""
        return SparkNet(num_classes=6, input_channels=3, dropout_rate=0.5)

    def test_sparknet_forward_pass(self, model):
        """Test SparkNet forward pass produces correct output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 227, 227)

        output = model(x)

        assert output.shape == (batch_size, 6)

    def test_sparknet_single_image(self, model):
        """Test SparkNet with single image."""
        x = torch.randn(1, 3, 227, 227)
        output = model(x)

        assert output.shape == (1, 6)

    def test_sparknet_eval_mode(self, model):
        """Test SparkNet in evaluation mode (no dropout)."""
        model.eval()
        x = torch.randn(2, 3, 227, 227)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 6)

    def test_sparknet_different_num_classes(self):
        """Test SparkNet with different number of classes."""
        for num_classes in [2, 5, 10]:
            model = SparkNet(num_classes=num_classes)
            x = torch.randn(2, 3, 227, 227)
            output = model(x)

            assert output.shape == (2, num_classes)

    def test_sparknet_get_features(self, model):
        """Test SparkNet feature extraction."""
        x = torch.randn(2, 3, 227, 227)
        features = model.get_features(x)

        # Features should be 1D per sample
        assert len(features.shape) == 2
        assert features.shape[0] == 2

    def test_sparknet_get_cam_target_layer(self, model):
        """Test SparkNet CAM target layer retrieval."""
        target_layer = model.get_cam_target_layer()

        # Should return a layer that can be used for Grad-CAM
        assert target_layer is not None

    def test_sparknet_parameter_count(self, model):
        """Test SparkNet has reasonable parameter count."""
        num_params = sum(p.numel() for p in model.parameters())

        # SparkNet should have around 2-3 million parameters
        assert 1_000_000 < num_params < 5_000_000

    def test_sparknet_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        model.train()
        x = torch.randn(2, 3, 227, 227, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_sparknet_output_probabilities(self, model):
        """Test that softmax outputs valid probabilities."""
        model.eval()
        x = torch.randn(2, 3, 227, 227)

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)

        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        # All probabilities should be non-negative
        assert (probs >= 0).all()
