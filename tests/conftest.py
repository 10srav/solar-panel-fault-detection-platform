"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Get the test device (CPU for CI)."""
    return torch.device("cpu")


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image tensor."""
    return torch.randn(1, 3, 227, 227)


@pytest.fixture
def sample_thermal_image():
    """Create a sample thermal image tensor."""
    return torch.randn(1, 1, 256, 256)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 1.0
    return mask


@pytest.fixture
def sample_thermal_array():
    """Create a sample thermal image array."""
    return np.random.uniform(20, 80, (100, 100)).astype(np.float32)
