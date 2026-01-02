"""Neural network models for solar panel fault detection."""

from src.models.sparknet import SparkNet, FireModule
from src.models.unet import UNet

__all__ = ["SparkNet", "FireModule", "UNet"]
