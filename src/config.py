"""Configuration management for the solar panel fault detection platform."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Data configuration."""

    rgb: dict[str, str] = Field(default_factory=dict)
    thermal: dict[str, str] = Field(default_factory=dict)
    split_ratio: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.6, "val": 0.2, "test": 0.2}
    )


class FireModuleConfig(BaseModel):
    """Fire module configuration."""

    squeeze_planes: int
    expand_planes: int


class SparkNetConfig(BaseModel):
    """SparkNet model configuration."""

    input_size: tuple[int, int] = (227, 227)
    num_classes: int = 6
    dropout_rate: float = 0.5
    fire_modules: dict[str, FireModuleConfig] = Field(default_factory=dict)


class UNetConfig(BaseModel):
    """U-Net model configuration."""

    input_channels: int = 1
    output_channels: int = 1
    input_size: tuple[int, int] = (256, 256)
    features: list[int] = Field(default_factory=lambda: [64, 128, 256, 512])


class LRSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: str = "step"
    step_size: int = 23
    gamma: float = 0.1


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    type: str = "adam"
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    patience: int = 30
    min_delta: float = 0.001


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-4
    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True


class AugmentationConfig(BaseModel):
    """Data augmentation configuration."""

    enabled: bool = True
    rotation_range: int = 15
    horizontal_flip: bool = True
    vertical_flip: bool = True
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    affine_scale: tuple[float, float] = (0.9, 1.1)


class SeverityWeightsConfig(BaseModel):
    """Severity scoring weights."""

    fault_area: float = 0.4
    temperature: float = 0.4
    growth_rate: float = 0.2


class SeverityThresholdsConfig(BaseModel):
    """Severity thresholds."""

    low: float = 0.3
    high: float = 0.7


class TemperatureConfig(BaseModel):
    """Temperature normalization configuration."""

    normalize_min: float = 20.0
    normalize_max: float = 80.0


class SeverityConfig(BaseModel):
    """Severity scoring configuration."""

    weights: SeverityWeightsConfig = Field(default_factory=SeverityWeightsConfig)
    thresholds: SeverityThresholdsConfig = Field(default_factory=SeverityThresholdsConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)


class CheckpointsConfig(BaseModel):
    """Model checkpoints configuration."""

    save_dir: str = "checkpoints"
    save_best_only: bool = True
    metric: str = "val_f1"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    log_dir: str = "logs"


class APIConfig(BaseModel):
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "postgresql://postgres:postgres@localhost:5432/solar_detection"
    pool_size: int = 10
    max_overflow: int = 20


class StorageConfig(BaseModel):
    """Object storage configuration."""

    endpoint: str = "http://localhost:9000"
    bucket: str = "solar-detection"
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"


class AlertingConfig(BaseModel):
    """Alerting configuration."""

    enabled: bool = True
    high_risk_threshold: str = "High"
    channels: dict[str, Any] = Field(default_factory=dict)


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = True
    endpoint: str = "/metrics"


class Config(BaseModel):
    """Main configuration class."""

    data: DataConfig = Field(default_factory=DataConfig)
    classes: list[str] = Field(
        default_factory=lambda: [
            "Clean",
            "Dusty",
            "Bird-drop",
            "Electrical-damage",
            "Physical-damage",
            "Snow-Covered",
        ]
    )
    sparknet: SparkNetConfig = Field(default_factory=SparkNetConfig)
    unet: UNetConfig = Field(default_factory=UNetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    severity: SeverityConfig = Field(default_factory=SeverityConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses default.

    Returns:
        Loaded configuration object.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        return Config()

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Expand environment variables
    config_dict = _expand_env_vars(config_dict)

    return Config(**config_dict)


def _expand_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively expand environment variables in config values."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _expand_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            result[key] = os.environ.get(env_var, "")
        elif isinstance(value, list):
            result[key] = [
                _expand_env_vars(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
