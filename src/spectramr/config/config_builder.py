"""
Phase 7.2: Configuration Builder
Builds training configurations with fluent API.
"""

import json
from pathlib import Path
from typing import Any

import yaml

from spectramr.config.schemas.loss import LossConfigSchema


class ConfigBuilder:
    """Builds training configurations with fluent API."""

    def __init__(self):
        """Initialize config builder."""
        from spectramr.models.losses.loss_recommender import LossRecommender

        self._config_dict: dict[str, Any] = {
            "model": {},
            "data": {},
            "optimization": {},
            "losses": {},
        }
        self._task_type: Any | None = None
        self._loss_recommender = LossRecommender()

    def for_task(self, task: str) -> "ConfigBuilder":
        """Set task type.

        Args:
            task: Task type (reconstruction, synthesis, segmentation, etc)

        Returns:
            self for chaining
        """
        try:
            from spectramr.models.losses.loss_recommender import TaskType

            self._task_type = TaskType(task)
        except ValueError:
            raise ValueError(f"Unknown task: {task}. Supported: {[t.value for t in TaskType]}")

        return self

    def with_architecture(self, model_type: str) -> "ConfigBuilder":
        """Set model architecture.

        Args:
            model_type: Model type (standard_unet, swin_unet, etc)

        Returns:
            self for chaining
        """
        self._config_dict["model"]["model_type"] = model_type
        return self

    def with_batch_size(self, batch_size: int) -> "ConfigBuilder":
        """Set batch size.

        Args:
            batch_size: Training batch size

        Returns:
            self for chaining
        """
        self._config_dict["data"]["batch_size"] = batch_size
        return self

    def with_learning_rate(self, lr: float) -> "ConfigBuilder":
        """Set learning rate.

        Args:
            lr: Learning rate value

        Returns:
            self for chaining
        """
        self._config_dict["optimization"]["learning_rate"] = lr
        return self

    def with_epochs(self, epochs: int) -> "ConfigBuilder":
        """Set number of training epochs.

        Args:
            epochs: Number of epochs

        Returns:
            self for chaining
        """
        self._config_dict["optimization"]["num_epochs"] = epochs
        return self

    def with_optimizer(self, optimizer: str = "adam") -> "ConfigBuilder":
        """Set optimizer type.

        Args:
            optimizer: Optimizer type (adam, sgd, etc)

        Returns:
            self for chaining
        """
        self._config_dict["optimization"]["optimizer"] = optimizer
        return self

    def enable_metrics(self) -> "ConfigBuilder":
        """Enable metrics tracking.

        Returns:
            self for chaining
        """
        self._config_dict["training"] = self._config_dict.get("training", {})
        self._config_dict["training"]["track_metrics"] = True
        return self

    def enable_mixed_precision(self) -> "ConfigBuilder":
        """Enable automatic mixed precision.

        Returns:
            self for chaining
        """
        self._config_dict["optimization"] = self._config_dict.get("optimization", {})
        self._config_dict["optimization"]["use_amp"] = True
        return self

    def add_curriculum_learning(self) -> "ConfigBuilder":
        """Enable curriculum learning.

        Returns:
            self for chaining
        """
        self._config_dict["training"] = self._config_dict.get("training", {})
        self._config_dict["training"]["curriculum"] = {
            "enabled": True,
            "schedule": "linear",
        }
        return self

    def use_gan_losses(self) -> "ConfigBuilder":
        """Enable GAN losses.

        Returns:
            self for chaining
        """
        self._config_dict["losses"] = self._config_dict.get("losses", {})
        self._config_dict["losses"]["use_gan"] = True
        return self

    def use_perceptual_losses(self) -> "ConfigBuilder":
        """Enable perceptual losses.

        Returns:
            self for chaining
        """
        self._config_dict["losses"] = self._config_dict.get("losses", {})
        self._config_dict["losses"]["use_perceptual"] = True
        return self

    def with_loss_config(self, config: LossConfigSchema) -> "ConfigBuilder":
        """Set custom loss configuration.

        Args:
            config: Loss configuration

        Returns:
            self for chaining
        """
        self._config_dict["losses"]["custom_config"] = config
        return self

    def with_checkpoint_dir(self, path: str) -> "ConfigBuilder":
        """Set checkpoint directory.

        Args:
            path: Directory path

        Returns:
            self for chaining
        """
        self._config_dict["training"] = self._config_dict.get("training", {})
        self._config_dict["training"]["checkpoint_dir"] = path
        return self

    def with_log_dir(self, path: str) -> "ConfigBuilder":
        """Set log directory.

        Args:
            path: Directory path

        Returns:
            self for chaining
        """
        self._config_dict["training"] = self._config_dict.get("training", {})
        self._config_dict["training"]["log_dir"] = path
        return self

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if "model_type" not in self._config_dict["model"]:
            raise ValueError("Model type not specified")

        if "batch_size" not in self._config_dict["data"]:
            raise ValueError("Batch size not specified")

        if "learning_rate" not in self._config_dict["optimization"]:
            raise ValueError("Learning rate not specified")

        return True

    def build(self) -> dict[str, Any]:
        """Build configuration dictionary.

        Returns:
            Complete configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.validate()

        # Auto-generate loss config if task is specified
        if self._task_type and "custom_config" not in self._config_dict["losses"]:
            recommendations = self._loss_recommender.recommend(
                task_type=self._task_type,
                use_gan=self._config_dict["losses"].get("use_gan"),
                use_perceptual=self._config_dict["losses"].get("use_perceptual"),
            )
            self._config_dict["losses"]["config"] = recommendations.suggested_config

        return self._config_dict

    def to_yaml(self, path: str | None = None) -> str:
        """Convert to YAML format.

        Args:
            path: Optional path to save YAML file

        Returns:
            YAML string
        """
        config_dict = self.build()

        # Prepare for YAML serialization
        yaml_dict = self._prepare_for_yaml(config_dict)

        yaml_str = yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False)

        if path:
            Path(path).write_text(yaml_str)

        return yaml_str

    def to_json(self, path: str | None = None) -> str:
        """Convert to JSON format.

        Args:
            path: Optional path to save JSON file

        Returns:
            JSON string
        """
        config_dict = self.build()

        # Prepare for JSON serialization
        json_dict = self._prepare_for_yaml(config_dict)

        json_str = json.dumps(json_dict, indent=2)

        if path:
            Path(path).write_text(json_str)

        return json_str

    def save(self, path: str) -> Path:
        """Save configuration to file.

        Args:
            path: File path (extension determines format)

        Returns:
            Path to saved file
        """
        path = Path(path)

        if path.suffix == ".yaml" or path.suffix == ".yml":
            self.to_yaml(str(path))
        elif path.suffix == ".json":
            self.to_json(str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        return path

    def _prepare_for_yaml(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Prepare config dictionary for YAML serialization.

        Args:
            config_dict: Configuration dictionary

        Returns:
            YAML-ready dictionary
        """
        result = {}

        for key, value in config_dict.items():
            if isinstance(value, dict):
                result[key] = self._prepare_for_yaml(value)
            elif hasattr(value, "model_dump"):
                # Pydantic v2 model
                result[key] = value.model_dump()
            else:
                result[key] = value

        return result

    def summary(self) -> str:
        """Get configuration summary.

        Returns:
            Summary string
        """
        lines = ["Configuration Summary:"]
        lines.append("=" * 50)

        config = self.build()

        if config.get("model"):
            lines.append(f"Model: {config['model'].get('model_type', 'Not specified')}")

        if config.get("data"):
            lines.append(f"Batch size: {config['data'].get('batch_size', 'Not specified')}")

        if config.get("optimization"):
            opt = config["optimization"]
            lines.append(f"Optimizer: {opt.get('optimizer', 'adam').upper()}")
            lines.append(f"Learning rate: {opt.get('learning_rate', 'Not specified')}")
            lines.append(f"Epochs: {opt.get('num_epochs', 'Not specified')}")

            if opt.get("use_amp"):
                lines.append("Mixed precision: Enabled")

        if self._task_type:
            lines.append(f"Task: {self._task_type.value}")

        return "\n".join(lines)


# Preset templates


class ConfigTemplates:
    """Predefined configuration templates."""

    @staticmethod
    def reconstruction_baseline() -> ConfigBuilder:
        """Baseline reconstruction configuration.

        Returns:
            ConfigBuilder with baseline settings
        """
        return (
            ConfigBuilder()
            .for_task("reconstruction")
            .with_architecture("standard_unet")
            .with_batch_size(32)
            .with_learning_rate(1e-3)
            .with_epochs(100)
            .enable_metrics()
        )

    @staticmethod
    def reconstruction_advanced() -> ConfigBuilder:
        """Advanced reconstruction configuration.

        Returns:
            ConfigBuilder with advanced settings
        """
        return (
            ConfigBuilder()
            .for_task("reconstruction")
            .with_architecture("swin_unet")
            .with_batch_size(16)
            .with_learning_rate(5e-4)
            .with_epochs(150)
            .enable_metrics()
            .enable_mixed_precision()
            .use_perceptual_losses()
        )

    @staticmethod
    def synthesis_gan() -> ConfigBuilder:
        """GAN-based synthesis configuration.

        Returns:
            ConfigBuilder with GAN settings
        """
        return (
            ConfigBuilder()
            .for_task("synthesis")
            .with_architecture("residual_unet")
            .with_batch_size(32)
            .with_learning_rate(2e-4)
            .with_epochs(200)
            .enable_metrics()
            .use_gan_losses()
        )

    @staticmethod
    def super_resolution() -> ConfigBuilder:
        """Super-resolution configuration.

        Returns:
            ConfigBuilder with SR settings
        """
        return (
            ConfigBuilder()
            .for_task("super_resolution")
            .with_architecture("swin_unet")
            .with_batch_size(16)
            .with_learning_rate(1e-3)
            .with_epochs(100)
            .enable_metrics()
            .use_perceptual_losses()
        )

    @staticmethod
    def denoising() -> ConfigBuilder:
        """Denoising configuration.

        Returns:
            ConfigBuilder with denoising settings
        """
        return (
            ConfigBuilder()
            .for_task("denoising")
            .with_architecture("standard_unet")
            .with_batch_size(32)
            .with_learning_rate(1e-3)
            .with_epochs(50)
            .enable_metrics()
        )


# Convenience functions


def build_config(task: str) -> ConfigBuilder:
    """Create config builder for task.

    Args:
        task: Task type

    Returns:
        ConfigBuilder instance
    """
    return ConfigBuilder().for_task(task)


def get_template(template_name: str) -> ConfigBuilder:
    """Get preset template.

    Args:
        template_name: Template name (reconstruction_baseline, synthesis_gan, etc)

    Returns:
        ConfigBuilder with preset settings
    """
    template_map = {
        "reconstruction_baseline": ConfigTemplates.reconstruction_baseline,
        "reconstruction_advanced": ConfigTemplates.reconstruction_advanced,
        "synthesis_gan": ConfigTemplates.synthesis_gan,
        "super_resolution": ConfigTemplates.super_resolution,
        "denoising": ConfigTemplates.denoising,
    }

    if template_name not in template_map:
        raise ValueError(
            f"Unknown template: {template_name}. Available: {list(template_map.keys())}"
        )

    return template_map[template_name]()
