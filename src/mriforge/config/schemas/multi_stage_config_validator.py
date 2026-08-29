"""Multi-Stage Training Configuration Schema Validator
Validates and normalizes multi-stage training configurations
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class MultiStageMode(Enum):
    """Multi-stage training modes"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class StageType(Enum):
    """Stage types for multi-stage training"""

    VAE = "vae"
    DIFFUSION = "diffusion"
    GAN = "gan"
    RECONSTRUCTION = "reconstruction"
    FLOW = "flow"


class DependencyList(list[str]):
    """List wrapper preserving display aliases for normalized dependencies."""

    def __init__(
        self,
        normalized: list[str],
        display: list[str] | None = None,
    ) -> None:
        """__init__.

        Args:
            normalized (list[str]): Description.
            display (Optional[list[str]]): Description.
        """
        super().__init__(normalized)
        self._display = list(display) if display is not None else list(normalized)

    def __eq__(self, other: object) -> bool:
        """__eq__.

        Args:
            other (object): Description.
        Returns:
            bool: Description.
        """
        if isinstance(other, list):
            return self._display == list(other)
        return super().__eq__(other)

    def __repr__(self) -> str:
        """__repr__.

        Returns:
            str: Description.
        """
        return repr(self._display)

    def __str__(self) -> str:
        """__str__.

        Returns:
            str: Description.
        """
        return str(self._display)

    def display(self) -> list[str]:
        """display.

        Returns:
            list[str]: Description.
        """
        return list(self._display)


@dataclass
class StageConfig:
    """Configuration for a single training stage"""

    name: str
    type: StageType
    model_type: str | None
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    optimizer: str = "adam"
    model_config: dict[str, Any] = field(default_factory=dict)
    data_config: dict[str, Any] = field(default_factory=dict)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    save_latents: bool = False
    input_latent_path: str | None = None
    output_latent_path: str | None = None
    dependencies: list[str] = field(default_factory=list)
    conditional_logic: dict[str, Any] = field(default_factory=dict)
    mode: str = "single"
    encoder: dict[str, Any] = field(default_factory=dict)
    decoder: dict[str, Any] = field(default_factory=dict)
    latent_exports: dict[str, Any] = field(default_factory=dict)
    input_bindings: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataFlowConfig:
    """Configuration for data flow between stages"""

    input_transforms: list[dict[str, Any]] = field(default_factory=list)
    output_transforms: list[dict[str, Any]] = field(default_factory=list)
    latent_sharing: dict[str, Any] = field(default_factory=dict)
    cross_stage_validation: dict[str, Any] = field(default_factory=dict)
    links: list[dict[str, Any]] = field(default_factory=list)
    manifest_path: str | None = None


@dataclass
class MultiStageTrainingConfig:
    """Complete multi-stage training configuration"""

    experiment_name: str
    mode: MultiStageMode = MultiStageMode.SEQUENTIAL
    stages: list[StageConfig] = field(default_factory=list)
    data_flow: DataFlowConfig = field(default_factory=DataFlowConfig)
    global_config: dict[str, Any] = field(default_factory=dict)
    validation_config: dict[str, Any] = field(default_factory=dict)
    monitoring_config: dict[str, Any] = field(default_factory=dict)


class MultiStageConfigValidator:
    """Validator for multi-stage training configurations"""

    def __init__(self) -> None:
        """__init__."""
        self.logger = logging.getLogger(__name__)

    def _derive_dependency_aliases(self, stage: StageConfig) -> dict[str, str]:
        """_derive_dependency_aliases.

        Args:
            stage (StageConfig): Description.
        Returns:
            dict[str, str]: Description.
        """
        aliases: dict[str, str] = {}
        if stage.type == StageType.GAN:
            base_names = {stage.name}
            if stage.name.endswith("_stage"):
                base_names.add(stage.name[: -len("_stage")])
            if stage.model_type:
                base_names.add(stage.model_type)
            for base in base_names:
                aliases[f"{base}_generator"] = stage.name
                aliases[f"{base}_discriminator"] = stage.name
        return aliases

    def validate_and_normalize(
        self,
        config_dict: dict[str, Any],
    ) -> MultiStageTrainingConfig:
        """Validate and normalize multi-stage training configuration

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Normalized MultiStageTrainingConfig object

        """
        """
        Validate and normalize multi-stage training configuration

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Normalized MultiStageTrainingConfig object
        """
        try:
            # Validate top-level structure
            self._validate_top_level_structure(config_dict)

            # Extract and validate experiment name
            experiment_name = config_dict.get("experiment_name", "multi_stage_default")

            # Validate and normalize mode
            mode_str = config_dict.get("mode", "sequential")
            mode = self._validate_mode(mode_str)

            # Validate and normalize stages
            stages_config = config_dict.get("stages", [])
            stages = self._validate_stages(stages_config)

            # Validate and normalize data flow
            data_flow_config = config_dict.get("data_flow", {})
            data_flow = self._validate_data_flow(data_flow_config)

            # Extract global configuration
            global_config = config_dict.get("global_config", {})

            # Validate validation configuration
            validation_config = self._validate_validation_config(
                config_dict.get("validation", {}),
            )

            # Validate monitoring configuration
            monitoring_config = self._validate_monitoring_config(
                config_dict.get("monitoring", {}),
            )

            # Validate stage dependencies and data flow
            self._validate_stage_dependencies(stages, data_flow)

            return MultiStageTrainingConfig(
                experiment_name=experiment_name,
                mode=mode,
                stages=stages,
                data_flow=data_flow,
                global_config=global_config,
                validation_config=validation_config,
                monitoring_config=monitoring_config,
            )

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e!s}")
            raise ValueError(f"Invalid multi-stage configuration: {e!s}")

    def _validate_top_level_structure(self, config: dict[str, Any]) -> None:
        """Validate top-level configuration structure"""
        required_keys = ["stages"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        if not isinstance(config["stages"], list) or len(config["stages"]) == 0:
            raise ValueError("Configuration must contain at least one stage")

    def _validate_mode(self, mode_str: str) -> MultiStageMode:
        """Validate and convert mode string to enum"""
        try:
            return MultiStageMode(mode_str.lower())
        except ValueError:
            valid_modes = [m.value for m in MultiStageMode]
            raise ValueError(f"Invalid mode '{mode_str}'. Valid modes: {valid_modes}")

    def _validate_stages(
        self,
        stages_config: list[dict[str, Any]],
    ) -> list[StageConfig]:
        """Validate and normalize stage configurations"""
        """Validate and normalize stage configurations"""
        stages = []

        for i, stage_dict in enumerate(stages_config):
            try:
                stage = self._validate_single_stage(stage_dict, i)
                stages.append(stage)
            except Exception as e:
                raise ValueError(f"Invalid stage {i}: {e!s}")

        return stages

    def _validate_single_stage(
        self,
        stage_dict: dict[str, Any],
        index: int,
    ) -> StageConfig:
        """Validate a single stage configuration"""
        # Required fields
        if "name" not in stage_dict:
            raise ValueError(f"Stage {index} missing required field: name")

        name = stage_dict["name"]

        encoder_section = stage_dict.get("encoder") or {}
        decoder_section = stage_dict.get("decoder") or {}
        if not isinstance(encoder_section, dict):
            raise ValueError(
                f"Stage {name} encoder section must be a mapping if provided",
            )
        if not isinstance(decoder_section, dict):
            raise ValueError(
                f"Stage {name} decoder section must be a mapping if provided",
            )

        mode = stage_dict.get("mode")
        if not mode:
            mode = "encoder_decoder" if (encoder_section or decoder_section) else "single"

        model_type = stage_dict.get("model_type")
        if not model_type and decoder_section:
            model_type = decoder_section.get("model_type")
        if not model_type and encoder_section:
            model_type = encoder_section.get("model_type")
        input_bindings = stage_dict.get("inputs")
        if input_bindings is None:
            input_bindings = stage_dict.get("input_bindings", {})
        if input_bindings and not isinstance(input_bindings, dict):
            raise ValueError(
                f"Stage {name} inputs section must be a mapping if provided",
            )

        if model_type:
            stage_type = self._infer_stage_type(model_type)
        else:
            if input_bindings:
                stage_type = StageType.RECONSTRUCTION
            else:
                raise ValueError(
                    f"Stage {name} must define model_type or decoder.model_type",
                )

        # Optional fields with defaults
        epochs = self._validate_positive_int(stage_dict.get("epochs", 100), "epochs")
        batch_size = self._validate_positive_int(
            stage_dict.get("batch_size", 8),
            "batch_size",
        )
        learning_rate = self._validate_positive_float(
            stage_dict.get("learning_rate", 0.001),
            "learning_rate",
        )

        optimizer = stage_dict.get("optimizer", "adam")
        if optimizer not in ["adam", "adamw", "sgd", "rmsprop"]:
            raise ValueError(f"Invalid optimizer '{optimizer}' for stage {name}")

        model_config = stage_dict.get("model_config", {})
        data_config = stage_dict.get("data_config", {})
        success_criteria = stage_dict.get("success_criteria", {})
        save_latents = stage_dict.get("save_latents", False)

        input_latent_path = stage_dict.get("input_latent_path")
        output_latent_path = stage_dict.get("output_latent_path")
        dependencies = stage_dict.get("dependencies", [])
        conditional_logic = stage_dict.get("conditional_logic", {})

        outputs_section = stage_dict.get("outputs", {})
        if outputs_section and not isinstance(outputs_section, dict):
            raise ValueError(
                f"Stage {name} outputs section must be a mapping if provided",
            )
        latent_exports = stage_dict.get("latent_exports")
        if latent_exports is None and isinstance(outputs_section, dict):
            latent_exports = outputs_section.get("latents", {})
        if latent_exports and not isinstance(latent_exports, dict):
            raise ValueError(
                f"Stage {name} outputs.latents must be a mapping if provided",
            )

        return StageConfig(
            name=name,
            type=stage_type,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            model_config=model_config,
            data_config=data_config,
            success_criteria=success_criteria,
            save_latents=save_latents,
            input_latent_path=input_latent_path,
            output_latent_path=output_latent_path,
            dependencies=dependencies,
            conditional_logic=conditional_logic,
            mode=mode,
            encoder=dict(encoder_section),
            decoder=dict(decoder_section),
            latent_exports=(dict(latent_exports) if isinstance(latent_exports, dict) else {}),
            input_bindings=(dict(input_bindings) if isinstance(input_bindings, dict) else {}),
        )

    def _infer_stage_type(self, model_type: str) -> StageType:
        """Infer stage type from model type"""
        model_lower = model_type.lower()

        if any(keyword in model_lower for keyword in ["vae", "vqvae", "vq_vae", "vqgan"]):
            return StageType.VAE
        if any(keyword in model_lower for keyword in ["diffusion", "cold", "score_based", "ddpm"]):
            return StageType.DIFFUSION
        if any(keyword in model_lower for keyword in ["gan", "adversarial"]):
            return StageType.GAN
        if any(keyword in model_lower for keyword in ["flow", "normalizing_flow"]):
            return StageType.FLOW
        return StageType.RECONSTRUCTION

    def _validate_data_flow(
        self,
        data_flow_config: dict[str, Any],
    ) -> DataFlowConfig:
        """Validate data flow configuration"""

        input_transforms = data_flow_config.get("input_transforms", [])
        output_transforms = data_flow_config.get("output_transforms", [])
        latent_sharing = data_flow_config.get("latent_sharing", {})
        cross_stage_validation = data_flow_config.get(
            "cross_stage_validation",
            {},
        )
        links = data_flow_config.get("links", [])
        manifest_path = data_flow_config.get("manifest_path")

        return DataFlowConfig(
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            latent_sharing=latent_sharing,
            cross_stage_validation=cross_stage_validation,
            links=links,
            manifest_path=manifest_path,
        )

    def _validate_validation_config(
        self,
        validation_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate validation configuration"""
        # Add validation logic here
        return validation_config

    def _validate_monitoring_config(
        self,
        monitoring_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate monitoring configuration"""
        """Validate monitoring configuration"""
        # Add validation logic here
        return monitoring_config

    def _validate_stage_dependencies(
        self,
        stages: list[StageConfig],
        data_flow: DataFlowConfig,
    ) -> None:
        """Validate stage dependencies and data flow"""
        stage_names = {stage.name for stage in stages}
        alias_map: dict[str, str] = {}
        for stage in stages:
            alias_map.update(self._derive_dependency_aliases(stage))

        for stage in stages:
            # Check dependencies exist
            normalized_deps: list[str] = []
            for dep in stage.dependencies:
                if dep in stage_names:
                    normalized_deps.append(dep)
                    continue
                if dep in alias_map:
                    normalized_deps.append(alias_map[dep])
                    continue
                # Raise error for unknown stages
                raise ValueError(
                    f"Stage '{stage.name}' depends on unknown stage '{dep}'",
                )

            if isinstance(stage.dependencies, DependencyList):
                display_deps = stage.dependencies.display()
            else:
                display_deps = list(stage.dependencies)
            stage.dependencies = DependencyList(normalized_deps, display=display_deps)

            # Check input latent paths are valid
            if stage.input_latent_path:
                # This would be validated against actual file system
                # in a full implementation
                logging.warning(
                    f"Latent path validation not implemented for stage {stage.name}",  # IMPL
                )

    def _validate_positive_int(self, value: Any, field_name: str) -> int:
        """Validate that value is a positive integer"""
        try:
            int_value = int(value)
            if int_value <= 0:
                raise ValueError(f"{field_name} must be positive")
            return int_value
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a positive integer")

    def _validate_positive_float(self, value: Any, field_name: str) -> float:
        """Validate that value is a positive float"""
        try:
            float_value = float(value)
            if float_value <= 0:
                raise ValueError(f"{field_name} must be positive")
            return float_value
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a positive float")

    def save_normalized_config(
        self,
        config: MultiStageTrainingConfig,
        output_path: str,
    ) -> None:
        """Save normalized configuration to file"""
        config_dict = {
            "experiment_name": config.experiment_name,
            "mode": config.mode.value,
            "stages": [
                {
                    "name": stage.name,
                    "type": stage.type.value,
                    "model_type": stage.model_type,
                    "epochs": stage.epochs,
                    "batch_size": stage.batch_size,
                    "learning_rate": stage.learning_rate,
                    "optimizer": stage.optimizer,
                    "model_config": stage.model_config,
                    "data_config": stage.data_config,
                    "success_criteria": stage.success_criteria,
                    "save_latents": stage.save_latents,
                    "input_latent_path": stage.input_latent_path,
                    "output_latent_path": stage.output_latent_path,
                    "dependencies": stage.dependencies,
                    "conditional_logic": stage.conditional_logic,
                    "mode": stage.mode,
                    "encoder": stage.encoder,
                    "decoder": stage.decoder,
                    "outputs": ({"latents": stage.latent_exports} if stage.latent_exports else {}),
                    "inputs": stage.input_bindings,
                }
                for stage in config.stages
            ],
            "data_flow": {
                "input_transforms": config.data_flow.input_transforms,
                "output_transforms": config.data_flow.output_transforms,
                "latent_sharing": config.data_flow.latent_sharing,
                "cross_stage_validation": (config.data_flow.cross_stage_validation),
                "links": config.data_flow.links,
                "manifest_path": config.data_flow.manifest_path,
            },
            "global_config": config.global_config,
            "validation": config.validation_config,
            "monitoring": config.monitoring_config,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Normalized configuration saved to {output_path}")


def create_multi_stage_config_validator() -> MultiStageConfigValidator:
    """Factory function for multi-stage config validator"""
    return MultiStageConfigValidator()
