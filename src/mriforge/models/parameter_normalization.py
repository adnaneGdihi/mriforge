"""Parameter Normalization and Validation System

This module provides a centralized system for normalizing parameter
names across different model types and validating parameter
configurations. It addresses the common issue where different
libraries and model implementations use different parameter names
for the same concepts (e.g., in_channels vs in_channels).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ParameterValidationError(Exception):
    """Raised when parameter validation fails."""


class ParameterNormalizationError(Exception):
    """Raised when parameter normalization fails."""


class ParameterCategory(Enum):
    """Categories of parameters for better organization."""

    CHANNELS = "channels"
    CLASSES = "classes"
    DIMENSIONS = "dimensions"
    ARCHITECTURE = "architecture"
    TRAINING = "training"
    ACTIVATION = "activation"
    OTHER = "other"


@dataclass
class ParameterMapping:
    """Defines a mapping from one parameter name to another."""

    source_name: str
    target_name: str
    category: ParameterCategory
    description: str = ""
    required: bool = False
    default_value: Any = None
    validator: Callable[[Any], bool] | None = None


@dataclass
class ModelParameterSchema:
    """Schema defining parameter mappings for a specific model type."""

    model_type: str
    mappings: list[ParameterMapping] = field(default_factory=list)
    required_params: set[str] = field(default_factory=set)
    optional_params: set[str] = field(default_factory=set)
    conflicting_params: list[tuple] = field(default_factory=list)
    # [(param1, param2), ...]


class ParameterNormalizer:
    """Central system for parameter normalization and validation.

    This class provides:
    - Parameter name mapping (e.g., in_channels -> in_channels)
    - Parameter validation with custom validators
    - Conflict detection between parameters
    - Schema-based normalization for different model types
    """

    def __init__(self):
        """__init__."""
        self._schemas: dict[str, ModelParameterSchema] = {}
        self._global_mappings: list[ParameterMapping] = []
        self._setup_default_schemas()

    def _setup_default_schemas(self):
        """Setup default parameter schemas for common model types."""
        self._setup_generator_schemas()
        self._setup_discriminator_schemas()
        self._setup_diffusion_schemas()

    def _setup_generator_schemas(self):
        """Setup schemas for generator models."""
        # Standard UNet-style generators
        standard_unet_schema = ModelParameterSchema(
            model_type="standard_unet",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map legacy in_channels to in_channels",
                    required=True,
                ),
                ParameterMapping(
                    "out_channels",
                    "out_channels",
                    ParameterCategory.CLASSES,
                    "Output classes parameter",
                    required=True,
                ),
            ],
            required_params={"in_channels", "out_channels"},
            optional_params={"output_activation"},
        )
        self._schemas["standard_unet"] = standard_unet_schema

        # Enhanced UNet generators
        enhanced_unet_schema = ModelParameterSchema(
            model_type="enhanced_unet",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map legacy in_channels to in_channels",
                    required=True,
                ),
                ParameterMapping(
                    "out_channels",
                    "out_channels",
                    ParameterCategory.CLASSES,
                    "Output classes parameter",
                    required=True,
                ),
            ],
            required_params={"in_channels", "out_channels"},
            optional_params={"output_activation", "attention_type"},
        )
        self._schemas["enhanced_unet"] = enhanced_unet_schema

        # EDSR-style generators
        edsr_schema = ModelParameterSchema(
            model_type="edsr",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map in_channels to in_channels for EDSR",
                    required=True,
                ),
                ParameterMapping(
                    "out_channels",
                    "out_channels",
                    ParameterCategory.CLASSES,
                    "Map out_channels to out_channels for EDSR",
                    required=True,
                ),
            ],
            required_params={"in_channels", "out_channels"},
            optional_params={"scale", "n_resblocks", "n_feats"},
        )
        self._schemas["edsr"] = edsr_schema

        # ViT-based generators
        vit_kan_schema = ModelParameterSchema(
            model_type="vit_kan",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map in_channels to in_channels for ViT",
                    required=True,
                ),
            ],
            required_params={"in_channels"},
            optional_params={"image_size", "patch_size", "dim", "depth", "heads"},
        )
        self._schemas["vit_kan"] = vit_kan_schema

        # Swin Transformer generators
        swin_schema = ModelParameterSchema(
            model_type="swin_transformer_kan",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "z_dim",
                    ParameterCategory.CHANNELS,
                    "Map in_channels to z_dim for Swin",
                    required=True,
                ),
            ],
            required_params={"z_dim"},
            optional_params={"dim", "depths", "heads", "window_size"},
        )
        self._schemas["swin_transformer_kan"] = swin_schema

        # Restormer generators
        restormer_schema = ModelParameterSchema(
            model_type="restormer",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map in_channels to in_channels",
                    required=True,
                ),
                ParameterMapping(
                    "out_channels",
                    "out_channels",
                    ParameterCategory.CLASSES,
                    "Map out_channels to out_channels",
                    required=True,
                ),
            ],
            required_params={"in_channels", "out_channels"},
            optional_params={
                "dim",
                "num_blocks",
                "num_refinement_blocks",
                "heads",
                "scale",
            },
        )
        self._schemas["restormer"] = restormer_schema

    def _setup_discriminator_schemas(self):
        """Setup schemas for discriminator models."""
        # Standard discriminators
        patchgan_schema = ModelParameterSchema(
            model_type="patchgan",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map in_channels to in_channels for discriminator",
                    required=True,
                ),
            ],
            required_params={"in_channels"},
            optional_params={"ndf", "n_layers"},
        )
        self._schemas["patchgan"] = patchgan_schema

    def _setup_diffusion_schemas(self):
        """Setup schemas for diffusion models."""
        # Diffusion generators
        diffusion_schema = ModelParameterSchema(
            model_type="score_based_unet",
            mappings=[],  # Diffusion models have different parameter structure
            required_params={"in_channels", "out_channels"},
            optional_params={"time_dim", "cond_dim"},
        )
        self._schemas["score_based_unet"] = diffusion_schema

        # TRELLIS generators
        trellis_gaussian_schema = ModelParameterSchema(
            model_type="trellis_gaussian",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map legacy in_channels to in_channels",
                    required=True,
                ),
                ParameterMapping(
                    "out_channels",
                    "out_channels",
                    ParameterCategory.CLASSES,
                    "Output classes parameter",
                    required=True,
                ),
                ParameterMapping(
                    "num_points",
                    "num_points",
                    ParameterCategory.DIMENSIONS,
                    "Number of 3D points to generate",
                    default_value=10000,
                ),
                ParameterMapping(
                    "resolution",
                    "resolution",
                    ParameterCategory.DIMENSIONS,
                    "Output resolution for 3D generation",
                    default_value=256,
                ),
            ],
            required_params={"in_channels", "out_channels"},
            optional_params={"num_points", "resolution", "guidance_scale"},
        )
        self._schemas["trellis_gaussian"] = trellis_gaussian_schema

        trellis_mesh_schema = ModelParameterSchema(
            model_type="trellis_mesh",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map legacy in_channels to in_channels",
                    required=True,
                ),
                ParameterMapping(
                    "out_channels",
                    "out_channels",
                    ParameterCategory.CLASSES,
                    "Output classes parameter",
                    required=True,
                ),
                ParameterMapping(
                    "num_faces",
                    "num_faces",
                    ParameterCategory.DIMENSIONS,
                    "Number of mesh faces to generate",
                    default_value=5000,
                ),
                ParameterMapping(
                    "resolution",
                    "resolution",
                    ParameterCategory.DIMENSIONS,
                    "Output resolution for mesh generation",
                    default_value=256,
                ),
            ],
            required_params={"in_channels", "out_channels"},
            optional_params={"num_faces", "resolution", "simplify_mesh"},
        )
        self._schemas["trellis_mesh"] = trellis_mesh_schema

        # X-Diffusion generators
        x_diffusion_latent_schema = ModelParameterSchema(
            model_type="x_diffusion_latent",
            mappings=[
                ParameterMapping(
                    "in_channels",
                    "in_channels",
                    ParameterCategory.CHANNELS,
                    "Map legacy in_channels to in_channels",
                    required=True,
                ),
                ParameterMapping(
                    "out_channels",
                    "out_channels",
                    ParameterCategory.CLASSES,
                    "Output classes parameter",
                    required=True,
                ),
                ParameterMapping(
                    "latent_dim",
                    "latent_dim",
                    ParameterCategory.DIMENSIONS,
                    "Latent space dimension",
                    default_value=512,
                ),
                ParameterMapping(
                    "num_slices",
                    "num_slices",
                    ParameterCategory.DIMENSIONS,
                    "Number of slices for 3D reconstruction",
                    default_value=32,
                ),
                ParameterMapping(
                    "timesteps",
                    "timesteps",
                    ParameterCategory.TRAINING,
                    "Number of diffusion timesteps",
                    default_value=1000,
                ),
                ParameterMapping(
                    "beta_schedule",
                    "beta_schedule",
                    ParameterCategory.TRAINING,
                    "Diffusion beta schedule type",
                    default_value="linear",
                ),
            ],
            required_params={"in_channels", "out_channels"},
            optional_params={
                "latent_dim",
                "num_slices",
                "timesteps",
                "beta_schedule",
                "guidance_scale",
            },
        )
        self._schemas["x_diffusion_latent"] = x_diffusion_latent_schema

    def register_schema(self, schema: ModelParameterSchema):
        """Register a parameter schema for a model type."""
        self._schemas[schema.model_type] = schema
        logger.debug(f"Registered parameter schema for {schema.model_type}")

    def add_global_mapping(self, mapping: ParameterMapping):
        """Add a global parameter mapping that applies to all models."""
        self._global_mappings.append(mapping)

    def normalize_parameters(
        self,
        model_type: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize parameters for a given model type.

        Args:
            model_type: The type of model
            kwargs: Original parameter dictionary

        Returns:
            Normalized parameter dictionary

        Raises:
            ParameterNormalizationError: If normalization fails

        """
        normalized_kwargs = kwargs.copy()

        # Apply global mappings first
        for mapping in self._global_mappings:
            if (
                mapping.source_name in normalized_kwargs
                and mapping.target_name not in normalized_kwargs
            ):
                normalized_kwargs[mapping.target_name] = normalized_kwargs.pop(
                    mapping.source_name,
                )
                logger.debug(
                    f"Applied global mapping: {mapping.source_name} -> {mapping.target_name}",
                )

        # Apply model-specific mappings
        if model_type in self._schemas:
            schema = self._schemas[model_type]
            for mapping in schema.mappings:
                if (
                    mapping.source_name in normalized_kwargs
                    and mapping.target_name not in normalized_kwargs
                ):
                    normalized_kwargs[mapping.target_name] = normalized_kwargs.pop(
                        mapping.source_name,
                    )
                    logger.debug(
                        f"Applied {model_type} mapping: "
                        f"{mapping.source_name} -> {mapping.target_name}",
                    )

        return normalized_kwargs

    def validate_parameters(self, model_type: str, kwargs: dict[str, Any]) -> list[str]:
        """Validate parameters for a given model type.

        Args:
            model_type: The type of model
            kwargs: Parameter dictionary to validate

        Returns:
            list of validation error messages (empty if valid)

        """
        errors = []

        if model_type not in self._schemas:
            errors.append(f"No parameter schema found for model type: {model_type}")
            return errors

        schema = self._schemas[model_type]

        # Check required parameters
        for param in schema.required_params:
            if param not in kwargs:
                errors.append(f"Missing required parameter: {param}")

        # Check conflicting parameters
        for param1, param2 in schema.conflicting_params:
            if param1 in kwargs and param2 in kwargs:
                errors.append(
                    f"Conflicting parameters: {param1} and {param2} cannot both be specified",
                )

        # Apply custom validators
        for mapping in schema.mappings:
            if mapping.validator and mapping.target_name in kwargs:
                if not mapping.validator(kwargs[mapping.target_name]):
                    errors.append(f"Parameter {mapping.target_name} failed validation")

        return errors

    def get_available_schemas(self) -> list[str]:
        """Get list of available model type schemas."""
        return list(self._schemas.keys())

    def get_schema(self, model_type: str) -> ModelParameterSchema | None:
        """Get the parameter schema for a model type."""
        return self._schemas.get(model_type)

    def suggest_mappings(
        self,
        model_type: str,
        available_params: list[str],
    ) -> dict[str, str]:
        """Suggest parameter mappings based on available parameters.

        Args:
            model_type: The type of model
            available_params: list of available parameter names

        Returns:
            Dictionary mapping available params to target params

        """
        suggestions = {}

        if model_type not in self._schemas:
            return suggestions

        schema = self._schemas[model_type]

        # Check global mappings
        for mapping in self._global_mappings:
            if mapping.source_name in available_params:
                suggestions[mapping.source_name] = mapping.target_name

        # Check model-specific mappings
        for mapping in schema.mappings:
            if mapping.source_name in available_params:
                suggestions[mapping.source_name] = mapping.target_name

        return suggestions


# Global normalizer instance
_parameter_normalizer = None


def get_parameter_normalizer() -> ParameterNormalizer:
    """Get the global parameter normalizer instance."""
    global _parameter_normalizer
    if _parameter_normalizer is None:
        _parameter_normalizer = ParameterNormalizer()
    return _parameter_normalizer


def normalize_model_parameters(
    model_type: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Convenience function to normalize parameters for a model type.

    Args:
        model_type: The type of model
        kwargs: Original parameter dictionary

    Returns:
        Normalized parameter dictionary

    """
    normalizer = get_parameter_normalizer()
    return normalizer.normalize_parameters(model_type, kwargs)


def validate_model_parameters(model_type: str, kwargs: dict[str, Any]) -> list[str]:
    """Convenience function to validate parameters for a model type.

    Args:
        model_type: The type of model
        kwargs: Parameter dictionary to validate

    Returns:
        list of validation error messages (empty if valid)

    """
    normalizer = get_parameter_normalizer()
    return normalizer.validate_parameters(model_type, kwargs)


def register_parameter_schema(schema: ModelParameterSchema):
    """Register a parameter schema for a model type."""
    normalizer = get_parameter_normalizer()
    normalizer.register_schema(schema)


def add_global_parameter_mapping(mapping: ParameterMapping):
    """Add a global parameter mapping."""
    normalizer = get_parameter_normalizer()
    normalizer.add_global_mapping(mapping)
