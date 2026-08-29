"""TRELLIS-specific 3D generation model configuration schema.

This module contains configuration for TRELLIS (TRiplane Equivariance Learned
from LInearly Scaled transformers) models used for multi-view 3D generation.

TRELLIS models use triplane representations (Gaussian or mesh) for efficient
3D volumetric generation from 2D input images.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TrellisConfigSchema(BaseModel):
    """Configuration parameters for TRELLIS-based generators.

    TRELLIS uses a triplane representation for efficient 3D generation.

    Example:
        >>> config = TrellisConfigSchema(
        ...     representation="gaussian",
        ...     input_resolution=(224, 224),
        ...     output_resolution=(64, 128, 128),
        ... )
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    # Core representation
    representation: str = Field(
        default="gaussian",
        description="Triplane representation type: gaussian or mesh",
    )
    input_resolution: tuple[int, int] = Field(
        default=(224, 224),
        description="Input image resolution (H, W)",
    )
    output_resolution: tuple[int, int, int] = Field(
        default=(64, 128, 128),
        description="Output volume resolution (D, H, W)",
    )
    coarse_resolution: tuple[int, int, int] | None = Field(
        default=None,
        description="Coarse resolution for hierarchical generation",
    )

    # Transformer architecture
    feature_dim: int = Field(
        default=256,
        gt=0,
        description="Feature dimension",
    )
    num_views: int = Field(
        default=4,
        gt=0,
        description="Number of views in triplane",
    )
    num_layers: int = Field(
        default=12,
        gt=0,
        description="Number of transformer layers",
    )
    num_heads: int = Field(
        default=8,
        gt=0,
        description="Number of attention heads",
    )
    mlp_ratio: float = Field(
        default=4.0,
        gt=0,
        description="MLP hidden dimension ratio",
    )

    # Generation stages
    num_stages: int = Field(
        default=3,
        gt=0,
        description="Number of coarse-to-fine generation stages",
    )
    stride: int = Field(
        default=8,
        gt=0,
        description="Patch stride for tokenization",
    )
    patch_size: int = Field(
        default=8,
        gt=0,
        description="Patch size for tokenization",
    )

    # Gaussian representation parameters
    gaussian_num_gaussians: int = Field(
        default=32768,
        gt=0,
        description="Number of Gaussians in representation",
    )
    gaussian_resolution: int = Field(
        default=32,
        gt=0,
        description="Resolution of Gaussian splatting grid",
    )
    gaussian_latent_dim: int = Field(
        default=64,
        gt=0,
        description="Latent dimension for Gaussian features",
    )

    # Mesh representation parameters
    mesh_num_vertices: int = Field(
        default=50000,
        gt=0,
        description="Number of vertices in mesh",
    )
    mesh_num_faces: int = Field(
        default=100000,
        gt=0,
        description="Number of faces in mesh",
    )
    mesh_latent_dim: int = Field(
        default=64,
        gt=0,
        description="Latent dimension for mesh features",
    )

    @field_validator("representation")
    @classmethod
    def validate_representation(cls, value: str) -> str:
        """Validate representation type."""
        valid = {"gaussian", "mesh"}
        if value not in valid:
            raise ValueError(f"representation must be one of {valid}")
        return value

    @field_validator("input_resolution")
    @classmethod
    def validate_input_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Validate input resolution tuple."""
        if len(value) != 2:
            raise ValueError("input_resolution must contain two integers")
        if any(v <= 0 for v in value):
            raise ValueError("input_resolution values must be positive")
        return value

    @field_validator("output_resolution")
    @classmethod
    def validate_output_resolution(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate output resolution tuple."""
        if len(value) != 3:
            raise ValueError("output_resolution must contain three integers")
        if any(v <= 0 for v in value):
            raise ValueError("output_resolution values must be positive")
        return value


class TrellisEncoderConfigSchema(BaseModel):
    """Encoder component configuration for TRELLIS VAEs.

    Encodes 2D images to latent tokens for feeding to TRELLIS decoder.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    name: str = Field(
        default="ElasticSLatEncoder",
        description="Encoder name/class",
    )
    resolution: int = Field(
        default=64,
        gt=0,
        description="Feature resolution",
    )
    in_channels: int = Field(
        default=1024,
        gt=0,
        description="Input channels from image",
    )
    model_channels: int = Field(
        default=768,
        gt=0,
        description="Model feature dimension",
    )
    latent_channels: int = Field(
        default=8,
        gt=0,
        description="Latent code dimension",
    )
    num_blocks: int = Field(
        default=12,
        gt=0,
        description="Number of transformer blocks",
    )
    num_heads: int = Field(
        default=12,
        gt=0,
        description="Number of attention heads",
    )
    mlp_ratio: float = Field(
        default=4.0,
        gt=0,
        description="MLP hidden dimension ratio",
    )
    attn_mode: str = Field(
        default="swin",
        description="Attention mechanism: swin or flash",
    )
    window_size: int = Field(
        default=8,
        gt=0,
        description="Swin attention window size",
    )
    use_fp16: bool = Field(
        default=False,
        description="Use float16 precision",
    )


class TrellisDecoderConfigSchema(BaseModel):
    """Decoder component configuration for TRELLIS VAEs.

    Decodes latent codes to 3D triplane representation.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    name: str = Field(
        default="ElasticSLatGaussianDecoder",
        description="Decoder name/class",
    )
    resolution: int = Field(
        default=64,
        gt=0,
        description="Feature resolution",
    )
    model_channels: int = Field(
        default=768,
        gt=0,
        description="Model feature dimension",
    )
    latent_channels: int = Field(
        default=8,
        gt=0,
        description="Latent code dimension",
    )
    num_blocks: int = Field(
        default=12,
        gt=0,
        description="Number of transformer blocks",
    )
    num_heads: int = Field(
        default=12,
        gt=0,
        description="Number of attention heads",
    )
    mlp_ratio: float = Field(
        default=4.0,
        gt=0,
        description="MLP hidden dimension ratio",
    )
    attn_mode: str = Field(
        default="swin",
        description="Attention mechanism: swin or flash",
    )
    window_size: int = Field(
        default=8,
        gt=0,
        description="Swin attention window size",
    )
    use_fp16: bool = Field(
        default=False,
        description="Use float16 precision",
    )
    output_channels: int = Field(
        default=1,
        gt=0,
        description="Output channels (typically 1 for volume)",
    )
    output_resolution: tuple[int, int, int] = Field(
        default=(64, 64, 64),
        description="Output volume resolution (D, H, W)",
    )
    representation_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Representation-specific configuration",
    )

    @field_validator("output_resolution")
    @classmethod
    def validate_output_resolution(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate output resolution tuple."""
        if len(value) != 3:
            raise ValueError("output_resolution must contain three integers")
        if any(v <= 0 for v in value):
            raise ValueError("output_resolution values must be positive")
        return value


class TrellisTrainerConfigSchema(BaseModel):
    """Training knobs for TRELLIS VAE models.

    Inspired by TRELLIS reference configurations.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    # Training steps and batching
    max_steps: int = Field(
        default=100000,
        gt=0,
        description="Maximum training steps",
    )
    batch_size_per_gpu: int = Field(
        default=1,
        gt=0,
        description="Batch size per GPU",
    )
    batch_split: int = Field(
        default=1,
        ge=1,
        description="Number of gradient accumulation steps",
    )

    # Optimizer
    optimizer: dict[str, Any] = Field(
        default_factory=lambda: {
            "name": "AdamW",
            "args": {"lr": 1e-4, "weight_decay": 0.0},
        },
        description="Optimizer configuration",
    )

    # EMA
    ema_rate: list[float] = Field(
        default_factory=lambda: [0.999],
        description="EMA decay rates",
    )

    # Precision
    fp16_mode: str = Field(
        default="off",
        description="Float16 mode: off, on, or bf16",
    )
    fp16_scale_growth: float = Field(
        default=0.001,
        ge=0,
        description="FP16 loss scale growth factor",
    )

    # Training dynamics
    elastic: dict[str, Any] = Field(
        default_factory=dict,
        description="Elastic/progressive training settings",
    )
    grad_clip: dict[str, Any] = Field(
        default_factory=dict,
        description="Gradient clipping configuration",
    )

    # Logging and sampling
    i_log: int = Field(
        default=500,
        gt=0,
        description="Log metrics every N steps",
    )
    i_sample: int = Field(
        default=5000,
        gt=0,
        description="Sample/validate every N steps",
    )
    i_save: int = Field(
        default=10000,
        gt=0,
        description="Save checkpoint every N steps",
    )

    # Loss configuration
    loss_type: str = Field(
        default="l1",
        description="Reconstruction loss type: l1, l2, mse, mae",
    )
    lambda_ssim: float = Field(
        default=0.0,
        ge=0,
        description="SSIM loss weight",
    )
    lambda_lpips: float = Field(
        default=0.0,
        ge=0,
        description="LPIPS loss weight",
    )
    lambda_kl: float = Field(
        default=1e-4,
        ge=0,
        description="KL divergence loss weight (for VAE)",
    )

    # Regularization
    regularizations: dict[str, float] = Field(
        default_factory=dict,
        description="Additional regularization terms",
    )


class TrellisVAEConfigSchema(BaseModel):
    """Top-level configuration for TRELLIS Structured Latent VAEs.

    Combines encoder, decoder, and trainer configurations.

    Example:
        >>> config = TrellisVAEConfigSchema(
        ...     encoder=TrellisEncoderConfigSchema(),
        ...     decoder=TrellisDecoderConfigSchema(),
        ...     trainer=TrellisTrainerConfigSchema(),
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    encoder: TrellisEncoderConfigSchema = Field(
        default_factory=TrellisEncoderConfigSchema,
        description="Encoder configuration",
    )
    decoder: TrellisDecoderConfigSchema = Field(
        default_factory=TrellisDecoderConfigSchema,
        description="Decoder configuration",
    )
    trainer: TrellisTrainerConfigSchema = Field(
        default_factory=TrellisTrainerConfigSchema,
        description="Trainer configuration",
    )


__all__ = [
    "TrellisConfigSchema",
    "TrellisDecoderConfigSchema",
    "TrellisEncoderConfigSchema",
    "TrellisTrainerConfigSchema",
    "TrellisVAEConfigSchema",
]
