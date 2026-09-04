"""Diffusion-specific training configuration schema.

This module defines TrainingConfigDiffusion for diffusion-based training paradigms.
It enforces diffusion objectives and validates diffusion-specific fields.
"""

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ..base import (
    DiffusionObjectiveConfig,
    LatentObjectiveConfig,
    ReconstructionObjectiveConfig,
)
from ..enums import NoiseSchedule, PredictionType
from .base import BaseTrainingConfigSchema

VALID_TIMESTEP_SAMPLERS = (
    "uniform",
    "importance",
    "linear_decay",
    "cosine_warm",
    "high_t_emphasis",
    "balanced_high_t",
)
VALID_INFERENCE_SAMPLERS = (
    "ddpm",
    "ddim",
    "dpm_solver",
    "dpm_solver_pp",
    "pndm",
    "euler",
    "euler_ancestral",
    "heun",
    "cold_mri",
    # SOTA plan step 2: Decomposed Diffusion Sampling (CG data-consistency).
    # Resolves to DDSReconSampler via the SamplerRegistry; requires a
    # score/diffusion model (enforced by the dds_requires_score_model audit).
    "dds",
    # SOTA plan Phase C: Dynamic Diffusion Posterior Sampling (3 dynamic-guidance
    # variants — adaptive_step / pigdm / ncchi — via DynamicDPSSampler). Also a
    # score-model posterior sampler (same audit guard).
    "dynamic_dps",
)


class DiffusionObjectiveConfigSchema(BaseModel):
    """Diffusion-specific objectives configuration.

    Only includes objectives relevant to diffusion training.
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    # Only include relevant objectives for diffusion
    reconstruction: ReconstructionObjectiveConfig = Field(
        default_factory=ReconstructionObjectiveConfig
    )
    diffusion: DiffusionObjectiveConfig = Field(default_factory=DiffusionObjectiveConfig)
    latent: LatentObjectiveConfig = Field(default_factory=LatentObjectiveConfig)


class TrainingConfigDiffusion(BaseTrainingConfigSchema):
    """Diffusion model training configuration.

    Enforces diffusion objectives and validates that:
    - Diffusion objective configuration is present
    - Timesteps are specified
    - Noise schedule is valid
    - Prediction type is valid

    Example:
        >>> config = TrainingConfigDiffusion(
        ...     training_mode="diffusion",
        ...     epochs=100,
        ...     objectives=DiffusionObjectiveConfigSchema(
        ...         diffusion=DiffusionObjectiveConfig(
        ...             timesteps=16,
        ...             noise_schedule="cosine"
        ...         )
        ...     ),
        ... )
    """

    # Override with Diffusion-specific objectives (no GAN)
    objectives: DiffusionObjectiveConfigSchema = Field(
        default_factory=DiffusionObjectiveConfigSchema,
        description="Training objectives (Diffusion must have objectives.diffusion configured)",
    )

    # Diffusion-specific hyperparameters (SSOT in training schema)
    timesteps: int = Field(
        default=1000,
        ge=1,
        description="Number of diffusion timesteps",
    )
    noise_schedule: NoiseSchedule = Field(
        default=NoiseSchedule.COSINE,
        description="Type of noise schedule: linear, cosine, sqrt, quadratic",
    )
    beta_start: float = Field(
        default=0.0001,
        ge=0,
        le=1,
        description="Starting value for noise schedule",
    )
    beta_end: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="Ending value for noise schedule",
    )

    # Diffusion-specific features
    enable_classifier_free_guidance: bool = Field(
        default=True,
        description="Enable classifier-free guidance during training",
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        description="Scale factor for guidance (SSOT in training schema)",
    )
    guidance_prob: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Probability of dropping guidance during training",
    )
    prediction_type: PredictionType = Field(
        default=PredictionType.EPSILON,
        description="Prediction target type: epsilon (noise), sample, v_prediction",
    )
    enable_diffusion_amp: bool = Field(
        default=False,
        description="Enable automatic mixed precision for diffusion",
    )
    enforce_output_range: bool = Field(
        default=True,
        description="Enforce output to [-1, 1] range",
    )
    enforce_zero_terminal_snr: bool = Field(
        default=False,
        description="Use zero terminal SNR in noise schedule",
    )

    # [STABILIZATION FIX 3.3] Curriculum Learning
    curriculum_start_timestep: int = Field(
        default=50,
        ge=1,
        description="Starting maximum timestep for curriculum (light degradation).",
    )
    curriculum_ramp_rate: float = Field(
        default=0.01,
        ge=0.0,
        description="Timesteps to add per iteration (e.g. 0.01 means 10 steps per 1000 iterations).",
    )
    sampling_steps: int = Field(
        default=50,
        ge=1,
        le=10000,
        description="Number of reverse diffusion steps for inference/validation (DDIM/Cold).",
    )

    # Timestep sampling strategy (addresses 2× acceleration regression)
    timestep_sampling_strategy: Literal[
        "uniform",
        "importance",
        "linear_decay",
        "cosine_warm",
        "high_t_emphasis",
        "balanced_high_t",
    ] = Field(
        default="uniform",
        description=(
            "Strategy for sampling timesteps during training. "
            "'uniform': Equal probability across [1, T_max] (default). "
            "'importance': Quadratic bias toward smaller timesteps — "
            "preserves low-acceleration performance during multi-scale training. "
            "'linear_decay': Linearly decreasing probability with timestep. "
            "'cosine_warm': Cosine-warm-up schedule that starts at small timesteps. "
            "'high_t_emphasis': Linearly INCREASING probability P(t) ∝ t — the "
            "mirror of 'linear_decay'. Over-samples the hard high-acceleration "
            "regime so a shared timestep-conditioned net does not abandon high t "
            "to the easy low-t task (experiment_11 R8x/R32x negative transfer). "
            "'balanced_high_t': 'high_t_emphasis' mixed with a uniform floor "
            "(P(t) ∝ (1-eps)*t/sum(t) + eps/N, eps=0.3) — keeps high-t gradient "
            "priority WITHOUT starving low t. Fixes the over-correction where pure "
            "high_t_emphasis abandoned low t (experiment_11 R2x val collapse, the "
            "accel-inversion where R2x validation reverse-samples an untrained band)."
        ),
    )

    # Inference sampler selector — explicit enum prevents typo silent-fallbacks.
    inference_sampler: Literal[
        "ddpm",
        "ddim",
        "dpm_solver",
        "dpm_solver_pp",
        "pndm",
        "euler",
        "euler_ancestral",
        "heun",
        "cold_mri",
        "dds",
        "dynamic_dps",
    ] = Field(
        default="ddim",
        description=(
            "Reverse-process sampler used for validation / inference. "
            "'ddim' (default) is fast and deterministic. 'ddpm' is the original "
            "stochastic schedule. 'dpm_solver_pp' converges in ~10 steps. "
            "'cold_mri' is for cold-diffusion paradigms (deterministic degradation)."
        ),
    )

    @field_validator("timestep_sampling_strategy")
    @classmethod
    def _validate_timestep_sampler(cls, v: str) -> str:
        if v not in VALID_TIMESTEP_SAMPLERS:
            raise ValueError(
                f"timestep_sampling_strategy must be one of {VALID_TIMESTEP_SAMPLERS}, got {v!r}"
            )
        return v

    @field_validator("inference_sampler")
    @classmethod
    def _validate_inference_sampler(cls, v: str) -> str:
        if v not in VALID_INFERENCE_SAMPLERS:
            raise ValueError(
                f"inference_sampler must be one of {VALID_INFERENCE_SAMPLERS}, got {v!r}"
            )
        return v

    # Data consistency
    enable_data_consistency: bool = Field(
        default=False,
        description="Enable physics-based data consistency",
    )
    dc_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight for data consistency loss",
    )

    # [FIX] Allow custom diffusion configuration (e.g. cold diffusion degradation)
    diffusion: dict = Field(
        default_factory=dict,
        description="Custom diffusion parameters (degradation, etc.)",
    )

    # Text Conditioning (DiffBoost)
    text_conditioning: dict = Field(
        default_factory=lambda: {
            "enabled": False,
            "model_id": "openai/clip-vit-large-patch14",
        },
        description="Text conditioning configuration (enabled, model_id)",
    )

    @field_validator("objectives")
    @classmethod
    def validate_diffusion_objectives(
        cls, obj: DiffusionObjectiveConfigSchema
    ) -> DiffusionObjectiveConfigSchema:
        """Ensure diffusion objectives are configured.

        NOTE: timesteps validation is now in losses.diffusion (SSOT).

        Args:
            obj: DiffusionObjectiveConfigSchema to validate

        Returns:
            Validated DiffusionObjectiveConfigSchema

        Raises:
            ValueError: If diffusion objectives are missing or invalid
        """
        if obj.diffusion is None:
            raise ValueError("Diffusion training requires objectives.diffusion configuration")
        return obj

    @field_validator("beta_end")
    @classmethod
    def validate_beta_range(cls, v: float, info: ValidationInfo) -> float:
        """Ensure beta_end > beta_start."""
        if "beta_start" in info.data and v <= info.data["beta_start"]:
            raise ValueError("beta_end must be greater than beta_start")
        return v

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }


__all__ = ["DiffusionObjectiveConfigSchema", "TrainingConfigDiffusion"]
