from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .base import BaseTrainingConfigSchema

VALID_FLOW_SAMPLERS = ("euler", "midpoint", "rk45", "heun")


class FlowConfig(BaseModel):
    """Flow matching configuration."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    sampling_method: Literal["euler", "midpoint", "rk45", "heun"] = Field(
        default="euler",
        description=(
            "ODE solver method for flow matching inference. "
            "'euler' is fastest; 'rk45' is most accurate; 'heun' is a 2nd-order compromise."
        ),
    )
    num_inference_steps: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Number of solver steps (1 for one-step rectified flow, 50-100 typical for high-quality)",
    )
    reflow_iterations: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of reflow iterations (0 disables; 1-3 typical for InstaFlow)",
    )
    guidance_scale: float = Field(
        default=1.0,
        ge=0.0,
        description="Classifier-free guidance scale (1.0 disables CFG; >1.0 amplifies condition).",
    )

    @field_validator("sampling_method")
    @classmethod
    def _validate_sampler(cls, v: str) -> str:
        if v not in VALID_FLOW_SAMPLERS:
            raise ValueError(f"sampling_method must be one of {VALID_FLOW_SAMPLERS}, got {v!r}")
        return v


class TrainingConfigFlow(BaseTrainingConfigSchema):
    """Flow Matching training configuration."""

    flow: FlowConfig = Field(
        default_factory=FlowConfig,
        description="Flow matching specific configuration",
    )

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }
