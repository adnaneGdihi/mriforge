"""Federated Learning configuration schema.

This module defines TrainingConfigFederated for federated learning paradigms.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .base import BaseTrainingConfigSchema

VALID_FED_AGGREGATIONS = (
    "fed_avg",
    "fed_prox",
    "fed_opt",
    "fed_adam",
    "scaffold",
    "fed_nova",
)


class FederatedConfig(BaseModel):
    """Federated Learning specific configuration."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    num_clients: int = Field(
        default=3,
        ge=2,
        description="Number of federated clients (must be >= 2 — single-client is not federated)",
    )
    local_epochs: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Number of local epochs per round",
    )
    aggregation_strategy: Literal[
        "fed_avg", "fed_prox", "fed_opt", "fed_adam", "scaffold", "fed_nova"
    ] = Field(
        default="fed_avg",
        description=f"Aggregation strategy. One of: {VALID_FED_AGGREGATIONS}",
    )
    client_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Fraction of clients to select per round (in (0, 1])",
    )
    num_rounds: int = Field(
        default=10,
        ge=1,
        le=100000,
        description="Number of federated rounds (must be >= 1)",
    )
    fed_prox_mu: float = Field(
        default=0.01,
        ge=0.0,
        description=(
            "Proximal regularization weight (FedProx only). Ignored for other "
            "aggregation strategies."
        ),
    )

    @field_validator("aggregation_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        if v not in VALID_FED_AGGREGATIONS:
            raise ValueError(
                f"aggregation_strategy must be one of {VALID_FED_AGGREGATIONS}, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _enforce_min_active_clients(self) -> "FederatedConfig":
        active = max(1, int(self.client_fraction * self.num_clients))
        if active < 2 and self.num_clients >= 2:
            raise ValueError(
                f"client_fraction={self.client_fraction} with num_clients={self.num_clients} "
                f"yields only {active} active client(s) per round. Set client_fraction "
                f">= {2 / self.num_clients:.3f} so at least 2 clients participate."
            )
        return self


class TrainingConfigFederated(BaseTrainingConfigSchema):
    """Federated Learning training configuration.

    Enforces federated-specific configurations.
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    federated_config: FederatedConfig = Field(
        default_factory=FederatedConfig,
        description="Federated learning configuration",
    )
