"""Core interface exports for the clean architecture layer.

This module re-exports the canonical model interfaces so callers can
import from ``core.interfaces`` without needing to know the internal
module layout (e.g., ``from core.interfaces import IGenerator``).
"""

from spectramr.models.interfaces.models import (
    IDiffusionModel,
    IDiscriminator,
    IGenerator,
    IModel,
    IModelFactory,
    ITransformer,
)

from .diffusion_process import IDiffusionProcess
from .i_downstream_model_service import IDownstreamModelService
from .i_privacy_accountant import IPrivacyAccountant

__all__ = [
    "IDiffusionModel",
    "IDiffusionProcess",
    "IDiscriminator",
    "IDownstreamModelService",
    "IGenerator",
    "IModel",
    "IModelFactory",
    "IPrivacyAccountant",
    "ITransformer",
]
