"""Model Interfaces Package
=======================

This package exposes the contracts that concrete model implementations follow.
The interfaces reinforce SOLID principles by encouraging loose coupling and
high cohesion across the codebase.
"""

from .latent_access import LatentAccessMixin, SupportsLatentAccess
from .models import (
    IConfigurable,
    IDiffusionModel,
    IDiscriminator,
    IGenerator,
    IModel,
    IModelFactory,
    ITransformer,
)

__all__ = [
    "IConfigurable",
    "IDiffusionModel",
    "IDiscriminator",
    "IGenerator",
    "IModel",
    "IModelFactory",
    "ITransformer",
    "LatentAccessMixin",
    "SupportsLatentAccess",
]
