"""Factories Package
================

This package contains all factory implementations following SOLID principles.
Factories provide dependency injection and loose coupling.
"""

from .model_factory import DIFFUSION_CONFIGS, ModelFactory, get_model_factory

__all__ = [
    "DIFFUSION_CONFIGS",
    "ModelFactory",
    "get_model_factory",
]
