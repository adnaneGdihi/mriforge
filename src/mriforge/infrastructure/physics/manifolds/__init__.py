"""Riemannian-manifold module — Fisher-information geometry on Bloch parameters.

Consumed by Idea 5 (Riemannian Bloch diffusion). See ``base.py`` for the
``RiemannianManifold`` protocol and ``bloch_manifold.py`` for the
relaxation-manifold implementation.
"""

from .base import RiemannianManifold
from .bloch_manifold import BlochRelaxationManifold

__all__ = ["BlochRelaxationManifold", "RiemannianManifold"]
