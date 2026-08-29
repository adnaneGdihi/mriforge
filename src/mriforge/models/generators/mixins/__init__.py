"""Generator Mixins Package
=========================

Reusable mixins for generator architectures to reduce code duplication
and enforce composition over inheritance.

Compliance:
- soft_design.md: Composition over Inheritance
- perf_design.md: Shallow inheritance hierarchies
"""

from .diffusion import DiffusionMixin
from .physics import LatentMixin, PhysicsMixin

__all__ = [
    "DiffusionMixin",
    "LatentMixin",
    "PhysicsMixin",
]
