"""Test-Time Adaptation (SoTa-DiT).

This package implements adaptation strategies for diffusion models at inference time.
Key components:
- Norm Adaptation: Fine-tuning normalization statistics or parameters.
- Data Consistency Loss: Guiding adaptation using measurement physics.
"""

from .sota_adapter import SoTaAdapter
from .transport import KidotTransport, WassersteinDiscriminator

__all__ = ["KidotTransport", "SoTaAdapter", "WassersteinDiscriminator"]
