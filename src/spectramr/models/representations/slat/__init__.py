"""Structured LATent (SLAT) Representations Package
================================================

This package provides the TRELLIS Structured LATent representation system
for unified 3D generation and decoding to multiple output formats.
"""

from .interfaces import ISLATDecoder, ISLATEcoder, ISLATRepresentation, SLATConfig
from .renderers import (
    BaseRenderer,
    RendererOutput,
    TrellisGaussianRenderer,
    TrellisMeshRenderer,
    VolumeRenderer,
)

__all__ = [
    "ISLATDecoder",
    "ISLATEcoder",
    "ISLATRepresentation",
    "SLATConfig",
    # Renderers
    "RendererOutput",
    "BaseRenderer",
    "TrellisGaussianRenderer",
    "TrellisMeshRenderer",
    "VolumeRenderer",
]
