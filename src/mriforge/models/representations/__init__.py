"""3D Representations Package
==========================

This package provides various 3D representation systems for the MRIForge
project, including Structured LATent (SLAT) representations from TRELLIS.
"""

from .slat import (
    BaseRenderer,
    ISLATDecoder,
    ISLATEcoder,
    ISLATRepresentation,
    RendererOutput,
    SLATConfig,
    TrellisGaussianRenderer,
    TrellisMeshRenderer,
    VolumeRenderer,
)

__all__ = [
    # SLAT representations
    "ISLATRepresentation",
    "ISLATEcoder",
    "ISLATDecoder",
    "SLATConfig",
    # Renderers
    "RendererOutput",
    "BaseRenderer",
    "TrellisGaussianRenderer",
    "TrellisMeshRenderer",
    "VolumeRenderer",
]
