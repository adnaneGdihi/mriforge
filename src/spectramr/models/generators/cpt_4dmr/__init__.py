"""CPT-4DMR: Continuous Spatio-Temporal 4D MRI Reconstruction
============================================================

This module implements continuous-time 4D MRI reconstruction using
Implicit Neural Representations (INRs) that decouple anatomy from motion.

Architecture:
- Spatial Anatomy Network (SAN): SIREN-based canonical volume
- Temporal Motion Network (TMN): DVF conditioned on time/respiratory signal

Key equation: I(x, t) = SAN(x + TMN(x, t, resp_signal))
"""

from .cpt_4dmr_generator import CPT4DMRGenerator
from .spatial_anatomy_net import SIREN, SpatialAnatomyNet
from .temporal_motion_net import TemporalMotionNet

__all__ = [
    "SIREN",
    "CPT4DMRGenerator",
    "SpatialAnatomyNet",
    "TemporalMotionNet",
]
