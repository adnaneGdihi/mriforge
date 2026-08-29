"""Adaptive conditioning & embedding package (2026-05-26).

A single conditioning seam for the VF / digital-twin pipeline: a canonical
:class:`ConditioningContext` bundling every condition source, a registry of
encoders that turn those sources into embeddings, a FiLM injector that
modulates feature maps, and a Koopman temporal encoder for periodic motion.
Lets an experiment declare — in YAML — which conditions the network should
adapt to (degradation severity, diffusion time, acquisition order/params,
contrast, motion pose).

Importing the package populates the conditioner registry (encoder + Koopman
modules register on import).
"""

from __future__ import annotations

from mriforge.models.conditioning.context import ConditioningContext
from mriforge.models.conditioning.encoders import (
    AcquisitionParamEncoder,
    AcquisitionTimeEncoder,
    ConditionEncoder,
    ConditioningEncoderBank,
    ContrastEncoder,
    FieldStrengthEncoder,
    MotionPoseEncoder,
    ScannerEncoder,
    SeverityTokenEncoder,
    SinusoidalTimeEncoder,
    SiteEncoder,
    get_conditioner,
    register_conditioner,
)
from mriforge.models.conditioning.injection import (
    AdaptiveConditioner,
    ConditioningInjector,
)
from mriforge.models.conditioning.koopman_motion import KoopmanMotionEncoder

__all__ = [
    "AcquisitionParamEncoder",
    "AcquisitionTimeEncoder",
    "AdaptiveConditioner",
    "ConditionEncoder",
    "ConditioningContext",
    "ConditioningEncoderBank",
    "ConditioningInjector",
    "ContrastEncoder",
    "FieldStrengthEncoder",
    "KoopmanMotionEncoder",
    "MotionPoseEncoder",
    "ScannerEncoder",
    "SeverityTokenEncoder",
    "SinusoidalTimeEncoder",
    "SiteEncoder",
    "get_conditioner",
    "register_conditioner",
]
