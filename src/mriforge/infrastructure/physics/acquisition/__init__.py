"""Acquisition-side physics primitives.

Currently hosts the MRF pulse-sequence acquisition manifold used by
:class:`CRLBMRFPulseDesignStrategy` (audit_plan_novel_fmri.md MRF §4).
"""

from .mrf_acquisition_manifold import (
    AcquisitionBound,
    MRFAcquisitionManifold,
    beltrami_pulse_curve,
    sar_estimate,
)

__all__ = [
    "AcquisitionBound",
    "MRFAcquisitionManifold",
    "beltrami_pulse_curve",
    "sar_estimate",
]
