"""External-sensor adapters and Cramér-Rao-optimal fusion (idea 9).

Provides per-modality 6-DOF pose adapters (RGB-D, IMU, ECG,
respiratory) and a single fusion routine that combines their
estimates with inverse-variance weights — the maximum-likelihood
form for Gaussian observation noise, equivalent to the Cramér-Rao
lower bound on fused pose variance.

See ``TODO/integration_plan_ulf_cheap_fast_mri.md`` §9.
"""

from .fusion import (
    FusedPose,
    PoseEstimate,
    cramer_rao_optimal_fusion,
    inverse_variance_weights,
)

__all__ = [
    "FusedPose",
    "PoseEstimate",
    "cramer_rao_optimal_fusion",
    "inverse_variance_weights",
]
