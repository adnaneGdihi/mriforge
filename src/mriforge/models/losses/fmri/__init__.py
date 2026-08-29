"""fMRI-specific losses (audit_plan_novel_fmri.md plan-faithful path).

The plan calls for losses to live under ``src/models/losses/fmri/``;
the actual implementations are shared with the MRF losses inside
:mod:`mriforge.models.losses.fmri_mrf_losses`, so this subpackage re-
exports them for the plan-faithful import path.
"""

from mriforge.models.losses.fmri_mrf_losses import (
    BeltramiEPIResidual as BeltramiEPI,
)
from mriforge.models.losses.fmri_mrf_losses import (
    HRFLikelihoodLoss as HRFLikelihood,
)

__all__ = ["BeltramiEPI", "HRFLikelihood"]
