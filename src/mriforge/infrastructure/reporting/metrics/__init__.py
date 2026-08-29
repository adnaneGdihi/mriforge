r"""Reporting-side IQM library.

These are *image-level* metrics intended to be computed on
prediction/target NIfTI / Numpy arrays (post-inference) rather than as
training losses. Heavy reuse of the codebase's existing
``mriforge.core.metrics`` registry where possible.
"""

from .gmsd import gradient_magnitude_similarity_deviation
from .haarpsi import haar_psi
from .hallucination import feature_preservation_profile, hallucination_index
from .vif_fsim import feature_similarity_index, visual_information_fidelity

__all__ = [
    "feature_preservation_profile",
    "feature_similarity_index",
    "gradient_magnitude_similarity_deviation",
    "haar_psi",
    "hallucination_index",
    "visual_information_fidelity",
]
