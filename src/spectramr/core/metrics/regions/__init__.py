"""Region-restricted metric scoring: regions, the eligibility gate, the scorer.

A global full-reference metric is an **area-weighted average**. On a 320x320
fastMRI brain slice, background/air is ~40% of the pixels and a typical fastMRI+
pathology bbox is ~40x40 px -- about **1.5% of the image**. A metric can therefore
post a near-perfect global severity-tracking score while being completely blind to
whether the diagnostically relevant 1.5% survived.

This package makes the scoring region explicit, and refuses to score a metric on a
region where that metric is not defined.

Entry point: :func:`~spectramr.core.metrics.regions.scorer.score_region`.
Read :mod:`~spectramr.core.metrics.regions.eligibility` first -- it is the gate.
"""

from spectramr.core.metrics.regions.crop import crop_to_region
from spectramr.core.metrics.regions.eligibility import (
    ROI_POLICY,
    Eligibility,
    Normalisation,
    RoiPolicy,
    RoiSupport,
    evaluate_eligibility,
    policy_for,
)
from spectramr.core.metrics.regions.reductions import (
    MAP_REDUCERS,
    MapReducer,
    masked_mean,
)
from spectramr.core.metrics.regions.scorer import (
    eligibility_table,
    score_region,
    score_region_set,
)
from spectramr.core.metrics.regions.types import (
    FULL_REGION_ID,
    RegionMask,
    RegionSet,
    RegionSource,
)

__all__ = [
    "FULL_REGION_ID",
    "MAP_REDUCERS",
    "ROI_POLICY",
    "Eligibility",
    "MapReducer",
    "Normalisation",
    "RegionMask",
    "RegionSet",
    "RegionSource",
    "RoiPolicy",
    "RoiSupport",
    "crop_to_region",
    "eligibility_table",
    "evaluate_eligibility",
    "masked_mean",
    "policy_for",
    "score_region",
    "score_region_set",
]
