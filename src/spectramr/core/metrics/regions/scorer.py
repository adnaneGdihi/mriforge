"""The region scorer: the one place the eligibility gate is enforced.

Every region-restricted metric value in the system comes through
:func:`score_region`. That is deliberate -- a gate with two doors is not a gate.

The contract:

* **Eligible** -> the metric runs on the restricted data and returns a number.
* **Ineligible** -> ``MetricOutcome.not_applicable`` with the declared reason, and
  **the metric object is never invoked.** Not called-and-discarded: never called.
  A metric that is not defined on a 40x40 ROI must not be handed one, because
  every one of them will return *something*.
* **Crashed** -> reported as a crash, distinct from ineligible.

The last two being distinguishable is the whole point (see
``docs/metric_outcome_contract.rst``): a NaN that means "does not apply here" and
a NaN that means "this metric is broken" look identical until you separate them.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from spectramr.core.metrics.outcome import (
    MetricNotApplicableError,
    MetricOutcome,
    MetricOutcomeReport,
)
from spectramr.core.metrics.regions.crop import crop_to_region
from spectramr.core.metrics.regions.eligibility import (
    Eligibility,
    RoiSupport,
    evaluate_eligibility,
)
from spectramr.core.metrics.regions.reductions import MAP_REDUCERS
from spectramr.core.metrics.regions.types import RegionMask, RegionSet

__all__ = [
    "eligibility_table",
    "score_region",
    "score_region_set",
]


def eligibility_table(metrics: list[str], regions: RegionSet) -> list[Eligibility]:
    """The full (metric x region) exclusion table.

    A **pure function of policy and geometry** -- no tensors, no metric calls. So a
    run can emit its complete exclusion table *before it computes anything*, and a
    reviewer can see what was excluded and why without waiting for a sweep.
    """
    return [evaluate_eligibility(metric, region) for region in regions for metric in metrics]


def score_region(
    metric: str,
    metric_fn: Callable[..., object],
    pred: torch.Tensor,
    target: torch.Tensor,
    region: RegionMask,
) -> MetricOutcome:
    """Score one metric on one region, through the gate.

    Args:
        metric: registry key -- used to look up the declared ROI policy.
        metric_fn: the metric callable. **Only invoked when the gate passes.**
        pred / target: full-FOV ``[B, C, H, W]`` tensors. The restriction happens
            here, not in the caller -- passing pre-cropped data would bypass the
            gate.
        region: the region to restrict to.
    """
    verdict = evaluate_eligibility(metric, region)
    if not verdict.eligible:
        assert verdict.reason is not None  # guaranteed by evaluate_eligibility
        return MetricOutcome.not_applicable(metric, verdict.reason, verdict.detail)

    mask = region.mask.to(pred.device)
    try:
        if verdict.support is RoiSupport.MAP_THEN_MASK:
            reducer = MAP_REDUCERS[metric]
            value = reducer.reduce(reducer.maps(pred, target), mask)
        elif verdict.support is RoiSupport.CROP_THEN_COMPUTE:
            value = metric_fn(crop_to_region(pred, region), crop_to_region(target, region))
        else:  # pragma: no cover - NOT_RESTRICTABLE never reaches here
            raise AssertionError(
                f"{metric}: {verdict.support} passed the gate; that is a gate bug."
            )
    except MetricNotApplicableError as exc:
        return MetricOutcome.not_applicable(metric, exc.reason, exc.detail)
    except Exception as exc:  # classified, not swallowed
        return MetricOutcome.crashed(metric, exc)

    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    return MetricOutcome.from_value(metric, float(value))  # a NaN return is a defect


def score_region_set(
    metrics: dict[str, Callable[..., object]],
    pred: torch.Tensor,
    target: torch.Tensor,
    regions: RegionSet,
) -> dict[str, MetricOutcomeReport]:
    """Score every metric on every region. Returns ``region_id -> report``.

    Ineligible metrics appear in the report as ``NOT_APPLICABLE`` with a reason.
    They are **not** ranked as all-NaN columns: an all-NaN column ranked "last"
    reads as *"this metric is bad at grey matter"* when the truth is *"this metric
    has no meaning on grey matter"*. Those are different claims, and the leaderboard
    must make a smaller, honest pool rather than a big dishonest one.
    """
    return {
        region.region_id: MetricOutcomeReport(
            outcomes=tuple(
                score_region(name, fn, pred, target, region) for name, fn in metrics.items()
            )
        )
        for region in regions
    }
