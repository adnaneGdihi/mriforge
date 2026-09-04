"""Region-conditional meta-evaluation: one dataset per region, one honest pool per region.

This is the seam that turns the region *machinery* into a region *axis*. Everything
it needs already existed -- :mod:`spectramr.core.metrics.regions` scores a metric on a
mask, :mod:`.leaderboard` ranks a cell -- and nothing connected them, so the sweep
scored the full FOV and the region packages were reachable only from their own tests.

The decomposition
-----------------
**One ``MetricEvaluationDataset`` per region, not a region field on
``DegradationSample``.** A region is a property of the *slice*, not of the
(clean, degraded) pair: a sample carries the same region no matter which artifact
degraded it. Putting it on the sample would multiply every tensor-holding sample by
``n_regions`` for no information gain. And ``BaseRanker.rank(metric_set, dataset)``
then needs **zero changes** -- the deliverable *is* a leaderboard per cell, so one
ranker run per cell is the decomposition, not overhead.

The pool rule: all-or-nothing, per region
-----------------------------------------
Eligibility is a function of policy **and geometry**, and geometry varies per slice
-- a lesion box is 40x40 on one slice and 80x80 on another. So a metric can be
eligible on some instances of ``path:lesion_any`` and not others.

A metric enters a region's pool **only if it is eligible on every instance of that
region in the cohort.** The alternative -- score it where it fits, NaN where it does
not -- looks harmless and is not: the rankers mask NaN before correlating, so the
metric's severity-correlation would be computed *only over the large lesions*. That
is a silent selection effect, and it biases exactly the cells the region axis exists
to interrogate. A partially-measured metric is excluded, and the exclusion says on
how many instances it failed.

Crashes stay separable from ineligibility (the vocabulary #213 introduced): a metric
that is undefined here and a metric that is broken are different facts, and both are
reported rather than merged into NaN.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
    MetricSet,
)
from spectramr.core.metrics.outcome import MetricStatus
from spectramr.core.metrics.regions.eligibility import Eligibility, evaluate_eligibility
from spectramr.core.metrics.regions.scorer import score_region
from spectramr.core.metrics.regions.types import FULL_REGION_ID, RegionMask, RegionSet

__all__ = [
    "RegionalEvaluationBundle",
    "build_regional_bundles",
    "region_ids_in",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegionalEvaluationBundle:
    """Everything needed to rank one region -- and everything that region threw away."""

    region_id: str

    dataset: MetricEvaluationDataset
    """Only the samples whose slice actually *has* this region. A lesion region exists
    on annotated slices only, so ``n_samples`` differs per region -- which is why
    ``LeaderboardCell`` carries its own ``n_samples`` and power-gates on it."""

    metric_set: MetricSet
    """The **eligible** pool only. Ineligible metrics are absent, not present-as-NaN:
    an all-NaN column ranked last reads as *"this metric is bad at grey matter"* when
    the truth is *"this metric has no meaning on grey matter"*."""

    excluded: Mapping[str, str] = field(default_factory=dict)
    """``metric -> why``. Declared-ineligible, with the reason and the instance count."""

    crashed: Mapping[str, str] = field(default_factory=dict)
    """``metric -> exception``. A BUG, kept distinct from ineligible. Also out of the
    pool -- a column that is half measurement and half crash is not a measurement."""

    eligibility: tuple[Eligibility, ...] = ()
    """The exclusion table for this region, as evaluated on its first instance. A pure
    function of policy + geometry, so it can be emitted before anything is computed."""

    @property
    def pool_size(self) -> int:
        """The denominator of every claim made about this region."""
        return len(self.metric_set.metrics)

    @property
    def n_samples(self) -> int:
        return self.dataset.n_samples

    def coverage_row(self) -> dict[str, object]:
        """One row of the run's ``coverage`` sheet.

        Exists so no reader can mistake "the best of the 31 metrics that are defined on
        grey matter" for "the best of 164".
        """
        return {
            "region": self.region_id,
            "n_samples": self.n_samples,
            "pool_size": self.pool_size,
            "n_excluded": len(self.excluded),
            "n_crashed": len(self.crashed),
        }


def region_ids_in(regions: Mapping[str, RegionSet]) -> tuple[str, ...]:
    """Every region id present anywhere in the cohort, ``full`` first.

    ``full`` leads because it is the control the regional rankings are compared
    against; a reader scanning the tables should meet it before the restrictions.
    """
    seen: set[str] = set()
    for region_set in regions.values():
        seen.update(r.region_id for r in region_set)
    rest = sorted(seen - {FULL_REGION_ID})
    return (FULL_REGION_ID, *rest) if FULL_REGION_ID in seen else tuple(rest)


def _instances(
    samples: Sequence[DegradationSample],
    regions: Mapping[str, RegionSet],
    region_id: str,
) -> list[tuple[int, RegionMask]]:
    """``(sample_index, mask)`` for every sample whose slice carries ``region_id``."""
    out: list[tuple[int, RegionMask]] = []
    for i, s in enumerate(samples):
        region_set = regions[s.content_id]
        try:
            out.append((i, region_set[region_id]))
        except KeyError:
            continue  # this slice has no such region -- not an error, just absent
    return out


def _pool_for(
    metrics: Sequence[str],
    instances: Sequence[tuple[int, RegionMask]],
) -> tuple[list[str], dict[str, str], tuple[Eligibility, ...]]:
    """The all-or-nothing eligible pool for one region, plus why each metric was cut.

    A metric survives only if it is eligible on **every** instance. See the module
    docstring: scoring it only where it fits would hand the rankers a column measured
    on a size-biased subset of the cohort.
    """
    pool: list[str] = []
    excluded: dict[str, str] = {}

    for metric in metrics:
        verdicts = [evaluate_eligibility(metric, mask) for _, mask in instances]
        failures = [v for v in verdicts if not v.eligible]
        if not failures:
            pool.append(metric)
            continue

        # Report the FIRST reason, and how widely it bit. "ineligible on 12/50" is the
        # part a reader needs: a metric cut on every instance is a policy decision, one
        # cut on 12 of 50 is a geometry effect, and they are read differently.
        first = failures[0]
        excluded[metric] = (
            f"{first.reason}: {first.detail} "
            f"[ineligible on {len(failures)}/{len(verdicts)} instances]"
        )

    table = tuple(evaluate_eligibility(m, instances[0][1]) for m in metrics) if instances else ()
    return pool, excluded, table


def build_regional_bundles(
    samples: Sequence[DegradationSample],
    metric_set: MetricSet,
    regions: Mapping[str, RegionSet],
    *,
    region_ids: Sequence[str] | None = None,
) -> list[RegionalEvaluationBundle]:
    """Score every metric on every region, and bundle each region for the rankers.

    Args:
        samples: the degradation sweep. ``content_id`` keys into ``regions``.
        metric_set: the full pool. Each region gets the eligible **subset** of it.
        regions: ``content_id -> RegionSet``. Regions are computed on the CLEAN
            reference (see :mod:`spectramr.core.metrics.regions.types`), so one set per
            slice covers every artifact x severity derived from it.
        region_ids: restrict to these regions. Defaults to every region in the cohort.

    Raises:
        KeyError: when a sample's ``content_id`` has no ``RegionSet``. There is no
            fallback to the full FOV: silently scoring the whole slice for a sample
            whose masks failed to load would put un-regioned rows into a regional
            leaderboard, and nothing downstream could tell.
    """
    missing = sorted({s.content_id for s in samples} - set(regions))
    if missing:
        raise KeyError(
            f"{len(missing)} sample content_id(s) have no RegionSet, e.g. {missing[:3]}. "
            "Refusing to fall back to the full FOV -- that would silently mix "
            "whole-slice rows into a regional leaderboard."
        )

    ids = tuple(region_ids) if region_ids is not None else region_ids_in(regions)
    names = metric_set.names()
    bundles: list[RegionalEvaluationBundle] = []

    for region_id in ids:
        instances = _instances(samples, regions, region_id)
        if not instances:
            logger.warning("region %s: no slice in the cohort carries it -- skipped.", region_id)
            continue

        pool, excluded, table = _pool_for(names, instances)
        if not pool:
            logger.warning(
                "region %s: every metric is ineligible (%d excluded). No leaderboard "
                "for this region -- that is a result, not an error.",
                region_id,
                len(excluded),
            )

        values: dict[str, list[float]] = {m: [] for m in pool}
        crashed: dict[str, str] = {}

        for idx, mask in instances:
            sample = samples[idx]
            pred = _as_bchw(sample.degraded)
            target = _as_bchw(sample.clean)
            for metric in pool:
                outcome = score_region(metric, metric_set.metrics[metric], pred, target, mask)
                if outcome.status is MetricStatus.CRASHED and metric not in crashed:
                    crashed[metric] = outcome.detail or "crashed"
                values[metric].append(outcome.value)

        # A crash is a defect, not a measurement. Drop the metric rather than hand the
        # rankers a column that is part signal and part exception.
        for metric in crashed:
            values.pop(metric, None)
            logger.warning(
                "region %s: metric %s CRASHED (%s) -- dropped from this region's pool.",
                region_id,
                metric,
                crashed[metric],
            )
        pool = [m for m in pool if m not in crashed]

        kept = [samples[i] for i, _ in instances]
        bundles.append(
            RegionalEvaluationBundle(
                region_id=region_id,
                dataset=MetricEvaluationDataset(samples=kept, metric_values=values),
                metric_set=MetricSet(
                    metrics={m: metric_set.metrics[m] for m in pool},
                    higher_is_better={m: metric_set.is_higher_better(m) for m in pool},
                    differentiable={m: metric_set.differentiable.get(m, True) for m in pool},
                ),
                excluded=excluded,
                crashed=crashed,
                eligibility=table,
            )
        )
        logger.info(
            "region %-22s n=%-5d pool=%-4d excluded=%-4d crashed=%d",
            region_id,
            len(kept),
            len(pool),
            len(excluded),
            len(crashed),
        )

    return bundles


def _as_bchw(t):
    """``[C, H, W]`` -> ``[1, C, H, W]``. The scorer wants a batch dim; samples have none."""
    return t.unsqueeze(0) if t.ndim == 3 else t
