"""Equivariance-Sensitivity Decomposition Ranker.

For each metric ``M_k``:

    IG_k = E_{x, g ~ G_NUIS} | M_k(g x, g hat_x) - M_k(x, hat_x) |
                                / |M_k(x, hat_x)|
    SG_k = E_{x, g ~ G_PERT} | M_k(x, g hat_x) - M_k(x, hat_x) |
    ESD_k = SG_k / (IG_k + eps)

The denominator ``IG_k`` is the Invariance Gap (smaller = better; metric
respects nuisance group). The numerator ``SG_k`` is the Sensitivity Gain
(larger = better; metric reacts to clinically meaningful changes). Their
ratio is unitless and comparable across metrics.

We sample each group element with ``n_haar_samples`` Monte-Carlo draws of
its parameters. The reference image pool is taken from the dataset's
unique ``(clean, degraded)`` pairs.

Cost model (per ``rank`` call): ``M * (|G_NUIS| + |G_PERT|) * N_pairs *
N_haar`` metric evaluations plus ``M * N_pairs`` base evaluations.
With defaults ``M=107, |G|=4+4, N_pairs=32, N_haar=4`` this is
~110k metric calls — the dominant cost in the Gen 3 pipeline. Progress
logs are emitted every ``log_every`` metrics so silent stalls are
visible in the log file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from ..group_actions import G_NUIS, G_PERT, GroupActionEntry
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker

logger = logging.getLogger(__name__)


@dataclass
class ESDConfig:
    n_haar_samples: int = 4
    max_pairs: int = 32
    eps: float = 1e-9
    log_every: int = 10  # emit progress every N metrics
    max_consecutive_failures: int = 8  # short-circuit persistently-failing metrics
    # §7/L5: tie each Haar draw to the sample's own deterministic seed (which the
    # simulator derives from (content_id, family, severity)). Re-runs and re-orderings
    # then produce bit-identical draws per cell, instead of depending on a single
    # global stream whose values shift with iteration order / subsampling.
    seed_by_sample: bool = False


def _safe_metric(metric, pred, target) -> float:
    try:
        return float(metric(pred, target))
    except Exception:
        return float("nan")


_SEED_MASK = (1 << 63) - 1


def _mix_seed(sample_seed: int, salt: int) -> int:
    """Stable per-(sample, group-entry) seed for reproducible Haar draws."""
    return (abs(int(sample_seed)) * 1_000_003 + salt) & _SEED_MASK


class ESDRanker(BaseRanker):
    method_name = "ESD"

    def __init__(
        self,
        config: ESDConfig | None = None,
        nuisance_group: list[GroupActionEntry] | None = None,
        pertinent_group: list[GroupActionEntry] | None = None,
    ) -> None:
        self.config = config or ESDConfig()
        self.nuisance_group = nuisance_group or G_NUIS
        self.pertinent_group = pertinent_group or G_PERT

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return RankingResult(method=self.method_name, scores={}, ranks=[], diagnostics={})

        # Subsample pairs.
        N = min(self.config.max_pairs, dataset.n_samples)
        stride = max(1, dataset.n_samples // N)
        pairs = [(s.clean, s.degraded, s.seed) for s in dataset.samples[::stride][:N]]
        gen = torch.Generator().manual_seed(0xC0FFEE)

        scores: dict[str, float] = {}
        ig_per_metric: dict[str, dict[str, float]] = {}
        sg_per_metric: dict[str, dict[str, float]] = {}

        names = list(metric_set.names())
        M = len(names)
        logger.info(
            "ESD ranker starting: M=%d metrics × %d groups × %d pairs × %d haar = ~%d metric calls",
            M,
            len(self.nuisance_group) + len(self.pertinent_group),
            len(pairs),
            self.config.n_haar_samples,
            M
            * (len(self.nuisance_group) + len(self.pertinent_group))
            * len(pairs)
            * self.config.n_haar_samples,
        )

        for m_idx, name in enumerate(names):
            metric = metric_set.metrics[name]
            ig_breakdown: dict[str, float] = {}
            sg_breakdown: dict[str, float] = {}

            ig_total: list[float] = []
            sg_total: list[float] = []

            # Compute per-pair base once per metric. The audit (2026-05)
            # noted the legacy code recomputed `base` inside every group
            # loop — a 7× redundancy at |G|=8. Hoisting it here saves
            # ~7·M·N_pairs metric calls per ESD run.
            pair_bases: list[tuple[torch.Tensor, torch.Tensor, float, int] | None] = []
            consecutive_failures = 0
            for clean, hat, pseed in pairs:
                base = _safe_metric(metric, hat, clean)
                if base != base:  # NaN
                    pair_bases.append(None)
                    consecutive_failures += 1
                    if consecutive_failures >= self.config.max_consecutive_failures:
                        # Persistently-failing metric — emit a warning and
                        # short-circuit so the ESD run does not stall here.
                        logger.warning(
                            "ESD: metric %r returned NaN on %d consecutive "
                            "base evaluations; short-circuiting to score=0.",
                            name,
                            consecutive_failures,
                        )
                        scores[name] = 0.0
                        break
                else:
                    consecutive_failures = 0
                    pair_bases.append((clean, hat, float(base), int(pseed)))
            if name in scores:
                # Short-circuited above; skip group loops.
                ig_per_metric[name] = {}
                sg_per_metric[name] = {}
                continue

            n_nuis = len(self.nuisance_group)
            for e_idx, entry in enumerate(self.nuisance_group):
                vals = []
                for pb in pair_bases:
                    if pb is None:
                        continue
                    clean, hat, base, pseed = pb
                    base_abs = abs(base) + self.config.eps
                    g = (
                        torch.Generator().manual_seed(_mix_seed(pseed, e_idx))
                        if self.config.seed_by_sample
                        else gen
                    )
                    for _ in range(self.config.n_haar_samples):
                        params = entry.sample_params(g)
                        gx, ghat = entry.action(clean, hat, params)
                        v = _safe_metric(metric, ghat, gx)
                        if v != v:
                            continue
                        vals.append(abs(v - base) / base_abs)
                ig_breakdown[entry.name] = float(sum(vals) / max(len(vals), 1)) if vals else 0.0
                ig_total.extend(vals)

            for e_idx, entry in enumerate(self.pertinent_group):
                vals = []
                for pb in pair_bases:
                    if pb is None:
                        continue
                    clean, hat, base, pseed = pb
                    # Salt past the nuisance group so the two groups draw independently.
                    g = (
                        torch.Generator().manual_seed(_mix_seed(pseed, n_nuis + e_idx))
                        if self.config.seed_by_sample
                        else gen
                    )
                    for _ in range(self.config.n_haar_samples):
                        params = entry.sample_params(g)
                        # Pertinent transforms are applied to ``hat`` only,
                        # measuring whether the metric notices the change.
                        gx, ghat = entry.action(clean, hat, params)
                        v = _safe_metric(metric, ghat, gx)
                        if v != v:
                            continue
                        vals.append(abs(v - base))
                sg_breakdown[entry.name] = float(sum(vals) / max(len(vals), 1)) if vals else 0.0
                sg_total.extend(vals)

            ig = sum(ig_total) / max(len(ig_total), 1) if ig_total else 0.0
            sg = sum(sg_total) / max(len(sg_total), 1) if sg_total else 0.0
            scores[name] = sg / (ig + self.config.eps)
            ig_per_metric[name] = ig_breakdown
            sg_per_metric[name] = sg_breakdown

            if self.config.log_every > 0 and (m_idx + 1) % self.config.log_every == 0:
                logger.info(
                    "ESD: %d/%d metrics done (last: %s, ESD=%.3f)",
                    m_idx + 1,
                    M,
                    name,
                    scores[name],
                )

        ranks = self._ranks_from_scores(scores)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=ranks,
            diagnostics={
                "invariance_gap": ig_per_metric,
                "sensitivity_gain": sg_per_metric,
                "n_pairs": len(pairs),
                "n_haar_samples": self.config.n_haar_samples,
            },
        )
