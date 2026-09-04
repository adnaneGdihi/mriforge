"""Score-Function Alignment Ranker.

Computes the Karras-weighted expected cosine alignment between

    - direction of metric improvement: ``-∇_{hat_x} M_k`` for "lower is
      better" metrics, ``+∇_{hat_x} M_k`` for "higher is better",

and

    - the score of the empirical clean-data distribution at multiple noise
      scales ``sigma``: ``s_phi(hat_x, sigma)``.

The score is in ``[-1, 1]``. ``1`` means minimizing the metric pushes
samples toward the clean-data manifold; ``-1`` means it pushes away. The
ranker reports the Karras-weighted average across noise scales as the
single score.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from ..gradient import metric_gradient
from ..score_network import build_default_score_fn
from ..simulator import _build_sample_context
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker


def _resolve_needs(name: str) -> tuple[tuple[str, ...], bool]:
    """Return ``(needs, needs_context)`` for a metric name; ``((), False)`` if
    the name is not a registry metric (e.g. a summary-metric placeholder)."""
    try:
        from spectramr.core.metrics.registry import MetricsRegistry

        return MetricsRegistry.needs(name), bool(MetricsRegistry.needs_context(name))
    except Exception:
        return (), False


logger = logging.getLogger(__name__)


@dataclass
class SFAConfig:
    sigma_levels: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
    sigma_data: float = 0.5
    spsa_delta: float = 1e-2
    spsa_samples: int = 6
    max_pairs: int = 64  # cap to keep CPU runs fast
    log_every: int = 10  # emit progress every N metrics


def _karras_weights(sigmas: tuple[float, ...], sigma_data: float) -> dict[float, float]:
    out = {}
    for s in sigmas:
        out[s] = (s**2 + sigma_data**2) / max(sigma_data**2, 1e-12)
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    if af.numel() != bf.numel():
        m = min(af.numel(), bf.numel())
        af = af[:m]
        bf = bf[:m]
    na = float(af.norm().item())
    nb = float(bf.norm().item())
    # A non-finite norm (a metric produced a NaN/inf gradient or score vector) must
    # collapse to a neutral 0.0 — note ``na < 1e-12`` is False when ``na`` is NaN,
    # so the zero-vector guard alone would let NaN through into the SFA score.
    if not (math.isfinite(na) and math.isfinite(nb)) or na < 1e-12 or nb < 1e-12:
        return 0.0
    out = float((af * bf).sum().item() / (na * nb))
    return out if math.isfinite(out) else 0.0


class SFARanker(BaseRanker):
    method_name = "SFA"

    def __init__(
        self,
        config: SFAConfig | None = None,
        score_fn=None,
    ) -> None:
        self.config = config or SFAConfig()
        self.score_fn = score_fn

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return RankingResult(method=self.method_name, scores={}, ranks=[], diagnostics={})
        # Build score function from clean references if not provided.
        if self.score_fn is None:
            seen: dict[str, torch.Tensor] = {}
            for s in dataset.samples:
                seen.setdefault(s.content_id, s.clean)
            self.score_fn = build_default_score_fn(list(seen.values()))

        weights = _karras_weights(self.config.sigma_levels, self.config.sigma_data)
        wsum = sum(weights.values()) + 1e-12

        # Subsample pairs for speed.
        N = min(self.config.max_pairs, dataset.n_samples)
        stride = max(1, dataset.n_samples // N)
        sample_idx = list(range(0, dataset.n_samples, stride))[:N]

        scores: dict[str, float] = {}
        per_sigma_curves: dict[str, dict[float, float]] = {}

        names = list(metric_set.names())
        M = len(names)
        logger.info(
            "SFA ranker starting: M=%d metrics × %d pairs × (%d SPSA + %d sigmas)"
            " = ~%d metric calls",
            M,
            len(sample_idx),
            2 * self.config.spsa_samples,
            len(self.config.sigma_levels),
            M * len(sample_idx) * (2 * self.config.spsa_samples + len(self.config.sigma_levels)),
        )

        for m_idx, name in enumerate(names):
            metric = metric_set.metrics[name]
            metric_sign = 1.0 if metric_set.is_higher_better(name) else -1.0
            differentiable = metric_set.is_differentiable(name)
            needs, needs_ctx = _resolve_needs(name)
            sigma_acc: dict[float, float] = dict.fromkeys(self.config.sigma_levels, 0.0)
            counts: dict[float, int] = dict.fromkeys(self.config.sigma_levels, 0)
            for idx in sample_idx:
                sample = dataset.samples[idx]
                clean = sample.clean
                hat = sample.degraded
                # Give NR/physics metrics their measurement context so the SPSA
                # gradient is taken on the true acquisition, not a NaN. Without
                # this the whole NR battery aligned at SFA=0 (context-free calls
                # NaN -> zero gradient -> zero cosine).
                ctx = _build_sample_context(clean, needs, real=sample.assets) if needs_ctx else None
                grad = metric_gradient(
                    metric,
                    clean,
                    hat,
                    differentiable_hint=differentiable,
                    delta=self.config.spsa_delta,
                    n_samples=self.config.spsa_samples,
                    seed=sample.seed,
                    context=ctx,
                )
                metric_dir = metric_sign * grad
                for sigma in self.config.sigma_levels:
                    score_vec = self.score_fn(hat, sigma)
                    cos = _cosine(metric_dir, score_vec)
                    sigma_acc[sigma] += cos
                    counts[sigma] += 1
            per_sigma_curves[name] = {
                s: (sigma_acc[s] / max(counts[s], 1)) for s in self.config.sigma_levels
            }
            scores[name] = (
                sum(weights[s] * per_sigma_curves[name][s] for s in self.config.sigma_levels) / wsum
            )

            if self.config.log_every > 0 and (m_idx + 1) % self.config.log_every == 0:
                logger.info(
                    "SFA: %d/%d metrics done (last: %s, SFA=%.3f)",
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
                "per_sigma_curves": per_sigma_curves,
                "sigma_levels": list(self.config.sigma_levels),
                "n_pairs": len(sample_idx),
            },
        )
