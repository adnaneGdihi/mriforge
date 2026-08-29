"""Latent Geodesic Discrepancy Ranker.

Builds a manifold from clean references in the encoder's latent space,
projects the degraded samples onto it via Nyström extension, and ranks
metrics by Spearman rank correlation against:

* the diffusion distance ``D_t`` from each clean reference to its
  corresponding degraded image, evaluated at multiple times ``t`` —
  multiple curves per metric let users see *at what scale* the metric
  is most informative,
* the geodesic distance ``G`` on ``-log w`` weights, which is sensitive
  to off-manifold drift.

The "score" reported per metric is the diffusion-time-averaged Spearman
correlation. The diagnostic dict carries the full curve plus the
geodesic-based variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..encoder import SimpleAEEncoder, embed_batch, fit_random_features
from ..manifold import (
    ManifoldConfig,
    build_knn_graph,
    diffusion_eigenstructure,
    dijkstra_geodesics,
)
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker


@dataclass
class LGDRConfig:
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    encoder_seed: int = 0
    embed_dim: int = 64


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Signed, tie-corrected (mid-rank) Spearman rank correlation; NaN-safe.

    Uses scipy mid-ranks (via :func:`_spearman_legacy.srocc_signed`) rather than the
    tie-blind ``argsort(argsort)``, which returned ``|rho| = 1`` on saturated/tied
    diffusion-distance trajectories and biased coarse-valued metrics upward
    (critique H2). The sign is preserved — LGDR consumes the signed correlation.
    """
    n = x.shape[0]
    if n < 4:
        return 0.0
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 4:
        return 0.0
    from ._spearman_legacy import srocc_signed

    return srocc_signed(x[mask], y[mask])


class LGDRRanker(BaseRanker):
    method_name = "LGDR"

    def __init__(self, config: LGDRConfig | None = None, encoder=None) -> None:
        self.config = config or LGDRConfig()
        self.encoder = encoder

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        n = dataset.n_samples
        if n == 0:
            return RankingResult(method=self.method_name, scores={}, ranks=[], diagnostics={})

        # Build encoder if none provided.
        if self.encoder is None:
            enc = SimpleAEEncoder(in_channels=1, embed_dim=self.config.embed_dim)
            enc = fit_random_features(enc, seed=self.config.encoder_seed)
            self.encoder = enc

        # Embed clean references (one per content) and degraded samples.
        unique_clean: dict[str, torch.Tensor] = {}
        for s in dataset.samples:
            if s.content_id not in unique_clean:
                unique_clean[s.content_id] = s.clean
        clean_keys = list(unique_clean.keys())
        clean_embeddings = embed_batch(self.encoder, [unique_clean[k] for k in clean_keys])

        # Manifold from clean references.
        weights, _, epsilon = build_knn_graph(clean_embeddings, self.config.manifold)
        eigvals, psi = diffusion_eigenstructure(weights, n_eigen=self.config.manifold.n_eigen)

        # Embed degraded samples.
        degraded_embeddings = embed_batch(self.encoder, [s.degraded for s in dataset.samples])

        # Per-sample diffusion distance to its own clean reference, at every t.
        # Nyström extension: for a query q, compute weights to all reference
        # points, normalize, and use it as a row in psi.
        ref_idx = torch.tensor(
            [clean_keys.index(s.content_id) for s in dataset.samples],
            dtype=torch.long,
        )

        def _nystrom(query: torch.Tensor) -> torch.Tensor:
            d = (
                torch.cdist(query.unsqueeze(0).float(), clean_embeddings.float(), p=2).squeeze(0)
                ** 2
            )
            w = torch.exp(-d / max(epsilon, 1e-9))
            if w.sum() <= 1e-12:
                return torch.zeros(psi.shape[1], dtype=torch.float64)
            w = w / w.sum()
            return (w.unsqueeze(1).double() * psi).sum(dim=0)

        # Compute diffusion distances at each time.
        per_t_distances: dict[int, torch.Tensor] = {}
        for t in self.config.manifold.diffusion_times:
            dists = torch.zeros(n, dtype=torch.float64)
            for i, s in enumerate(dataset.samples):
                ref_psi = psi[ref_idx[i]]
                q_psi = _nystrom(degraded_embeddings[i])
                if eigvals.shape[0] <= 1:
                    dists[i] = float((q_psi - ref_psi).norm().item())
                else:
                    coeff = (eigvals[1:] ** (2 * t)).double()
                    diff = (q_psi[1:] - ref_psi[1:]).double()
                    dists[i] = float(((diff**2) * coeff).sum().sqrt().item())
            per_t_distances[t] = dists

        # Geodesic distances among clean references; degraded sample's geodesic
        # is the geodesic from its content's clean to the *nearest* clean to
        # the degraded embedding.
        geo_matrix = dijkstra_geodesics(weights)
        geo_per_sample = torch.zeros(n, dtype=torch.float64)
        for i, s in enumerate(dataset.samples):
            d_to_refs = torch.cdist(
                degraded_embeddings[i].unsqueeze(0).float(),
                clean_embeddings.float(),
                p=2,
            ).squeeze(0)
            nearest = int(torch.argmin(d_to_refs).item())
            ref_i = int(ref_idx[i].item())
            g = float(geo_matrix[ref_i, nearest].item())
            if not math.isfinite(g):
                g = float(d_to_refs[nearest].item())  # fallback
            geo_per_sample[i] = g

        # Score per metric: average Spearman over diffusion times. Use signed
        # version so "higher score = better" — for a "lower is better" metric
        # we expect positive correlation with distances; for "higher is
        # better", negative (so flip sign).
        scores: dict[str, float] = {}
        per_t_curves: dict[str, dict[int, float]] = {}
        geo_scores: dict[str, float] = {}
        for name in metric_set.names():
            y = dataset.metric_array(name)
            sign = -1.0 if metric_set.is_higher_better(name) else 1.0
            curves = {}
            for t, d in per_t_distances.items():
                curves[t] = sign * _spearman(y, d)
            per_t_curves[name] = curves
            scores[name] = sum(curves.values()) / max(len(curves), 1) if curves else 0.0
            geo_scores[name] = sign * _spearman(y, geo_per_sample)

        ranks = self._ranks_from_scores(scores)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=ranks,
            diagnostics={
                "per_t_curves": per_t_curves,
                "geodesic_scores": geo_scores,
                "epsilon": epsilon,
                "n_clean": int(clean_embeddings.shape[0]),
                "diffusion_times": list(self.config.manifold.diffusion_times),
            },
        )
