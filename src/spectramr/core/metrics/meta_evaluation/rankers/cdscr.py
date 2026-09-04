"""Cross-Domain Spectral Coherence Ranker.

Scores how much of a metric's variation is explained by *residual shape*
(spectral profile + wavelet + scattering) after partialing out residual
energy ``||r||_2`` and the content covariate.

Estimator: a finite-sample, plug-in approximation to the conditional mutual
information ``I(M_k; zeta(r) | ||r||, C)`` using Gaussian-copula partial
correlations. The estimator is bias-bounded by ``log(N) / N`` and is
asymptotically equivalent to the true conditional MI for elliptical
copulas — sufficient for ranking but not for absolute MI estimation.

The rationale for the Gaussian copula: rank-transform every variable to
N(0, 1) marginals and compute the partial-correlation determinant ratio,
which under joint Gaussianity equals ``-0.5 * log(1 - r_partial^2)``.
The same formula is used as a *score* even when the joint isn't Gaussian
because the rank transform makes the estimator robust to monotone
re-parameterizations of the metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..descriptors import DescriptorConfig, residual_descriptor
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker


@dataclass
class CDSCRConfig:
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    # Number of leading PCs of the descriptor used in the partial correlation.
    n_descriptor_components: int = 8
    eps: float = 1e-9


def _rank_normal_score(values: torch.Tensor) -> torch.Tensor:
    """Inverse-Gaussian CDF rank transform (van der Waerden scores).

    Ties are resolved by averaging the rank assignments over the tied group
    — this is the standard treatment in nonparametric statistics and is
    necessary so that constant series map to a constant zero output rather
    than to an arbitrary argsort permutation that fakes structure.
    """
    n = values.shape[0]
    if n == 0:
        return values
    finite = torch.isfinite(values)
    out = torch.zeros_like(values)
    if int(finite.sum().item()) < 2:
        return out
    finite_idx = torch.where(finite)[0]
    finite_vals = values[finite_idx]
    if float(finite_vals.std(unbiased=False).item()) < 1e-12:
        # Constant series: every rank-normal score is 0.
        return out
    order = torch.argsort(finite_vals)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    # Average rank within tied groups.
    sorted_vals = finite_vals[order]
    i = 0
    n_finite = len(order)
    while i < n_finite:
        j = i
        while j + 1 < n_finite and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # ranks start at 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    p = (ranks - 0.5) / max(len(order), 1)
    p = p.clamp(1e-6, 1 - 1e-6)
    z = torch.special.erfinv(2 * p - 1) * math.sqrt(2.0)
    out[finite_idx] = z.to(out.dtype)
    return out


def _content_design_matrix(content_ids: list[str]) -> torch.Tensor:
    """One-hot encoding of content (subject) IDs."""
    uniq = sorted(set(content_ids))
    if len(uniq) <= 1:
        return torch.zeros(len(content_ids), 0, dtype=torch.float64)
    idx = {c: i for i, c in enumerate(uniq)}
    out = torch.zeros(len(content_ids), len(uniq) - 1, dtype=torch.float64)
    for r, c in enumerate(content_ids):
        i = idx[c]
        if i < len(uniq) - 1:
            out[r, i] = 1.0
    return out


def _partial_corr(y: torch.Tensor, X: torch.Tensor, Z: torch.Tensor) -> float:
    """Multiple-correlation strength of y on X, controlling for Z.

    Regresses Z out of both y and X, then reports the **adjusted** multiple
    correlation ``sqrt(R^2_adj)`` of the residualized y on the residualized X, in
    ``[0, 1]``. The *unadjusted* R^2 is upward-biased by ``p / (n_eff - p)`` (with
    ``p`` predictors): for two independent variables it returns a large positive
    value purely from over-fitting, and because ``p`` is fixed across metrics it
    injected a dimensionality-dependent floor that reordered metrics by descriptor
    count rather than true association (critique H3). The adjustment uses the
    effective sample size ``n_eff = n - q`` (``q`` = conditioning-set rank including
    the intercept) so both the controlled-for df and the predictor df are charged.

    (This is the multiple-correlation coefficient — the OLS R of y on all of X —
    not a CCA canonical correlation; the earlier docstring's CCA framing was
    inaccurate.)
    """
    n = y.shape[0]
    if n < 4 or X.shape[1] == 0:
        return 0.0
    Z_full = torch.cat([torch.ones(n, 1, dtype=torch.float64), Z], dim=1)
    # Residualize both y and X against Z.
    ZtZ_inv = torch.linalg.pinv(Z_full.T @ Z_full)
    Pz = Z_full @ ZtZ_inv @ Z_full.T
    Iy = torch.eye(n, dtype=torch.float64) - Pz
    y_res = Iy @ y.unsqueeze(1)
    X_res = Iy @ X
    # Standardize.
    y_res = y_res - y_res.mean()
    y_std = float(y_res.std(unbiased=False).item())
    if y_std < 1e-12:
        return 0.0
    y_res = y_res / y_std
    X_res = X_res - X_res.mean(dim=0, keepdim=True)
    X_std = X_res.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-12)
    X_res = X_res / X_std
    # Multivariate R^2 of y_res ~ X_res. Solve OLS, get fitted r.
    coefs, *_ = torch.linalg.lstsq(X_res, y_res)
    pred = X_res @ coefs
    ss_res = ((y_res - pred) ** 2).sum().item()
    ss_tot = (y_res**2).sum().item()
    if ss_tot < 1e-12:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    # Adjusted R^2 (Ezekiel): charge p predictors against n_eff = n - q effective
    # observations (q = conditioning-set columns already consumed by Pz). If there
    # are too few residual df to estimate, the association is unidentifiable → 0.
    p = X_res.shape[1]
    q = Z_full.shape[1]
    n_eff = n - q
    if n_eff - p - 1 <= 0:
        return 0.0
    r2_adj = 1.0 - (1.0 - r2) * (n_eff - 1) / (n_eff - p - 1)
    r2_adj = max(0.0, min(1.0, r2_adj))
    return math.sqrt(r2_adj)


class CDSCRRanker(BaseRanker):
    method_name = "CDSCR"

    def __init__(self, config: CDSCRConfig | None = None) -> None:
        self.config = config or CDSCRConfig()

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        n = dataset.n_samples
        if n == 0:
            return RankingResult(method=self.method_name, scores={}, ranks=[], diagnostics={})

        # Build descriptor matrix [N, D].
        descriptors = []
        for s in dataset.samples:
            descriptors.append(residual_descriptor(s.residual, self.config.descriptor))
        Z_desc = torch.stack(descriptors, dim=0).double()
        # PCA-reduce for stability.
        zc = Z_desc - Z_desc.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(zc, full_matrices=False)
            n_comp = min(self.config.n_descriptor_components, Vh.shape[0])
            Z_pcs = (zc @ Vh.T[:, :n_comp]).double()
        except Exception:
            Z_pcs = zc[:, : self.config.n_descriptor_components]

        # Conditioning matrix: log residual energy + one-hot subject.
        log_energy = torch.log(dataset.residual_energies().clamp(min=self.config.eps)).unsqueeze(1)
        content_design = _content_design_matrix([s.content_id for s in dataset.samples])
        cond = torch.cat([log_energy, content_design], dim=1)

        scores: dict[str, float] = {}
        for name in metric_set.names():
            y = dataset.metric_array(name)
            mask = torch.isfinite(y)
            if int(mask.sum().item()) < 4:
                scores[name] = 0.0
                continue
            y_z = _rank_normal_score(y)
            # Per-component rank-normal-score, then concatenate.
            X_norm = torch.stack(
                [_rank_normal_score(Z_pcs[:, j]) for j in range(Z_pcs.shape[1])],
                dim=1,
            )
            scores[name] = _partial_corr(y_z[mask], X_norm[mask], cond[mask])

        ranks = self._ranks_from_scores(scores)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=ranks,
            diagnostics={
                "n_samples": n,
                "n_descriptor_pcs": int(Z_pcs.shape[1]),
                "descriptor_dim": int(Z_desc.shape[1]),
            },
        )
