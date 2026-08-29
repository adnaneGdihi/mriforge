"""Functional Severity-Profile Decomposition Ranker.

Treats each metric's response surface across the joint severity hypercube
as a function in ``L^2([0, 1]^D)`` and decomposes the *set* of metrics
under FPCA. The first eigenfunction ``xi_1`` is the "severity-aligned"
mode (monotone increase with overall severity); its loadings give the
ranking. Higher-order modes describe artifact-specific or contradictory
metric clusters.

Implementation:

1. Group samples by ``degradation_family`` and ``content_id``.
2. For each metric, build an [F, S] matrix of values per family x severity.
3. Stack across families to get [K, F*S] — one row per metric.
4. Center, eigendecompose ``K K^T`` (Gram matrix).
5. Loading on mode ``alpha`` is ``a_{k, alpha}``. Sign-fix so ``xi_1`` is
   monotone increasing in severity, then report
   ``a_{k,1}^2 / sum a_{k,alpha}^2`` x sign(a_{k,1}).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker


@dataclass
class FSPDConfig:
    n_modes_to_keep: int = 6
    # Within-content centring before the family x severity average (critique §4).
    # The metric-wise (column) centring of F_c removes metric offsets but leaves
    # content-specific offsets in every row; if content composition is unbalanced
    # across severity cells, the leading mode v1 can align with content (anatomy)
    # rather than severity. Subtracting each content's own mean response first
    # residualises that confound so the decomposition tracks degradation.
    residualize_content: bool = False


def _dataset_grid(
    dataset: MetricEvaluationDataset,
) -> tuple[list[str], list[float]]:
    """Return the dataset-wide (families, severities) grid.

    Used so every metric's response matrix has the same shape, even when
    some metrics return all-NaN — otherwise stacking the rows for FPCA
    fails with a shape mismatch.
    """
    families = sorted({s.degradation_family for s in dataset.samples})
    severities = sorted({float(s.severity.get("theta", 0.0)) for s in dataset.samples})
    return families, severities


def _build_response_matrix(
    dataset: MetricEvaluationDataset,
    metric_name: str,
    families: list[str],
    severities: list[float],
    residualize_content: bool = False,
) -> torch.Tensor:
    """Aggregate per-(family, severity) by averaging across content IDs.

    The shape is fixed by the dataset-wide ``families`` and ``severities``
    so rows from different metrics can be stacked unconditionally.

    When ``residualize_content`` is set, each content's own mean response is
    subtracted from its samples before the per-cell average (critique §4): this
    removes content-specific baseline offsets so the decomposition tracks
    severity rather than anatomy under unbalanced content composition.
    """
    raw = [
        (s.degradation_family, float(s.severity.get("theta", 0.0)), s.content_id, float(v))
        for s, v in zip(dataset.samples, dataset.metric_values[metric_name], strict=True)
        if v == v  # skip NaN
    ]
    if residualize_content:
        content_sum: dict[str, float] = defaultdict(float)
        content_n: dict[str, int] = defaultdict(int)
        for _fam, _theta, content, value in raw:
            content_sum[content] += value
            content_n[content] += 1
        content_mean = {c: content_sum[c] / content_n[c] for c in content_sum}
        raw = [(fam, theta, c, value - content_mean[c]) for fam, theta, c, value in raw]

    by_key: dict[tuple[str, float], list[float]] = defaultdict(list)
    for fam, theta, _content, value in raw:
        by_key[(fam, theta)].append(value)
    out = torch.zeros(len(families), len(severities), dtype=torch.float64)
    for i, fam in enumerate(families):
        for j, theta in enumerate(severities):
            xs = by_key.get((fam, theta), [])
            out[i, j] = float(sum(xs) / max(len(xs), 1)) if xs else float("nan")
    return out


class FSPDRanker(BaseRanker):
    method_name = "FSPD"

    def __init__(self, config: FSPDConfig | None = None) -> None:
        self.config = config or FSPDConfig()

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return RankingResult(method=self.method_name, scores={}, ranks=[], diagnostics={})

        names = metric_set.names()
        if not names:
            return RankingResult(method=self.method_name, scores={}, ranks=[], diagnostics={})

        common_families, common_severities = _dataset_grid(dataset)
        rows = []
        per_metric_surfaces: dict[str, torch.Tensor] = {}
        for name in names:
            mat = _build_response_matrix(
                dataset,
                name,
                common_families,
                common_severities,
                residualize_content=self.config.residualize_content,
            )
            per_metric_surfaces[name] = mat
            # Standardize per-row by removing per-family mean to suppress
            # global metric-scale differences. NaNs (no data for a cell) are
            # zero-filled so they don't dominate the SVD.
            normed = mat.clone()
            mean_per_fam = torch.nanmean(normed, dim=1, keepdim=True)
            mean_per_fam[~torch.isfinite(mean_per_fam)] = 0.0
            normed = normed - mean_per_fam
            normed = torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize to unit Frobenius norm so loadings are comparable.
            n = float(normed.norm().item())
            if n > 1e-12:
                normed = normed / n
            rows.append(normed.flatten())

        # [K, P] matrix.
        if not rows:
            return RankingResult(method=self.method_name, scores={}, ranks=[], diagnostics={})
        F = torch.stack(rows, dim=0)
        center = F - F.mean(dim=0, keepdim=True)

        # SVD: center = U S V^T. Loadings a_{k, alpha} = (U S)_{k, alpha}.
        try:
            U, S, _Vh = torch.linalg.svd(center, full_matrices=False)
        except Exception:
            return RankingResult(
                method=self.method_name,
                scores=dict.fromkeys(names, 0.0),
                ranks=names,
                diagnostics={},
            )
        loadings = U * S.unsqueeze(0)
        K = min(self.config.n_modes_to_keep, loadings.shape[1])
        loadings = loadings[:, :K]

        # Sign-fix mode 1 to be "monotone increase in severity". This is only
        # meaningful when the leading mode actually tracks severity; if mode 1 is
        # ~orthogonal to the severity ramp (it captures a family-contrast / content
        # confound rather than a monotone trend), its sign — and therefore the FSPD
        # score's sign — is arbitrary, so we report no severity signal instead of an
        # arbitrarily-signed score (critique M5).
        mode1_cos = 1.0
        severity_aligned = True
        if loadings.shape[1] > 0:
            v1 = _Vh[0].clone()
            n_fam = len(common_families)
            n_sev = len(common_severities)
            if n_fam * n_sev == v1.numel():
                surface = v1.reshape(n_fam, n_sev)
                avg_curve = surface.mean(dim=0).double()
                ramp = torch.linspace(-1, 1, n_sev).double()
                proj = float((avg_curve * ramp).sum().item())
                cos_denom = float((avg_curve.norm() * ramp.norm()).item())
                mode1_cos = proj / cos_denom if cos_denom > 1e-12 else 0.0
                if abs(mode1_cos) < 0.05:
                    severity_aligned = False
                elif proj < 0:
                    loadings[:, 0] = -loadings[:, 0]

        # Score: fraction of energy in mode 1 with sign of first loading.
        total_energy = (loadings**2).sum(dim=1).clamp(min=1e-30)
        mode1_energy = loadings[:, 0] ** 2
        sign1 = torch.sign(loadings[:, 0])
        scores_t = sign1 * (mode1_energy / total_energy)
        if not severity_aligned:
            scores_t = torch.zeros_like(scores_t)
        scores = {n: float(scores_t[i].item()) for i, n in enumerate(names)}

        ranks = self._ranks_from_scores(scores)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=ranks,
            diagnostics={
                "loadings": {name: loadings[i].tolist() for i, name in enumerate(names)},
                "singular_values": S.tolist(),
                "n_modes": K,
                "families": common_families,
                "severities": common_severities,
                "mode1_severity_cosine": mode1_cos,
                "mode1_severity_aligned": severity_aligned,
            },
        )
