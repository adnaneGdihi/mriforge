"""MRF-specific quantitative metrics for the 2026 plan.

Registered metrics:
    * ``geodesic_mrf_parameter_error`` — Fisher-geodesic distance on
      the 5-D Bloch parameter manifold (MRF §2).
    * ``cosine_preservation_score`` — agreement between embedded and
      original cosine similarities (MRF §3).
    * ``cross_scanner_t1t2_concordance`` — Lin's concordance for
      paired-scanner T1/T2 estimates (MRF §5).

References:
    [11] Amari & Nagaoka, *Methods of Information Geometry*, AMS, 2000.
    [21] Buonincontri et al., "Multi-site repeatability and
         reproducibility of MR fingerprinting", *NeuroImage*, 195, 2019.
"""

from __future__ import annotations

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.core.metrics.registry import register_metric
from spectramr.infrastructure.physics.manifolds.bloch_mrf_manifold import (
    cayley_hilbert_chart,
)


@register_metric(
    "geodesic_mrf_parameter_error",
    workflows=frozenset({Regime.FINGERPRINTING}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class GeodesicMRFParameterError:
    """Chart-space Euclidean distance on the Bloch MRF manifold.

    The Cayley-Hilbert chart is conformal at the manifold centre, so
    chart-Euclidean distance is a first-order-correct surrogate for
    the Fisher geodesic distance. Replacing this with the full
    Fisher geodesic requires the differentiable Bloch operator and
    is deferred to inference time.
    """

    def __init__(self) -> None:
        self._name = "geodesic_mrf_parameter_error"

    @property
    def name(self) -> str:
        return self._name

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> float:
        if prediction.shape != target.shape:
            raise ValueError(f"shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
        zp = cayley_hilbert_chart(prediction)
        zq = cayley_hilbert_chart(target)
        return float((zp - zq).pow(2).sum(-1).sqrt().mean().item())


@register_metric("cosine_preservation_score")
class CosinePreservationScore:
    """How well an embedding preserves angular structure.

    ``score = 1 - mean | cos∠(ψ(f_i), ψ(f_j)) - cos∠(f_i, f_j) |``
    over a random batch of fingerprint pairs. Higher is better,
    bounded in ``(−∞, 1]``.
    """

    def __init__(self) -> None:
        self._name = "cosine_preservation_score"

    @property
    def name(self) -> str:
        return self._name

    @property
    def higher_is_better(self) -> bool:
        return True

    @staticmethod
    def _pairwise_cos(x: torch.Tensor) -> torch.Tensor:
        nrm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_n = x / nrm
        return x_n @ x_n.transpose(-1, -2)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> float:
        # prediction = embedded fingerprints [B, d]; target = raw [B, N]
        cos_pred = self._pairwise_cos(prediction)
        cos_tgt = self._pairwise_cos(target)
        return float(1.0 - (cos_pred - cos_tgt).abs().mean().item())


@register_metric(
    "cross_scanner_t1t2_concordance",
    aliases=["cross_scanner_t1_t2_concordance", "CrossScannerT1T2"],
    workflows=frozenset({Regime.QUANTITATIVE}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class CrossScannerT1T2Concordance:
    """Lin's concordance correlation for paired T1/T2 across scanners.

    Tagged ``mri_quantitative``, NOT ``mri_fingerprinting``, despite living in
    this MRF module: it grades T1/T2 relaxation maps, which is a quantitative
    claim. MRF is one way to produce those maps, not what the metric measures.

    Caller supplies prediction = scanner-A estimate, target = scanner-B
    estimate. Returns the average of T1 and T2 concordance.
    """

    def __init__(self) -> None:
        self._name = "cross_scanner_t1t2_concordance"

    @property
    def name(self) -> str:
        return self._name

    @property
    def higher_is_better(self) -> bool:
        return True

    @staticmethod
    def _lin_ccc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(unbiased=False), b.var(unbiased=False)
        cov = ((a - ma) * (b - mb)).mean()
        return 2.0 * cov / (va + vb + (ma - mb).pow(2) + 1e-8)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> float:
        # Channels [B, 2, ...] in (T1, T2) order, or last-dim [..., 2].
        if prediction.shape != target.shape:
            raise ValueError("shape mismatch")
        if prediction.dim() >= 4:
            t1_pred, t2_pred = prediction[:, 0].flatten(), prediction[:, 1].flatten()
            t1_tgt, t2_tgt = target[:, 0].flatten(), target[:, 1].flatten()
        else:
            t1_pred, t2_pred = (
                prediction[..., 0].flatten(),
                prediction[..., 1].flatten(),
            )
            t1_tgt, t2_tgt = target[..., 0].flatten(), target[..., 1].flatten()
        ccc_t1 = self._lin_ccc(t1_pred, t1_tgt)
        ccc_t2 = self._lin_ccc(t2_pred, t2_tgt)
        return float((0.5 * (ccc_t1 + ccc_t2)).item())


__all__ = [
    "CosinePreservationScore",
    "CrossScannerT1T2Concordance",
    "GeodesicMRFParameterError",
]
