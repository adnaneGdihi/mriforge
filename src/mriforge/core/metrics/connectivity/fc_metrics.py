"""Functional-connectivity metrics on the SPD manifold (fMRI §4).

Registered metric:
    * ``geodesic_fc_error`` — mean affine-invariant geodesic distance
      between predicted and reference FC matrices.

References:
    [18] X. Pennec et al., "A Riemannian framework for tensor
         computing", *Int. J. Comput. Vis.*, 66(1), 2006.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.registry import register_metric
from mriforge.infrastructure.physics.manifolds.spd_manifold import SPDManifold


@register_metric("geodesic_fc_error", aliases=["GeodesicFCErrorMAE"])
class GeodesicFCErrorMetric:
    """Mean affine-invariant geodesic distance between SPD matrices."""

    def __init__(self, n: int = 64) -> None:
        self._n = n
        self._name = "geodesic_fc_error"

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
            raise ValueError(
                f"shape mismatch: pred {tuple(prediction.shape)} vs target {tuple(target.shape)}"
            )
        n = prediction.shape[-1]
        manifold = SPDManifold(n=n)
        d = manifold.geodesic_distance(prediction, target)
        return float(d.mean().item())


__all__ = ["GeodesicFCErrorMetric"]
