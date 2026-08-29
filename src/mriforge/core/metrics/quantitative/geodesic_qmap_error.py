"""Geodesic-qMap error metric (Idea 5).

The geometrically correct analogue of Euclidean MAE on (M0, T1, T2)
maps. Computes the Fisher-metric geodesic distance between predicted
and ground-truth parameter triples at every voxel, then reduces to a
mean.
"""

from __future__ import annotations

import torch

from mriforge.config.schemas.enums import Regime, Task
from mriforge.core.metrics.registry import register_metric
from mriforge.infrastructure.physics.manifolds import BlochRelaxationManifold


@register_metric(
    "geodesic_qmap_error",
    aliases=["GeodesicQMapMAE"],
    workflows=frozenset({Regime.QUANTITATIVE}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class GeodesicQMapErrorMetric:
    """Mean geodesic distance on the Bloch relaxation manifold.

    Tagged ``mri_quantitative``: the channel axis is read as ``(M0, T1, T2)`` and
    the distance is taken on the relaxation manifold, so this grades a
    quantitative parameter map and nothing else.
    """

    def __init__(self) -> None:
        self.manifold = BlochRelaxationManifold()
        self._name = "geodesic_qmap_error"

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
        # Expected layout [B, 3, H, W] — channels (M0, T1, T2).
        p = prediction.permute(0, 2, 3, 1).reshape(-1, 3)
        q = target.permute(0, 2, 3, 1).reshape(-1, 3)
        d = self.manifold.geodesic_distance(p, q)
        return float(d.mean().item())


__all__ = ["GeodesicQMapErrorMetric"]
