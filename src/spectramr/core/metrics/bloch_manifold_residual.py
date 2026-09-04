r"""Bloch-manifold-residual metric (Idea 2 / B-2).

Per-voxel L2 distance from a reconstruction to its projection onto the
Bloch manifold :math:`\mathcal M_{\text{Bloch}}(\mathcal P)`:

.. math::

    \mathrm{BMR}(\hat x) =
      \bigl\lVert \hat x - P_{\mathcal M}(\hat x) \bigr\rVert_2

reduced to a scalar (mean over voxels). It measures how *physically
realisable* a reconstruction is under the configured pulse sequence: a
reconstruction whose voxel signals are all explainable by some
:math:`(M_0, T_1, T_2)` lies on the manifold and scores ~0; hallucinated
or noise-dominated content pushes the residual up.

Lower is better. The projection reuses
:class:`~spectramr.models.physics.bloch_manifold_projector.BlochManifoldProjector`
(the same operator the DPS sampler interleaves), so the metric and the
sampler agree by construction. The metric ignores its ``target`` argument
(it is reference-free) — it is kept in the signature only to satisfy the
``__call__(prediction, target, **kwargs)`` metric contract.
"""

from __future__ import annotations

import logging

import torch

from spectramr.core.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    name="bloch_manifold_residual",
    aliases=["bloch_residual", "manifold_residual_l2"],
)
class BlochManifoldResidual:
    r"""Reference-free metric: mean per-voxel distance to the Bloch manifold.

    Args:
        pulse_sequence: Pulse-sequence label forwarded to the projector.
        n_inner_iterations: Gauss-Newton inner iterations for the per-voxel
            fit (kept small — this runs at validation time).
        param_bounds: Optional ``{T1,T2,M0: (lo,hi)}`` box bounds.
        normalize: When True, divide the residual by the prediction's RMS
            magnitude so the score is scale-invariant (comparable across
            differently-normalised reconstructions).

    The projector is constructed lazily on first call (so importing the
    metric module is cheap and does not allocate the Bloch layer).
    """

    def __init__(
        self,
        pulse_sequence: str = "spgr",
        n_inner_iterations: int = 8,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        normalize: bool = True,
    ) -> None:
        self.pulse_sequence = pulse_sequence
        self.n_inner_iterations = int(n_inner_iterations)
        self.param_bounds = param_bounds
        self.normalize = bool(normalize)
        self._projector = None  # lazy

    def _get_projector(self):
        if self._projector is None:
            # Imported here to keep the metric module import light and to
            # avoid a hard models<-core edge at import time.
            from spectramr.models.physics.bloch_manifold_projector import (
                BlochManifoldProjector,
            )

            self._projector = BlochManifoldProjector(
                pulse_sequence=self.pulse_sequence,
                param_bounds=self.param_bounds,
                n_inner_iterations=self.n_inner_iterations,
            )
        return self._projector

    @torch.no_grad()
    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **_: object,
    ) -> float:
        """Compute the scalar Bloch-manifold residual of ``prediction``.

        ``target`` is accepted (and ignored) for metric-contract
        compatibility — the score is reference-free.
        """
        if prediction.ndim not in (4, 5):
            raise ValueError(
                "bloch_manifold_residual expects a [B,C,H,W] (or [B,C,H,W,D]) "
                f"prediction; got shape {tuple(prediction.shape)}."
            )
        projector = self._get_projector()
        res = projector.residual(prediction)  # [B,1,H,W]
        mean_res = res.mean()
        if self.normalize:
            mag = prediction.abs() if torch.is_complex(prediction) else prediction
            rms = torch.sqrt((mag.to(torch.float32) ** 2).mean()).clamp_min(1e-8)
            mean_res = mean_res / rms
        return float(mean_res.item())

    @property
    def name(self) -> str:
        return "bloch_manifold_residual"

    @property
    def higher_is_better(self) -> bool:
        return False


__all__ = ["BlochManifoldResidual"]
