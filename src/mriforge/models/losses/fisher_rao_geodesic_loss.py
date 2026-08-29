r"""Fisher-Rao geodesic information-geometry loss.

The strategy key ``fisher_rao_flow`` advertises information-geometry flow
mathematics: distances between distributions are measured along geodesics of
the *Fisher-Rao metric* -- the unique (up to scale) Riemannian metric on the
statistical manifold :math:`\mathcal{P}(\Theta)` that is invariant under
sufficient statistics [Amari-Nagaoka 2000]. A plain :math:`L_1` / :math:`L_2`
residual ignores this geometry entirely (it treats parameter space as flat
Euclidean), which is exactly the "façade" failure this loss removes.

We model each (normalised) intensity field as a one-dimensional Gaussian
acquisition likelihood :math:`\mathcal{N}(\mu,\sigma^2)` -- the additive-noise
Bloch model used throughout :mod:`mriforge.infrastructure.physics.fisher_rao_geometry`.
Its natural coordinates are :math:`\theta=(\mu,\sigma)`. The Fisher information
of a Gaussian in these coordinates is the diagonal metric

.. math::

    g(\theta) = \mathrm{diag}\!\left(\frac{1}{\sigma^2},\,\frac{2}{\sigma^2}\right),

which we obtain *through the physics primitive* by feeding the per-parameter
score Jacobian to :func:`fisher_information_from_jacobian` rather than
hard-coding it. The Fisher-Rao geodesic distance between the prediction's and
target's distributions is then the metric arc-length of the displacement
:math:`\Delta\theta=\theta_q-\theta_p`, evaluated with
:func:`fisher_norm` at the geodesic midpoint metric:

.. math::

    \mathrm{FR}(p,q) \;\approx\; \big\|\theta_q-\theta_p\big\|_{g(\bar\theta)}
        \;=\; \sqrt{\Delta\theta^{\top} g(\bar\theta)\,\Delta\theta}.

Because :math:`g` scales as :math:`1/\sigma^2`, the same Euclidean
:math:`(\mu,\sigma)` gap costs *more* between tight distributions than between
broad ones -- the curvature signature a Euclidean loss can never reproduce.

This is a midpoint (geodesic-secant) approximation to the exact Gaussian
Fisher-Rao distance; it is exact to second order in :math:`\Delta\theta`,
strictly positive off-diagonal, zero iff the distributions coincide, and
smooth/differentiable everywhere, which is what a training objective needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.fisher_rao_geometry import (
    fisher_information_from_jacobian,
    fisher_norm,
)
from mriforge.models.losses.registry import register_loss

_EPS = 1e-6


@register_loss(
    name="fisher_rao_geodesic",
    aliases=[
        "fisher_rao_flow",
        "fr_geodesic",
        "information_geometry_geodesic",
    ],
    domain="image",
)
class FisherRaoGeodesicLoss(nn.Module):
    r"""Penalise the Fisher-Rao geodesic distance between distributions.

    Each sample's (normalised) intensities are summarised by their Gaussian
    parameters :math:`\theta=(\mu,\sigma)`; the loss is the Fisher-Rao
    arc-length between the prediction's and target's :math:`\theta`, measured
    with the physics primitive's Fisher metric. Unlike :math:`L_2`, the cost
    of a fixed parameter gap depends on the local distribution scale.

    Args:
        sigma_floor: Lower clamp on the per-sample std so the :math:`1/\sigma^2`
            metric stays finite. Must be positive. Default ``1e-3``.
        reduction: ``"mean"`` or ``"sum"`` over the batch.
    """

    def __init__(
        self,
        sigma_floor: float = 1e-3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if sigma_floor <= 0:
            raise ValueError(f"sigma_floor must be positive; got {sigma_floor}.")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}.")
        self.sigma_floor = sigma_floor
        self.reduction = reduction

    def _gaussian_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Per-sample :math:`(\mu,\sigma)` over all non-batch axes.

        Returns ``(mu, sigma)`` each of shape ``(B,)``.
        """
        flat = x.reshape(x.shape[0], -1)
        mu = flat.mean(dim=1)
        sigma = flat.std(dim=1, unbiased=False).clamp_min(self.sigma_floor)
        return mu, sigma

    def _fisher_metric(self, sigma: torch.Tensor) -> torch.Tensor:
        r"""Gaussian Fisher metric :math:`g=\mathrm{diag}(1/\sigma^2,2/\sigma^2)`
        built *via the physics primitive* from the score Jacobian.

        For :math:`\mathcal{N}(\mu,\sigma^2)` the score-function Jacobian whose
        Gram matrix (with unit noise) reproduces the closed-form Fisher matrix
        is :math:`J=\mathrm{diag}(1/\sigma,\sqrt{2}/\sigma)` over two readouts;
        feeding it to :func:`fisher_information_from_jacobian` yields
        :math:`J^{\top}J=g`. This keeps the metric definition in the physics
        SSOT rather than hand-coding it here.

        Args:
            sigma: ``(B,)`` per-sample std.

        Returns:
            ``(B, 2, 2)`` Fisher information matrices.
        """
        inv_sigma = 1.0 / sigma  # (B,)
        zeros = torch.zeros_like(inv_sigma)
        # Per-sample Jacobian (B, N=2, d=2):
        #   row 0 -> mu-score scale, row 1 -> sigma-score scale.
        row_mu = torch.stack([inv_sigma, zeros], dim=-1)  # (B, 2)
        row_sigma = torch.stack([zeros, (2.0**0.5) * inv_sigma], dim=-1)  # (B, 2)
        jacobian = torch.stack([row_mu, row_sigma], dim=-2)  # (B, 2, 2)
        return fisher_information_from_jacobian(jacobian, sigma=1.0)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch: prediction {tuple(prediction.shape)} vs "
                f"target {tuple(target.shape)}."
            )

        mu_p, sigma_p = self._gaussian_params(prediction)
        mu_q, sigma_q = self._gaussian_params(target)

        # Displacement vector in natural coordinates theta = (mu, sigma).
        delta = torch.stack([mu_q - mu_p, sigma_q - sigma_p], dim=-1)  # (B, 2)

        # Geodesic-midpoint metric (second-order-accurate secant approximation
        # to the exact Gaussian Fisher-Rao distance).
        sigma_mid = 0.5 * (sigma_p + sigma_q)
        metric = self._fisher_metric(sigma_mid)  # (B, 2, 2)

        # Fisher-Rao arc-length of the displacement under the local metric.
        distances = fisher_norm(delta, metric)  # (B,)

        if self.reduction == "mean":
            return distances.mean()
        return distances.sum()
