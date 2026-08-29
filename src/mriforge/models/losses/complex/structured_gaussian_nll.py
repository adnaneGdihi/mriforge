r"""Structured-Gaussian negative log-likelihood (Proposal 1, paired branch).

The exact Gaussian likelihood of the identified degradation operator on
*paired* data is

.. math::

    \mathcal L_{\mathrm{pair}}
      = \frac1{N_p}\sum_n\Big[r_n^{H}\,\Sigma(u_n)^{-1} r_n
                              + \log\det\Sigma(u_n)\Big],
    \qquad r_n = y_n - \exp\!\big(\Omega(s(u_n))\big)\,x_n,

where :math:`\Sigma = \mathcal F^{-1}\operatorname{diag}(\rho)\mathcal F +
U U^{H}`. Both the quadratic term and the log-determinant are evaluated
matrix-free through
:class:`mriforge.infrastructure.physics.structured_covariance.StructuredCovariance`
(FFT for the k-space-diagonal part, Woodbury / determinant-lemma for the
low-rank part).

This loss **is** the complete paired likelihood — it does not compose with
other reconstruction losses; the strategy combines it only with the
unpaired pushforward-MMD term through the strategy weight :math:`\lambda`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.structured_covariance import StructuredCovariance
from mriforge.models.losses.registry import register_loss

__all__ = ["StructuredGaussianNLL"]


@register_loss(name="structured_gaussian_nll", domain="complex")
class StructuredGaussianNLL(nn.Module):
    r"""Negative log-likelihood under the structured Gaussian noise model.

    Args:
        eps: Positivity floor added to the PSD ``rho`` inside the covariance.
        reduction: ``"mean"`` (default) or ``"sum"`` over the batch.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        residual: torch.Tensor,
        rho_grid: torch.Tensor,
        u: torch.Tensor | None,
        *,
        channels: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        r"""Evaluate :math:`\mathcal L_{\mathrm{pair}}`.

        Args:
            residual: Complex residual :math:`r = y - \exp(\Omega)x`,
                shape ``[B, C, H, W]``.
            rho_grid: Non-negative k-space PSD, ``[B, 1, H, W]`` or
                ``[B, C, H, W]``.
            u: Low-rank covariance factor ``[B, N, r]`` (``N = C*H*W``) or
                ``None`` for a pure k-space-diagonal covariance.
            channels, height, width: Operand dims; inferred from
                ``residual`` when omitted.

        Returns:
            Scalar loss (reduced over the batch).
        """
        if residual.dim() != 4:
            raise ValueError(f"residual must be [B, C, H, W], got shape {tuple(residual.shape)}.")
        _b, c, h, w = residual.shape
        channels = channels or c
        height = height or h
        width = width or w

        cov = StructuredCovariance(
            rho_grid,
            u,
            channels=channels,
            height=height,
            width=width,
            eps=self.eps,
        )
        quad = cov.quadratic_form(residual)  # [B], real
        logdet = cov.logdet()  # [B], real
        per_sample = quad + logdet
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()
