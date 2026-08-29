r"""Pushforward maximum-mean-discrepancy (Proposal 1, unpaired branch).

For *unpaired* data the identified operator is fitted by matching the
pushforward of the clean marginal through the twin degradation operator,
:math:`\mathcal D_\#\,p_{\text{clean}}`, against the observed degraded
marginal :math:`p_{\text{obs}}`, measured by a characteristic-kernel
maximum mean discrepancy:

.. math::

    \mathrm{MMD}^2_{\mathcal H}(P, Q) =
        \mathbb E_{x,x'}\,k(x,x')
      + \mathbb E_{y,y'}\,k(y,y')
      - 2\,\mathbb E_{x,y}\,k(x,y).

The kernel is a mixture of Gaussians (multiple bandwidths), which is
characteristic on :math:`\mathbb R^d`, so a zero MMD implies equal
distributions. This term composes additively with
``structured_gaussian_nll`` through the strategy weight :math:`\lambda`.

References
----------
A. Gretton *et al.*, "A kernel two-sample test," *JMLR*, 13, 2012.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.losses.registry import register_loss

__all__ = ["PushforwardMMD"]


def _as_feature_matrix(x: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B, C, H, W]`` (real or complex) to a real feature matrix ``[B, D]``.

    Complex inputs are mapped to magnitude, the natural reconstruction-domain
    observable for low-field MRI degraded marginals.
    """
    if torch.is_complex(x):
        x = x.abs()
    return x.reshape(x.shape[0], -1)


@register_loss(name="pushforward_mmd", domain="image")
class PushforwardMMD(nn.Module):
    r"""Squared MMD between the pushforward and observed degraded marginals.

    Args:
        sigmas: Gaussian-kernel bandwidths (the mixture). When ``None``,
            a median-heuristic bandwidth (with multiplicative spread) is
            used per call.
        unbiased: Use the unbiased U-statistic (drops the diagonal) when
            both batches have at least two samples.
    """

    def __init__(
        self,
        sigmas: tuple[float, ...] | None = (0.5, 1.0, 2.0, 4.0, 8.0),
        unbiased: bool = True,
    ) -> None:
        super().__init__()
        self.sigmas = sigmas
        self.unbiased = unbiased

    def _kernel_sum(
        self,
        d2: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        drop_diagonal: bool,
    ) -> torch.Tensor:
        """Mean of the mixture-RBF kernel over a pairwise squared-distance matrix."""
        k = torch.zeros_like(d2)
        for s in sigmas:
            k = k + torch.exp(-d2 / (2.0 * s * s))
        n = d2.shape[0]
        if drop_diagonal and n > 1:
            off = k.sum() - torch.diagonal(k).sum()
            return off / (n * (n - 1))
        return k.mean()

    def _cross_mean(self, d2: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        k = torch.zeros_like(d2)
        for s in sigmas:
            k = k + torch.exp(-d2 / (2.0 * s * s))
        return k.mean()

    def forward(self, generated: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathrm{MMD}^2` between two image batches.

        Args:
            generated: Pushforward (twin-degraded) samples ``[B, C, H, W]``.
            observed: Observed degraded samples ``[M, C, H, W]``.

        Returns:
            Scalar non-negative squared MMD.
        """
        x = _as_feature_matrix(generated)
        y = _as_feature_matrix(observed)
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"feature dim mismatch: generated {x.shape[1]} vs observed {y.shape[1]}."
            )

        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)

        if self.sigmas is not None:
            sigmas = torch.tensor(self.sigmas, device=x.device, dtype=x.dtype)
        else:
            # Median heuristic from the cross distances, spread over a small grid.
            med = torch.median(xy.detach().sqrt()).clamp_min(1e-6)
            sigmas = med * torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], device=x.device, dtype=x.dtype)

        drop = self.unbiased and x.shape[0] > 1 and y.shape[0] > 1
        k_xx = self._kernel_sum(xx, sigmas, drop_diagonal=drop)
        k_yy = self._kernel_sum(yy, sigmas, drop_diagonal=drop)
        k_xy = self._cross_mean(xy, sigmas)
        mmd2 = k_xx + k_yy - 2.0 * k_xy
        # The unbiased estimator can dip slightly negative; clamp at 0.
        return mmd2.clamp_min(0.0)
