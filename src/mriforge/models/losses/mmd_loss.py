r"""Maximum Mean Discrepancy (MMD) — kernel-based two-sample distance.

For two empirical distributions :math:`P, Q` with samples
:math:`\{x_i\}, \{y_j\}` and a positive-definite kernel :math:`k`,

.. math::
    \mathrm{MMD}^2(P, Q) =
        \tfrac{1}{n(n-1)}\sum_{i\ne j} k(x_i, x_j)
      + \tfrac{1}{m(m-1)}\sum_{i\ne j} k(y_i, y_j)
      - \tfrac{2}{nm}\sum_{i,j} k(x_i, y_j).

In MRI MMD is the canonical *unbalanced* harmonisation /
domain-alignment loss: it forces feature distributions across two
sites or two acquisition protocols to match without requiring paired
data, and unlike adversarial alignment it has a closed-form gradient
and no min-max instabilities.

This implementation uses a sum-of-Gaussian-kernels (multi-bandwidth)
:math:`k(x,y) = \sum_\sigma \exp(-\|x-y\|^2/(2\sigma^2))` which
empirically performs best.

Reference:
    A. Gretton *et al.*, "A kernel two-sample test," *JMLR*, 13, 2012.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.losses.registry import register_loss


def _multi_kernel_mmd(x: torch.Tensor, y: torch.Tensor, sigmas: tuple[float, ...]) -> torch.Tensor:
    """Squared MMD with sum-of-Gaussian-kernels (unbiased estimator)."""
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"MMD needs at least 2 samples per side, got n={n}, m={m}.")
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    k_xx = torch.zeros_like(xx)
    k_yy = torch.zeros_like(yy)
    k_xy = torch.zeros_like(xy)
    for sigma in sigmas:
        denom = 2.0 * sigma * sigma
        k_xx = k_xx + torch.exp(-xx / denom)
        k_yy = k_yy + torch.exp(-yy / denom)
        k_xy = k_xy + torch.exp(-xy / denom)
    eye_n = torch.eye(n, device=x.device, dtype=x.dtype)
    eye_m = torch.eye(m, device=y.device, dtype=y.dtype)
    sum_xx = (k_xx * (1 - eye_n)).sum() / max(n * (n - 1), 1)
    sum_yy = (k_yy * (1 - eye_m)).sum() / max(m * (m - 1), 1)
    sum_xy = k_xy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy


@register_loss(name="mmd", aliases=["MMDLoss", "maximum_mean_discrepancy"], domain="agnostic")
class MMDLoss(nn.Module):
    r"""Multi-bandwidth Gaussian-kernel MMD.

    Forward signature: ``prediction``/``target`` are flattened per-batch
    to ``(N, D)`` feature vectors before computing MMD. They do *not*
    need to be paired and may have different batch sizes.

    Args:
        sigmas: Bandwidths for the sum-of-Gaussians kernel. Default
            :math:`(1, 2, 4, 8, 16)` covers a broad scale range.
        flatten_spatial: If True (default) flatten ``(B, C, H, W) →
            (B, C·H·W)`` before computing MMD. If False the input must
            already be ``(B, D)``.
    """

    def __init__(
        self,
        sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
        flatten_spatial: bool = True,
    ) -> None:
        super().__init__()
        if not sigmas:
            raise ValueError("sigmas must contain at least one bandwidth.")
        if any(s <= 0 for s in sigmas):
            raise ValueError(f"all sigmas must be > 0, got {sigmas}")
        self.sigmas = tuple(sigmas)
        self.flatten_spatial = flatten_spatial

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.flatten_spatial:
            x = prediction.reshape(prediction.shape[0], -1)
            y = target.reshape(target.shape[0], -1)
        else:
            x, y = prediction, target
        return _multi_kernel_mmd(x, y, self.sigmas)
