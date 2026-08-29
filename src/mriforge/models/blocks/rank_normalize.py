r"""Rank-normalize block: exact monotone-contrast invariance (MCGI core).

The per-image empirical-CDF rank transform

.. math::

   R(\mathbf x)(v) = F_{\mathbf x}\bigl(\mathbf x(v)\bigr)

is *exactly* invariant to the group :math:`G_+` of strictly increasing intensity
remaps, because the empirical CDF transforms covariantly,
:math:`F_{g\circ\mathbf x}(g(t)) = F_{\mathbf x}(t)`, so evaluating at the image
gives identical ranks: :math:`R(g\circ\mathbf x) = R(\mathbf x)`. An encoder
fronted by :math:`R` therefore factors through the :math:`G_+`-orbit and is
contrast-agnostic by construction (for anatomy-defined tasks; invariant features
discard absolute-contrast information).

Two paths:

- **hard** (inference / the guarantee): exact integer ranks via double argsort,
  mapped to ``[0, 1]``. Bit-exact invariance.
- **soft** (training): a differentiable surrogate
  :math:`r_i = \frac1N\sum_j \sigma\!\bigl((x_i - x_j)/\tau\bigr)`, which
  converges to the hard rank as :math:`\tau \to 0`. It is :math:`O(N^2)` in the
  number of voxels, so use it on patches / small maps and switch to the hard
  rank at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rank_normalize(
    x: torch.Tensor,
    *,
    hard: bool = True,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Per-(batch, channel) empirical-CDF rank transform to ``[0, 1]``.

    Args:
        x: Input ``[B, C, *spatial]``. Ranks are computed within each
            ``(batch, channel)`` map over the flattened spatial dims.
        hard: Exact double-argsort ranks (bit-exact ``G_+`` invariance). When
            ``False``, the differentiable soft-rank surrogate is used.
        temperature: Soft-rank temperature ``tau`` (smaller → closer to hard).

    Returns:
        Tensor of the same shape as ``x``, valued in ``[0, 1]``.
    """
    b, c = x.shape[0], x.shape[1]
    flat = x.reshape(b, c, -1)
    n = flat.shape[-1]
    denom = max(n - 1, 1)

    if hard:
        order = flat.argsort(dim=-1)
        ranks = torch.empty_like(order)
        arange = torch.arange(n, device=x.device).expand_as(order)
        ranks.scatter_(-1, order, arange)
        out = ranks.to(x.dtype) / denom
    else:
        # r_i = (1/N) sum_j sigmoid((x_i - x_j) / tau); pairwise, O(N^2).
        diff = flat.unsqueeze(-1) - flat.unsqueeze(-2)  # [B, C, N, N]
        out = torch.sigmoid(diff / temperature).mean(dim=-1)  # [B, C, N]

    return out.reshape_as(x)


class RankNormalize2d(nn.Module):
    """``nn.Module`` wrapper over :func:`rank_normalize`.

    Uses the soft rank in ``train()`` mode (so gradients flow to the front end)
    and the exact hard rank in ``eval()`` mode when ``hard_eval`` is set, giving
    the by-construction invariance guarantee at inference.

    Args:
        temperature: Soft-rank temperature used in training mode.
        hard_eval: Use the exact hard rank in eval mode.
    """

    def __init__(self, temperature: float = 0.05, hard_eval: bool = True) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.hard_eval = bool(hard_eval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_hard = self.hard_eval and not self.training
        return rank_normalize(x, hard=use_hard, temperature=self.temperature)


__all__ = ["RankNormalize2d", "rank_normalize"]
