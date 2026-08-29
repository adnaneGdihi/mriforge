r"""Poincaré-ball hyperbolic operations for hierarchical embeddings.

The Poincaré ball :math:`\mathbb{B}^d_c=\{\mathbf{z}\in\mathbb{R}^d:c\|\mathbf{z}\|^2<1\}`
of curvature :math:`-c<0` admits a Riemannian metric for which Möbius
operations are isometries. Sarkar's theorem (1971; Nickel-Kiela 2017)
states that any tree of depth :math:`D` embeds with distortion
:math:`1+\varepsilon` into :math:`\mathbb{B}^d_c` for :math:`d=\Omega(\log D)`,
exponentially better than Euclidean — making the Poincaré ball the right
representational geometry for hierarchical signals (e.g. cardiac-cycle →
systole/diastole → sub-phases).

This module exposes the standard hyperbolic primitives:

* :func:`exp_map_zero` — exponential map at the origin
  :math:`\exp_0^c(\mathbf{v})=\tanh(\sqrt c\|\mathbf{v}\|)\mathbf{v}/(\sqrt c\|\mathbf{v}\|)`.
* :func:`log_map_zero` — its inverse.
* :func:`mobius_add` — Möbius addition :math:`\oplus_c`.
* :func:`poincare_distance` — geodesic distance
  :math:`d(\mathbf{u},\mathbf{v})=(2/\sqrt c)\,\mathrm{artanh}(\sqrt c\|-\mathbf{u}\oplus_c\mathbf{v}\|)`.
* :class:`PoincareLayer` — Euclidean → Poincaré projection module.
* :func:`hyperbolic_tree_distance_loss` — auxiliary loss penalising
  cardiac sub-tree neighbours.

Reduction at :math:`c\to 0^+`: all Möbius operations degenerate to Euclidean
ones, so the network smoothly recovers the standard Euclidean baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "PoincareLayer",
    "exp_map_zero",
    "hyperbolic_tree_distance_loss",
    "log_map_zero",
    "mobius_add",
    "poincare_distance",
]


def _safe_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return x.norm(dim=-1, keepdim=True).clamp_min(eps)


def exp_map_zero(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    r"""Exponential map at the origin
    :math:`\exp_0^c(\mathbf{v})=\tanh(\sqrt c\|\mathbf{v}\|)\,\mathbf{v}/(\sqrt c\|\mathbf{v}\|)`.

    Maps Euclidean tangent vectors at :math:`\mathbf{0}` into the Poincaré
    ball of curvature :math:`-c`.
    """
    if c < 0:
        raise ValueError("c must be non-negative; use c=0 for Euclidean limit.")
    if c == 0:
        return v
    norm_v = _safe_norm(v)
    return torch.tanh(c**0.5 * norm_v) * v / (c**0.5 * norm_v)


def log_map_zero(z: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    r"""Logarithmic map at the origin (inverse of :func:`exp_map_zero`).

    .. math::

        \log_0^c(\mathbf{z}) = \mathrm{artanh}(\sqrt c\|\mathbf{z}\|)\,
                               \mathbf{z}/(\sqrt c\|\mathbf{z}\|).
    """
    if c < 0:
        raise ValueError("c must be non-negative.")
    if c == 0:
        return z
    norm_z = _safe_norm(z)
    # Clamp norm to keep argument of artanh in (-1, 1).
    arg = (c**0.5 * norm_z).clamp(max=1.0 - 1e-5)
    return torch.atanh(arg) * z / (c**0.5 * norm_z)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    r"""Möbius addition :math:`\mathbf{x}\oplus_c\mathbf{y}`.

    .. math::

        \mathbf{x}\oplus_c\mathbf{y} = \frac{(1+2c\langle\mathbf{x},\mathbf{y}\rangle+c\|\mathbf{y}\|^2)\mathbf{x}+(1-c\|\mathbf{x}\|^2)\mathbf{y}}{1+2c\langle\mathbf{x},\mathbf{y}\rangle+c^2\|\mathbf{x}\|^2\|\mathbf{y}\|^2}.
    """
    if c < 0:
        raise ValueError("c must be non-negative.")
    if c == 0:
        return x + y
    x_dot_y = (x * y).sum(dim=-1, keepdim=True)
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * x_dot_y + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * x_dot_y + (c**2) * x_sq * y_sq
    return num / denom.clamp_min(1e-7)


def poincare_distance(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    r"""Geodesic distance
    :math:`d_c(\mathbf{u},\mathbf{v})=(2/\sqrt c)\,\mathrm{artanh}(\sqrt c\|-\mathbf{u}\oplus_c\mathbf{v}\|)`.

    At :math:`c\to 0^+` this reduces to the Euclidean distance
    :math:`\|\mathbf{u}-\mathbf{v}\|`.
    """
    if c < 0:
        raise ValueError("c must be non-negative.")
    if c == 0:
        return (u - v).norm(dim=-1)
    diff = mobius_add(-u, v, c=c)
    norm_diff = _safe_norm(diff).squeeze(-1)
    arg = (c**0.5 * norm_diff).clamp(max=1.0 - 1e-5)
    return (2.0 / c**0.5) * torch.atanh(arg)


class PoincareLayer(nn.Module):
    r"""Euclidean → Poincaré projection module.

    Pipeline: Euclidean linear → :func:`exp_map_zero` → Möbius bias →
    Poincaré-ball output. At :math:`c=0` reduces to a standard linear layer.

    Args:
        in_features: Input Euclidean dimension.
        out_features: Output Poincaré dimension.
        curvature: Negative curvature parameter :math:`c\ge 0`. Default 1.0.
        bias: Include a Möbius bias term. Default True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        curvature: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if curvature < 0:
            raise ValueError("curvature must be non-negative.")
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        z = exp_map_zero(h, c=self.curvature)
        if self.bias is not None:
            z = mobius_add(z, self.bias.expand_as(z), c=self.curvature)
        return z


def hyperbolic_tree_distance_loss(
    embeddings: torch.Tensor,
    sibling_mask: torch.Tensor,
    c: float = 1.0,
) -> torch.Tensor:
    r"""Pull together hierarchical siblings in the Poincaré ball.

    .. math::

        \mathcal{L}_{\text{tree}} = \sum_{(t,t')\,:\,\text{same phase}}
            d_{\mathbb{B}^d_c}(\mathbf{e}_t,\mathbf{e}_{t'})^2.

    Args:
        embeddings: ``(T, d)`` Poincaré-ball embeddings of cardiac frames.
        sibling_mask: ``(T, T)`` boolean tensor — True iff
            :math:`(t, t')` belong to the same cardiac sub-tree.
        c: Curvature parameter.

    Returns:
        Scalar mean squared pairwise hyperbolic distance over siblings.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be (T, d); got {tuple(embeddings.shape)}.")
    if sibling_mask.shape != (embeddings.shape[0], embeddings.shape[0]):
        raise ValueError("sibling_mask must be (T, T).")
    t = embeddings.shape[0]
    # Pairwise hyperbolic distances.
    e_i = embeddings.unsqueeze(1).expand(t, t, -1)
    e_j = embeddings.unsqueeze(0).expand(t, t, -1)
    dists = poincare_distance(
        e_i.reshape(-1, embeddings.shape[1]), e_j.reshape(-1, embeddings.shape[1]), c=c
    ).reshape(t, t)
    if sibling_mask.sum() == 0:
        return embeddings.new_zeros(())
    return (dists**2)[sibling_mask].mean()
