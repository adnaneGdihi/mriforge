r"""Tropical (max-plus) geometry primitives for MRF & quantitative mapping.

A *tropical polynomial* in :math:`d` variables over the max-plus semiring
:math:`(\mathbb{R}\cup\{-\infty\}, \max, +)` is

.. math::

    p(\mathbf{r}) = \max_{i\in I}\,\bigl\{\langle a_i,\mathbf{r}\rangle + b_i\bigr\},

with :math:`a_i\in\mathbb{R}^d`, :math:`b_i\in\mathbb{R}`, :math:`|I|=K`.
The set of such functions is *exactly* the set of continuous piecewise-affine
convex functions on :math:`\mathbb{R}^d` with at most :math:`K` linear pieces
[Maclagan & Sturmfels 2015, §2].

Two applications in this codebase:

* **Tropical MRF Bloch dictionary** (Batch-1 #1) — under the EPG short-TR
  small-flip-angle expansion, the log-magnitude signal at readout
  :math:`n` is a tropical polynomial in
  :math:`(\log T_1, \log T_2, \log M_0)`. Matching reduces to locating
  the cone of the polyhedral fan that contains the measurement, which is
  logarithmic in :math:`K` (vs linear-in-dictionary-size Euclidean
  matched filtering).
* **Tropical piecewise-smooth quantitative maps** (Batch-2 #9) — :math:`T_1`,
  :math:`T_2`, :math:`M_0` maps are piecewise-smooth with sharp tissue
  boundaries. Parameterising the map as a difference of two tropical
  polynomials :math:`p=p_+-p_-` (spans *all* continuous piecewise-affine
  functions [Zhang-Naitzat-Lim 2018]) yields a representation that is
  boundary-sharp by construction and exposes the implicit segmentation as
  the common refinement of two polytope complexes.

The module exports:

* :func:`tropical_polynomial` — evaluate :math:`p(\mathbf{r})` at points.
* :func:`log_sum_exp_tropical` — soft (differentiable) approximation
  :math:`p_\beta=\frac{1}{\beta}\log\sum_i\exp(\beta(\langle a_i,\mathbf{r}\rangle+b_i))`
  converging to :math:`p` as :math:`\beta\to\infty`.
* :func:`active_cone_indices` — which monomial achieves the max at each point
  (the cone classifier :math:`\sigma_n` from the MRF proposal).
* :class:`TropicalRationalMap` — learnable :math:`p_+ - p_-` parameterisation
  for piecewise-affine quantitative maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "TropicalRationalMap",
    "active_cone_indices",
    "log_sum_exp_tropical",
    "tropical_polynomial",
]


def tropical_polynomial(
    points: torch.Tensor, slopes: torch.Tensor, intercepts: torch.Tensor
) -> torch.Tensor:
    r"""Evaluate the tropical polynomial :math:`\max_i\langle a_i,\mathbf{r}\rangle+b_i`.

    Args:
        points: ``(..., d)`` tensor of evaluation points.
        slopes: ``(K, d)`` matrix of monomial slopes :math:`a_i`.
        intercepts: ``(K,)`` vector of monomial intercepts :math:`b_i`.

    Returns:
        ``(...,)`` tensor giving :math:`\max_i\langle a_i,\mathbf{r}\rangle+b_i`.
    """
    if slopes.ndim != 2:
        raise ValueError(f"slopes must be (K, d); got {tuple(slopes.shape)}.")
    if intercepts.shape != (slopes.shape[0],):
        raise ValueError(
            f"intercepts must have shape (K,) = ({slopes.shape[0]},); "
            f"got {tuple(intercepts.shape)}."
        )
    if points.shape[-1] != slopes.shape[1]:
        raise ValueError(f"points have dim {points.shape[-1]}, slopes have dim {slopes.shape[1]}.")
    affine = points @ slopes.T + intercepts  # (..., K)
    return affine.max(dim=-1).values


def log_sum_exp_tropical(
    points: torch.Tensor,
    slopes: torch.Tensor,
    intercepts: torch.Tensor,
    beta: float = 16.0,
) -> torch.Tensor:
    r"""Soft (differentiable) tropical polynomial via log-sum-exp.

    :math:`p_\beta(\mathbf{r})=\frac{1}{\beta}\log\sum_i\exp(\beta(\langle a_i,\mathbf{r}\rangle+b_i))`.
    Converges to :math:`\max_i` as :math:`\beta\to\infty` and is everywhere
    differentiable for finite :math:`\beta`.

    Args:
        points: ``(..., d)``.
        slopes: ``(K, d)``.
        intercepts: ``(K,)``.
        beta: Sharpness :math:`\beta>0`; default 16.0 (corresponds to a
            piecewise-affine approximation accurate to :math:`\log K/\beta`).

    Returns:
        ``(...,)`` soft-max-plus values.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive; got {beta}.")
    affine = points @ slopes.T + intercepts
    return torch.logsumexp(beta * affine, dim=-1) / beta


def active_cone_indices(
    points: torch.Tensor, slopes: torch.Tensor, intercepts: torch.Tensor
) -> torch.Tensor:
    r"""Index :math:`i^*(\mathbf{r})=\arg\max_i\langle a_i,\mathbf{r}\rangle+b_i`.

    The mapping :math:`\mathbf{r}\mapsto i^*` partitions :math:`\mathbb{R}^d`
    into the top cones of the tropical-polynomial polyhedral fan; it is the
    cone classifier :math:`\sigma_n` used by the tropical-MRF proposal to
    pre-locate the dictionary entry before regression.

    Args:
        points: ``(..., d)``.
        slopes: ``(K, d)``.
        intercepts: ``(K,)``.

    Returns:
        ``(...,)`` integer tensor of indices in :math:`\{0,1,\ldots,K-1\}`.
    """
    affine = points @ slopes.T + intercepts
    return affine.argmax(dim=-1)


class TropicalRationalMap(nn.Module):
    r"""Difference-of-tropical-polynomials parameterisation :math:`p=p_+-p_-`.

    Spans *all* continuous piecewise-affine functions on :math:`\mathbb{R}^d`
    [Zhang-Naitzat-Lim 2018, Prop. 2], not just convex ones. The implicit
    segmentation is the common refinement of the two polytope complexes
    defined by :math:`p_+` and :math:`p_-`.

    Args:
        in_dim: Coordinate dimension :math:`d`.
        out_dim: Number of independent piecewise-affine maps to produce
            (e.g. 3 for joint :math:`T_1, T_2, M_0` regression).
        n_monomials_pos: Number of monomials :math:`K_+` in :math:`p_+`.
        n_monomials_neg: Number of monomials :math:`K_-` in :math:`p_-`.
        beta: Log-sum-exp sharpness for differentiable training; ``None``
            uses the hard max-plus at inference. Default 16.0.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_monomials_pos: int = 32,
        n_monomials_neg: int = 16,
        beta: float = 16.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive.")
        if n_monomials_pos <= 0 or n_monomials_neg <= 0:
            raise ValueError("n_monomials_pos and n_monomials_neg must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.beta = beta
        # Slopes and intercepts for the K_+ and K_- monomials, one bank per
        # output channel.
        scale = 1.0 / max(1.0, in_dim**0.5)
        self.slopes_pos = nn.Parameter(torch.randn(out_dim, n_monomials_pos, in_dim) * scale)
        self.intercepts_pos = nn.Parameter(torch.zeros(out_dim, n_monomials_pos))
        self.slopes_neg = nn.Parameter(torch.randn(out_dim, n_monomials_neg, in_dim) * scale)
        self.intercepts_neg = nn.Parameter(torch.zeros(out_dim, n_monomials_neg))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        r"""Evaluate :math:`p(\mathbf{r})` per output channel.

        Args:
            points: ``(..., d)`` evaluation coordinates.

        Returns:
            ``(..., out_dim)`` piecewise-affine map values.
        """
        if points.shape[-1] != self.in_dim:
            raise ValueError(f"points last dim {points.shape[-1]} != in_dim {self.in_dim}.")
        # affine[..., c, k] = <slopes_pos[c, k], points> + intercepts_pos[c, k]
        affine_pos = torch.einsum("ckd,...d->...ck", self.slopes_pos, points)
        affine_pos = affine_pos + self.intercepts_pos
        affine_neg = torch.einsum("ckd,...d->...ck", self.slopes_neg, points)
        affine_neg = affine_neg + self.intercepts_neg
        p_pos = torch.logsumexp(self.beta * affine_pos, dim=-1) / self.beta
        p_neg = torch.logsumexp(self.beta * affine_neg, dim=-1) / self.beta
        return p_pos - p_neg

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"K_pos={self.slopes_pos.shape[1]}, K_neg={self.slopes_neg.shape[1]}, "
            f"beta={self.beta}"
        )
