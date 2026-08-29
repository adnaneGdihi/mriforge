"""Normalizing-flow building blocks (Glow [Kingma & Dhariwal 2018]).

Canonical home for the reusable invertible primitives used by the
``glow`` and ``equivariant_flow`` generative models:

- :class:`~mriforge.models.blocks.flow.actnorm.ActNorm2d`
- :class:`~mriforge.models.blocks.flow.invertible_1x1_conv.InvertibleConv1x1LU`
- :class:`~mriforge.models.blocks.flow.affine_coupling.AffineCoupling`
- :class:`~mriforge.models.blocks.flow.squeeze.Squeeze2x`,
  :class:`~mriforge.models.blocks.flow.squeeze.Split`

Each block implements an exact forward/inverse pair plus the
log-Jacobian-determinant term required for the change-of-variables
likelihood.
"""

from __future__ import annotations

from mriforge.models.blocks.flow.actnorm import ActNorm2d
from mriforge.models.blocks.flow.affine_coupling import AffineCoupling
from mriforge.models.blocks.flow.invertible_1x1_conv import InvertibleConv1x1LU
from mriforge.models.blocks.flow.squeeze import Split, Squeeze2x

__all__ = [
    "ActNorm2d",
    "AffineCoupling",
    "InvertibleConv1x1LU",
    "Split",
    "Squeeze2x",
]
