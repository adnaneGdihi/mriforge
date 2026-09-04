"""C_n group-equivariant building blocks for equivariant flows.

These blocks implement the discrete-rotation (cyclic group ``C_n``)
analogues of the Glow flow primitives, following the group-convolution
framework of Cohen & Welling. Feature tensors carry the regular
representation of ``C_n``: channels are organised as ``(C / n) x n`` where
the ``n`` axis indexes group elements, and the group acts by cyclically
permuting that axis together with the spatial rotation.

- :class:`~spectramr.models.blocks.equivariant.group_conv.GroupConv2d`
- :class:`~spectramr.models.blocks.equivariant.group_actnorm.GroupActNorm2d`
- :class:`~spectramr.models.blocks.equivariant.group_coupling.GroupCoupling`

Reference: T. S. Cohen and M. Welling, "Group equivariant convolutional
networks," ICML 2016; J. Köhler et al., "Equivariant flows," ICML 2020.
"""

from __future__ import annotations

from spectramr.models.blocks.equivariant.group_actnorm import GroupActNorm2d
from spectramr.models.blocks.equivariant.group_conv import GroupConv2d
from spectramr.models.blocks.equivariant.group_coupling import GroupCoupling

__all__ = ["GroupActNorm2d", "GroupConv2d", "GroupCoupling"]
