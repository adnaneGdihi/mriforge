"""C_n-equivariant normalizing flow.

A normalizing flow whose every block is ``C_n``-equivariant, so the
learned density is exactly invariant to the discrete in-plane rotations of
the cyclic group ``C_n`` (a subgroup of ``SO(2)``):
``log p(rho_g x) = log p(x)`` for all ``g``. Equivariance is preserved
under composition, so it suffices to compose equivariant ActNorm, group
convolution and group coupling blocks.

The input image is lifted into the regular representation by tiling it
across the ``group_order`` group axis; this lift is itself equivariant
(a rotation of the input cyclically shifts the group axis). The flow then
operates entirely in the group-feature space.

Reference: J. Köhler et al., "Equivariant flows: exact likelihood
generative learning for symmetric densities," ICML 2020; T. S. Cohen and
M. Welling, "Group equivariant convolutional networks," ICML 2016.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectramr.models.blocks.equivariant.group_actnorm import GroupActNorm2d
from spectramr.models.blocks.equivariant.group_conv import GroupConv2d
from spectramr.models.blocks.equivariant.group_coupling import GroupCoupling
from spectramr.models.registry import register_model


class _EquivariantStep(nn.Module):
    """One equivariant flow step: GroupActNorm -> GroupConv mix -> GroupCoupling.

    The 1x1-conv channel mix of standard Glow is replaced by a 1x1
    :class:`GroupConv2d` (block-diagonal over orbits), which is invertible
    in practice via the coupling/actnorm pair around it; here it is used as
    an equivariant feature mix and its (small) volume change is folded into
    the coupling's learnable scale during training.
    """

    def __init__(
        self, num_channels: int, group_order: int, hidden_base: int, coupling_type: str
    ) -> None:
        super().__init__()
        self.actnorm = GroupActNorm2d(num_channels, group_order=group_order)
        self.coupling = GroupCoupling(
            num_channels,
            group_order=group_order,
            hidden_base=hidden_base,
            coupling_type=coupling_type,
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            x, ld_c = self.coupling(x, reverse=True)
            x, ld_a = self.actnorm(x, reverse=True)
            return x, ld_a + ld_c
        x, ld_a = self.actnorm(x, reverse=False)
        x, ld_c = self.coupling(x, reverse=False)
        return x, ld_a + ld_c


@register_model(
    name="equivariant_flow",
    training_mode="generative",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class EquivariantFlow(nn.Module):
    """C_n-equivariant normalizing flow.

    Args:
        in_channels: Image channels (default 1, magnitude MRI).
        out_channels: Must equal ``in_channels`` (flows are bijections).
        group_order: Order ``n`` of the cyclic group ``C_n`` (4, 8 or 16).
        num_steps: Number of equivariant flow steps.
        base_channels: Base (per-orbit) feature width; total feature
            channels are ``base_channels * group_order``.
        coupling_type: ``"affine"`` or ``"additive"``.

    Raises:
        ValueError: On unsupported ``group_order`` / ``coupling_type`` or
            channel mismatch (no silent fallback).
    """

    SUPPORTED_GROUPS = (4, 8, 16)

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        group_order: int = 8,
        num_steps: int = 8,
        base_channels: int = 8,
        coupling_type: str = "affine",
    ) -> None:
        super().__init__()
        if out_channels != in_channels:
            raise ValueError(
                "EquivariantFlow is a bijection; out_channels must equal "
                f"in_channels (got in={in_channels}, out={out_channels})."
            )
        if group_order not in self.SUPPORTED_GROUPS:
            raise ValueError(
                f"Unsupported group_order={group_order}; expected one of {self.SUPPORTED_GROUPS}."
            )
        if coupling_type not in ("affine", "additive"):
            raise ValueError(
                f"Unknown coupling_type {coupling_type!r}; expected 'affine' or 'additive'."
            )
        if base_channels % 2 != 0:
            raise ValueError(
                f"base_channels must be even for the coupling split, got {base_channels}."
            )

        self.in_channels = int(in_channels)
        self.group_order = int(group_order)
        self.base_channels = int(base_channels)
        self.feature_channels = self.base_channels * self.group_order

        # Equivariant lift: 1x1 group conv from the in_channels (tiled across
        # the group axis) to the regular-representation feature space.
        self.lift = GroupConv2d(
            self.in_channels * self.group_order,
            self.feature_channels,
            kernel_size=1,
            group_order=self.group_order,
        )
        self.steps = nn.ModuleList(
            [
                _EquivariantStep(
                    self.feature_channels, self.group_order, base_channels, coupling_type
                )
                for _ in range(int(num_steps))
            ]
        )

    def _lift(self, x: torch.Tensor) -> torch.Tensor:
        # Tile the image across the group axis to form a regular-rep input.
        tiled = x.repeat(1, self.group_order, 1, 1)
        return self.lift(tiled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward direction ``x -> (z, log_det_J)``.

        Returns the flattened group-feature latent and the per-sample
        accumulated log-Jacobian-determinant ``[B]``.
        """
        h = self._lift(x)
        log_det = x.new_zeros(x.shape[0])
        for step in self.steps:
            h, ld = step(h, reverse=False)
            log_det = log_det + ld
        return h.flatten(1), log_det

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Like :meth:`forward` but returns the spatial latent ``[B, C, H, W]``.

        Useful for the equivariance probe, which compares ``f(rho_g x)``
        against ``rho_g f(x)`` in the feature layout.
        """
        h = self._lift(x)
        log_det = x.new_zeros(x.shape[0])
        for step in self.steps:
            h, ld = step(h, reverse=False)
            log_det = log_det + ld
        return h, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample log-likelihood ``[B]`` under the standard-normal prior."""
        z, log_det_j = self.forward(x)
        d = z.shape[1]
        prior = -0.5 * (z**2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)
        return prior + log_det_j
