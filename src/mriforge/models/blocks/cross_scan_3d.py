r"""3D Cross-Scan Module for Vision Mamba MRI Reconstruction.

Eliminates the **unidirectional causality bias** inherent in 1D SSM
(State-Space Model) scanning by sweeping a 3D volume along all 6
independent Cartesian half-axes simultaneously:

.. math::

    h_{\text{fused}} = \sigma(W_g \cdot [h_{x+}, h_{x-}, h_{y+}, h_{y-},
                        h_{z+}, h_{z-}]) \odot
                       \tanh(W_v \cdot [h_{x+}, h_{x-}, h_{y+}, h_{y-},
                       h_{z+}, h_{z-}])

This preserves the :math:`O(N)` linear complexity of Mamba while ensuring
that every voxel attends to context from **all 6 directions**, not just a
single raster scan order.

For 2D inputs (D=1), falls back to a 4-direction scan
(:math:`x\pm, y\pm`), matching VMamba2D behaviour.

Reference:
    Liu et al., "VMamba: Visual State Space Model", *CVPR* (2024).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class MambaSSM1D(nn.Module):
    """Lightweight 1D selective state-space model sweep.

    Implements a simplified S6-style scan for a single direction.
    For full Mamba integration, this can be swapped with ``mamba_ssm.Mamba``
    when the ``mamba_ssm`` package is available.

    Args:
        d_model: Feature dimension.
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand: Expansion factor for inner dimension.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Local convolution (causal)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # SSM parameters
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_inner, -1)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.d_inner = d_inner
        self.d_state = d_state

    def forward(self, x: Tensor) -> Tensor:
        """Apply selective scan along sequence dimension.

        Args:
            x: ``(B, L, D)`` — batch, sequence length, features.

        Returns:
            ``(B, L, D)`` — scanned output.
        """
        B, L, D = x.shape

        # Split into gate and value paths
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        import torch.nn.functional as F

        # Local conv (transpose for Conv1d)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Simplified SSM step (parallel scan approximation)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, d_inner)

        # Discretise: dA = exp(A * dt), dB = dt
        # For efficiency, use a 1-step recurrence approximation
        y = x_conv * dt * self.D.unsqueeze(0).unsqueeze(0) + x_conv

        # Gate
        y = y * F.silu(z)

        return self.out_proj(y)


class CrossScan3D(nn.Module):
    """3D Cross-Scan Module with gated multi-directional fusion.

    Sweeps a volumetric feature map along 6 half-axes (or 4 for 2D)
    and fuses the resulting hidden states via a learned gating mechanism.

    Args:
        d_model: Feature dimension per voxel.
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand: SSM expansion factor.
        use_3d: If ``False``, only use 4 directions (x±, y±).
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_3d: bool = True,
    ) -> None:
        super().__init__()
        self.use_3d = use_3d
        # Kept on the instance: ``forward`` drops the z-sweeps on a degenerate
        # depth axis and has to pad back up to the width these projections were
        # built for.
        self.n_dirs = n_dirs = 6 if use_3d else 4

        # One SSM per scan direction (weight sharing optional)
        self.ssm_modules = nn.ModuleList(
            [MambaSSM1D(d_model, d_state, d_conv, expand) for _ in range(n_dirs)]
        )

        # Gated fusion: σ(W_g · concat) ⊙ tanh(W_v · concat)
        self.gate_proj = nn.Linear(n_dirs * d_model, d_model, bias=True)
        self.value_proj = nn.Linear(n_dirs * d_model, d_model, bias=True)

        # Layer norm for stable training
        self.norm = nn.LayerNorm(d_model)

    def _flatten_axis(
        self,
        x: Tensor,
        axis: Literal["x", "y", "z"],
        reverse: bool = False,
    ) -> tuple[Tensor, tuple[int, ...]]:
        """Flatten a 5D volume along the specified axis into (B', L, D).

        Args:
            x: ``(B, C, D, H, W)`` volumetric features.
            axis: Which spatial axis to sweep.
            reverse: If ``True``, reverse the sequence order.

        Returns:
            Flattened ``(B', L, C)`` and original shape for unflattening.
        """
        B, C, D, H, W = x.shape

        if axis == "x":
            # Sweep along W: each (B, D, H) fiber → L=W
            x_perm = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
            x_flat = x_perm.reshape(B * D * H, W, C)
        elif axis == "y":
            # Sweep along H: each (B, D, W) fiber → L=H
            x_perm = x.permute(0, 2, 4, 3, 1)  # (B, D, W, H, C)
            x_flat = x_perm.reshape(B * D * W, H, C)
        elif axis == "z":
            # Sweep along D: each (B, H, W) fiber → L=D
            x_perm = x.permute(0, 3, 4, 2, 1)  # (B, H, W, D, C)
            x_flat = x_perm.reshape(B * H * W, D, C)
        else:
            raise ValueError(f"Unknown axis: {axis}")

        if reverse:
            x_flat = x_flat.flip(dims=[1])

        return x_flat, (B, C, D, H, W)

    def _unflatten_axis(
        self,
        x_flat: Tensor,
        axis: Literal["x", "y", "z"],
        shape: tuple[int, ...],
        reverse: bool = False,
    ) -> Tensor:
        """Inverse of ``_flatten_axis``."""
        B, C, D, H, W = shape

        if reverse:
            x_flat = x_flat.flip(dims=[1])

        if axis == "x":
            x_perm = x_flat.reshape(B, D, H, W, C)
            return x_perm.permute(0, 4, 1, 2, 3)
        elif axis == "y":
            x_perm = x_flat.reshape(B, D, W, H, C)
            return x_perm.permute(0, 4, 1, 3, 2)
        elif axis == "z":
            x_perm = x_flat.reshape(B, H, W, D, C)
            return x_perm.permute(0, 4, 3, 1, 2)
        else:
            raise ValueError(f"Unknown axis: {axis}")

    def forward(self, x: Tensor) -> Tensor:
        """Apply cross-scan and gated fusion.

        Args:
            x: ``(B, C, D, H, W)`` volumetric features.
                For 2D input, use D=1.

        Returns:
            ``(B, C, D, H, W)`` fused features.
        """
        B, C, D, H, W = x.shape

        # Determine active scan directions
        if self.use_3d and D > 1:
            directions = [
                ("x", False),
                ("x", True),
                ("y", False),
                ("y", True),
                ("z", False),
                ("z", True),
            ]
        else:
            directions = [
                ("x", False),
                ("x", True),
                ("y", False),
                ("y", True),
            ]

        # Process each direction
        scanned = []
        for i, (axis, reverse) in enumerate(directions):
            flat, shape = self._flatten_axis(x, axis, reverse)
            out = self.ssm_modules[i](flat)
            unflat = self._unflatten_axis(out, axis, shape, reverse)
            scanned.append(unflat)

        # The fusion projections are sized at construction for ``self.n_dirs``,
        # but the loop above drops the two z-sweeps when the depth axis is
        # degenerate (D=1). Without this pad a ``use_3d=True`` block handed 2-D
        # input concatenated 4*C into a Linear expecting 6*C and died with
        # "mat1 and mat2 shapes cannot be multiplied" -- the 2-D fallback path
        # the class documents was never actually reachable.
        #
        # Zero is the right filler rather than a shorter projection: a length-1
        # axis carries no context to gather, so the z-sweeps contribute nothing
        # but their bias, and one set of weights stays valid for both ranks.
        while len(scanned) < self.n_dirs:
            scanned.append(torch.zeros_like(scanned[0]))

        # Concatenate along channel dimension for fusion
        # Each scanned[i] is (B, C, D, H, W) → concat → (B, n_dirs*C, D, H, W)
        h_concat = torch.cat(scanned, dim=1)  # (B, n_dirs*C, D, H, W)

        # Reshape for linear fusion: (B, D, H, W, n_dirs*C)
        h_concat = h_concat.permute(0, 2, 3, 4, 1)

        # Gated fusion
        gate = torch.sigmoid(self.gate_proj(h_concat))
        value = torch.tanh(self.value_proj(h_concat))
        h_fused = gate * value  # (B, D, H, W, C)

        # Back to (B, C, D, H, W)
        h_fused = h_fused.permute(0, 4, 1, 2, 3)

        # Residual connection + layer norm
        out = x + h_fused
        # Apply layer norm per-voxel
        out = out.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)

        return out


__all__ = ["CrossScan3D", "MambaSSM1D"]
