"""MedGS: Folded Gaussian Splatting for MRI."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Complex, Float
from torch import Tensor

from mriforge.infrastructure.physics.implementations.gaussian_fourier_op import (
    differentiable_gaussian_fourier,
)
from mriforge.models.registry import register_model

from .gaussian_primitives import GaussianCloud as BaseGaussianCloud


@register_model(name="medgs", training_mode="reconstruction")
class MedGS(BaseGaussianCloud):
    """Gaussian Cloud with Polynomial Trajectory (Folded Gaussians)."""

    def __init__(
        self,
        num_gaussians: int = 1000,
        volume_shape: tuple[int, int, int] = (128, 128, 128),
        poly_degree: int = 2,  # Quadratic trajectory
        **kwargs,
    ):
        """__init__.

        Args:
            num_gaussians (int): Description.
            volume_shape (tuple[int, int, int]): Description.
            poly_degree (int): Description.
        """
        base_kwargs = {k: v for k, v in kwargs.items() if k in ["scaling_modifier", "sh_degree"]}
        super().__init__(num_gaussians=num_gaussians, volume_shape=volume_shape, **base_kwargs)

        self.poly_degree = poly_degree
        c0 = self._xyz.data.clone()
        c_rest = torch.randn(num_gaussians, poly_degree, 3) * 0.01

        self._coeffs = nn.Parameter(torch.cat([c0.unsqueeze(1), c_rest], dim=1))
        del self._xyz

    def get_xyz_at_time(self, t: Float[Tensor, B]) -> Float[Tensor, "B N 3"]:
        """Compute Gaussian positions at specific times t."""
        device = self._coeffs.device
        t = t.to(device)
        powers = torch.stack([t**d for d in range(self.poly_degree + 1)], dim=1)
        xyz_t = torch.einsum("bd,ndk->bnk", powers, self._coeffs)
        return xyz_t

    def forward(
        self,
        k_coords: Float[Tensor, "K 3"],
        slice_indices: Float[Tensor, K] | None = None,
        **kwargs,
    ) -> Complex[Tensor, K]:
        """Compute FT at k_coords.

        Accepts either explicit k-space trajectory coordinates ``[K, 3]`` or a
        spatial image tensor ``[B, C, H, W]`` (the latter is passed by the
        generic pretraining strategy). When a 4D image is received the spatial
        grid is flattened and extended to 3D by appending a zero z-coordinate
        so the Gaussian Fourier operator receives a valid ``[K, 3]`` input.
        """
        device = self._coeffs.device
        k_coords = k_coords.to(device)

        # ── Shape normalisation ──────────────────────────────────────────────
        # Strategy passes image tensors [B, C, H, W]; reshape to [K, 3].
        if k_coords.dim() == 4:
            # Flatten spatial dims and take first batch item / channel
            b, c, h, w = k_coords.shape
            flat = k_coords[0, 0].reshape(-1)  # [H*W]
            # Build 2-D grid coordinates normalised to [-1, 1]
            ys = torch.linspace(-1.0, 1.0, h, device=device)
            xs = torch.linspace(-1.0, 1.0, w, device=device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            k_coords = torch.stack(
                [grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(h * w, device=device)], dim=-1
            )  # [K, 3]
        elif k_coords.dim() == 2 and k_coords.shape[-1] != 3:
            # Already flat but wrong trailing dim — treat rows as spatial points
            # and embed into 3-D by zero-padding.
            n = k_coords.shape[0] * k_coords.shape[1]
            k_coords = k_coords.reshape(n, 1)
            k_coords = torch.cat([k_coords, torch.zeros(n, 2, device=device)], dim=-1)  # [K, 3]
        # ── End shape normalisation ──────────────────────────────────────────

        if slice_indices is None:
            slice_indices = torch.zeros(k_coords.shape[0], device=device)
        else:
            slice_indices = slice_indices.to(device)

        if torch.all(slice_indices == slice_indices[0]):
            t = slice_indices[0:1]
            xyz = self.get_xyz_at_time(t).squeeze(0)  # [N, 3]
            return self._compute_ft_with_xyz(k_coords, xyz)
        else:
            xyz = self.get_xyz_at_time(slice_indices)  # [K, N, 3]
            return self._compute_ft_with_xyz(k_coords, xyz, per_k_xyz=True)

    def _compute_ft_with_xyz(
        self,
        k_coords: Float[Tensor, "K 3"],
        xyz: Float[Tensor, "... 3"],
    ) -> Complex[Tensor, K]:
        """Internal helper handling explicit xyz using centralized functional projector."""
        device = self._coeffs.device

        # Use our robust centralized function
        # This handles device movement and NaN stability internally
        res = differentiable_gaussian_fourier(
            means=xyz,
            scales=self.get_scaling,
            quaternions=self.get_rotation,
            features=self.get_opacity,  # Features are opacities for MedGS usually
            k_trajectory=k_coords,
        )

        # If per_k_xyz=True, xyz is [K, N, 3].
        # differentiable_gaussian_fourier handles broadcasting if k_trajectory is [K, 3].
        # But wait! If xyz is [K, N, 3] and k_trajectory is [K, 3].
        # our function does: k_trajectory.unsqueeze(-2) -> [K, 1, 3]
        # means.unsqueeze(-3) -> [1, K, N, 3] (if xyz is [K, N, 3])
        # This will work correctly!

        return res.squeeze(-1)  # Output is (..., K)
