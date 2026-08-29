"""Local Implicit Neural Representations and Patch Coordination.

This module implements the logic for patching volumes and querying Local INRs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from mriforge.models.registry import register_model

from .disentangled_encoder import DisentangledEncoder
from .hypernetwork import Hypernetwork, HyperSIREN


class PatchCoordinator(nn.Module):
    """Handles coordinate transforms and patching."""

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        overlap: float = 0.25,
    ):
        """__init__.

        Args:
            volume_shape (tuple[int, int, int]): Description.
            patch_size (tuple[int, int, int]): Description.
            overlap (float): Description.
        """
        super().__init__()
        self.volume_shape = volume_shape
        self.patch_size = patch_size
        self.overlap = overlap

        # Calculate strides
        self.strides = [int(p * (1 - overlap)) for p in patch_size]

        # Generate patch centers/starts
        self.starts = []
        for d in range(3):
            size = volume_shape[d]
            p_size = patch_size[d]
            stride = self.strides[d]

            dim_starts = list(range(0, size - p_size + 1, stride))
            if dim_starts[-1] + p_size < size:
                dim_starts.append(size - p_size)
            self.starts.append(dim_starts)

        self.patches_grid = []
        for x in self.starts[0]:
            for y in self.starts[1]:
                for z in self.starts[2]:
                    self.patches_grid.append((x, y, z))

    def crop_patches(self, volume: Float[Tensor, "B C D H W"]) -> Float[Tensor, "B*P C Dp Hp Wp"]:
        """Crop volume into patches."""
        B, C, D, H, W = volume.shape
        patches = []

        pD, pH, pW = self.patch_size

        for sx, sy, sz in self.patches_grid:
            # Note: volume order usually D, H, W
            # self.starts are computed for dim 0, 1, 2 of spatial dims
            patch = volume[:, :, sx : sx + pD, sy : sy + pH, sz : sz + pW]
            patches.append(patch)

        # Stack: [P, B, C, ...] -> [B*P, C, ...]
        # We interleave: Patch1_B1, Patch1_B2, ... Patch2_B1...
        # Ideally [B, P, C, ...] flattened to [B*P, C, ...]
        # Here we stack list to [P, B, C, ...]

        patches_stack = torch.stack(patches, dim=1)  # [B, P, C, D, H, W]
        patches_stack = patches_stack.flatten(0, 1)  # [B*P, C, D, H, W]

        return patches_stack

    def global_to_local(
        self, points: Float[Tensor, "B N 3"], patch_idx: Int[Tensor, B]
    ) -> Float[Tensor, "B N 3"]:
        """Convert global [0, 1] coords to local [-1, 1] coords.

        Args:
            points: Global coordinates in [0, 1] range, shape [B, N, 3]
            patch_idx: Index into self.patches_grid for each batch item, shape [B]

        Returns:
            Local coordinates in [-1, 1] range for the specified patch
        """
        B, N, _ = points.shape
        device = points.device

        # Get patch starts for each batch item
        patch_starts = torch.tensor(
            [self.patches_grid[idx.item()] for idx in patch_idx],
            device=device,
            dtype=points.dtype,
        )  # [B, 3]

        # Convert volume_shape and patch_size to tensors
        vol_shape = torch.tensor(self.volume_shape, device=device, dtype=points.dtype)
        p_size = torch.tensor(self.patch_size, device=device, dtype=points.dtype)

        # Normalize patch starts to [0, 1] range
        patch_min = patch_starts / vol_shape  # [B, 3]
        patch_extent = p_size / vol_shape  # [3]

        # Expand for broadcasting: [B, 1, 3]
        patch_min = patch_min.unsqueeze(1)

        # Transform: local = 2 * (global - patch_min) / patch_extent - 1
        local_coords = 2.0 * (points - patch_min) / patch_extent - 1.0

        return local_coords


@register_model(name="local_inr", training_mode="reconstruction")
class LocalINR(nn.Module):
    """Single Local INR module (Encoder + Hypernet + SIREN)."""

    def __init__(
        self,
        in_channels: int = 1,
        z_geo_dim: int = 256,
        z_phy_dim: int = 64,
        hidden_features: int = 64,
        hidden_layers: int = 3,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            z_geo_dim (int): Description.
            z_phy_dim (int): Description.
            hidden_features (int): Description.
            hidden_layers (int): Description.
        """
        super().__init__()

        self.encoder = DisentangledEncoder(
            in_channels=in_channels,
            anatomy_dim=z_geo_dim,
            physics_dim=z_phy_dim,
        )

        self.siren = HyperSIREN(hidden_features=hidden_features, hidden_layers=hidden_layers)

        self.hypernet = Hypernetwork(
            z_geo_dim=z_geo_dim, z_phy_dim=z_phy_dim, hyper_siren=self.siren
        )

    def forward(
        self,
        local_volume: Float[Tensor, "B C D H W"],
        coords: Float[Tensor, "B N 3"],
        contrast_volume: Float[Tensor, "B C D H W"] | None = None,
    ) -> Float[Tensor, "B N 1"]:
        """Forward pass for a batch of local patches.

        Args:
            local_volume: Input patches (defines geometry).
            coords: Coordinates to query in local frame [-1, 1].
            contrast_volume: Volume defining contrast (defines physics).
               If None, uses local_volume (self-reco).
               If provided, we use local_volume for z_geo and contrast_volume for z_phy.

        forward method for LocalINR.

        Executes PyTorch tensor operations.

        Args:
            local_volume (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            contrast_volume (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B N 1']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode Geometry
        z_geo, _ = self.encoder(local_volume)

        # Encode Physics
        target_vol = contrast_volume if contrast_volume is not None else local_volume
        _, z_phy = self.encoder(target_vol)

        # Predict Weights
        params = self.hypernet(z_geo, z_phy)

        # Synthesize
        output = self.siren(coords, params)

        return output


class INRVolume(nn.Module):
    """Full Volume INR manager."""

    def __init__(
        self,
        volume_shape: tuple[int, int, int] = (128, 128, 128),
        patch_size: tuple[int, int, int] = (32, 32, 32),
        in_channels: int = 1,
        hidden_features: int = 64,
    ):
        """__init__.

        Args:
            volume_shape (tuple[int, int, int]): Description.
            patch_size (tuple[int, int, int]): Description.
            in_channels (int): Description.
            hidden_features (int): Description.
        """
        super().__init__()
        self.coordinator = PatchCoordinator(volume_shape, patch_size)
        self.local_inr = LocalINR(in_channels=in_channels, hidden_features=hidden_features)

    # Full reconstruction logic would involve sweeping patches, querying grids,
    # and stitching (using overlap blending).
    # For now, we expose the ability to train on patches.

    def training_step_patches(
        self,
        volume: Float[Tensor, "B C D H W"],
        valid_mask: Float[Tensor, "B C D H W"] | None = None,
    ) -> Float[Tensor, ""]:
        """Extract random patches and train reconstruction."""
        # Crop all patches
        patches = self.coordinator.crop_patches(volume)  # [B*P, C, Dp, Hp, Wp]

        # Generate random coordinates within patches [-1, 1]
        B_P = patches.shape[0]
        N_points = 1024
        coords = torch.rand(B_P, N_points, 3, device=volume.device) * 2 - 1

        # Ground truth sampling (grid_sample)
        # Note: grid_sample expects (x,y,z) in [-1, 1]
        # patches is [B*P, C, D, H, W]
        # grid shape required: [B*P, N, 1, 1, 3] for 5D input
        grid = coords.view(B_P, N_points, 1, 1, 3)
        gt_values = F.grid_sample(patches, grid, align_corners=True)  # [B*P, C, N, 1, 1]
        gt_values = gt_values.view(B_P, N_points, 1)

        # Reconstruct
        pred_values = self.local_inr(patches, coords)

        loss = F.mse_loss(pred_values, gt_values)
        return loss
