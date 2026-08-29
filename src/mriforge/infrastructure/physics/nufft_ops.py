"""NUFFT Operations — Golden-Angle Radial Trajectory & Differentiable Forward Model.

Wraps ``torchkbnufft`` to provide a clean interface for non-Cartesian MRI
simulation and reconstruction.  All operations are fully differentiable,
enabling end-to-end optimisation of implicit representations (MR-NeRF)
directly against raw k-space data.

Trajectory Generation:
    Golden-angle radial sampling uses the angle increment
    :math:`\\Delta\\theta = 111.246°` (golden angle) to provide nearly
    uniform k-space coverage with any number of spokes.

Reference:
    Winkelmann et al., "An Optimal Radial Profile Order Based on the
    Golden Ratio," *IEEE TMI*, 2007.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Radial-MRI golden angle in radians: π·(√5 − 1)/2 ≈ 1.9416 rad ≈ 111.246°
# (Winkelmann et al. 2007). NOTE: this is the radial golden angle, NOT the 2-D
# phyllotaxis angle π·(3 − √5) ≈ 137.508° — the constant previously held the
# latter while every docstring claimed 111.246° (see kspace_trajectory_generator).
GOLDEN_ANGLE = math.pi * (math.sqrt(5.0) - 1.0) / 2.0


class GoldenAngleTrajectory(nn.Module):
    """Generate golden-angle radial k-space trajectories.

    Each spoke is rotated by the golden angle from the previous, ensuring
    near-uniform coverage regardless of the total number of spokes.

    Args:
        im_size: Image dimensions ``(H, W)``.
        num_spokes: Number of radial spokes per frame.
        samples_per_spoke: Readout samples per spoke.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        num_spokes: int = 128,
        samples_per_spoke: int = 256,
    ) -> None:
        super().__init__()
        self.im_size = im_size
        self.num_spokes = num_spokes
        self.samples_per_spoke = samples_per_spoke

    def generate(
        self,
        num_frames: int = 1,
        spoke_offset: int = 0,
    ) -> Tensor:
        """Generate k-space trajectory coordinates.

        Args:
            num_frames: Number of temporal frames.
            spoke_offset: Starting spoke index (for multi-frame sequences).

        Returns:
            Trajectory ``[num_frames, 2, num_spokes * samples_per_spoke]``
            with coordinates in ``[-π, π]`` (torchkbnufft convention).
        """
        all_trajectories = []

        for frame in range(num_frames):
            ktraj_list = []
            for spoke_idx in range(self.num_spokes):
                global_spoke = spoke_offset + frame * self.num_spokes + spoke_idx
                angle = global_spoke * GOLDEN_ANGLE

                # Radial samples from -π to π
                r = torch.linspace(-math.pi, math.pi, self.samples_per_spoke)
                kx = r * math.cos(angle)
                ky = r * math.sin(angle)
                ktraj_list.append(torch.stack([kx, ky], dim=0))

            # [2, num_spokes * samples_per_spoke]
            ktraj = torch.cat(ktraj_list, dim=1)
            all_trajectories.append(ktraj)

        return torch.stack(all_trajectories, dim=0)  # [T, 2, N]

    def forward(
        self,
        num_frames: int = 1,
        spoke_offset: int = 0,
    ) -> Tensor:
        """Alias for :meth:`generate`.

        forward method for GoldenAngleTrajectory.

        Executes PyTorch tensor operations.

        Args:
            num_frames (int): Expected input tensor.
            spoke_offset (int): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.generate(num_frames, spoke_offset)


class NUFFTForwardModel(nn.Module):
    """Differentiable Non-Uniform FFT forward/adjoint model.

    Wraps ``torchkbnufft`` for forward projection (image → k-space) and
    adjoint (k-space → image) operations on non-Cartesian trajectories.

    Args:
        im_size: Image dimensions ``(H, W)``.
        grid_size_factor: Oversampling factor for NUFFT grid.
        numpoints: Interpolation kernel width.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        grid_size_factor: float = 2.0,
        numpoints: int = 6,
    ) -> None:
        super().__init__()
        self.im_size = im_size

        import torchkbnufft as tkbn

        grid_size = tuple(int(s * grid_size_factor) for s in im_size)
        self.nufft_forward = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, numpoints=numpoints)
        self.nufft_adjoint = tkbn.KbNufftAdjoint(
            im_size=im_size, grid_size=grid_size, numpoints=numpoints
        )

        logger.info(
            "[NUFFTForwardModel] im_size=%s, grid=%s, numpoints=%d",
            im_size,
            grid_size,
            numpoints,
        )

    def forward_project(
        self,
        image: Tensor,
        trajectory: Tensor,
    ) -> Tensor:
        """Project image to non-Cartesian k-space.

        Args:
            image: Complex image ``[B, C, H, W]`` (``torch.complex64``).
            trajectory: K-space coordinates ``[B, 2, N]`` or ``[2, N]``.

        Returns:
            K-space data ``[B, C, N]`` (complex).
        """
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0).expand(image.shape[0], -1, -1)

        return self.nufft_forward(image, trajectory)

    def adjoint_project(
        self,
        kdata: Tensor,
        trajectory: Tensor,
    ) -> Tensor:
        """Adjoint NUFFT: k-space → image.

        Args:
            kdata: K-space data ``[B, C, N]`` (complex).
            trajectory: K-space coordinates ``[B, 2, N]`` or ``[2, N]``.

        Returns:
            Image ``[B, C, H, W]`` (complex).
        """
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0).expand(kdata.shape[0], -1, -1)

        return self.nufft_adjoint(kdata, trajectory)

    def forward(
        self,
        image: Tensor,
        trajectory: Tensor,
    ) -> Tensor:
        """Forward pass — alias for :meth:`forward_project`.

        forward method for NUFFTForwardModel.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.forward_project(image, trajectory)


__all__ = ["GOLDEN_ANGLE", "GoldenAngleTrajectory", "NUFFTForwardModel"]
