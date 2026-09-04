"""NUFFT Utilities

This module contains helper functions for Non-Uniform Fast Fourier Transform (NUFFT)
operations, specifically for handling trajectory-based reconstruction.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def apply_nufft_adjoint(
    kspace: torch.Tensor,
    trajectory: torch.Tensor,
    dcf: torch.Tensor | None = None,
    im_size: tuple[int, int] = (256, 256),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Apply NUFFT Adjoint operation (K-Space -> Image).

    Handles dimension reshaping, density compensation, and torchkbnufft invocation.

    Args:
        kspace: Input k-space data (B, C, N) or (B, N)
        trajectory: K-space trajectory (B, N, 2) or (B, 2, N)
        dcf: Density Compensation Function (B, N) or (B, 1, N)
        im_size: Target image size (H, W)
        device: Device to perform operation on

    Returns:
        Reconstructed image (B, C, H, W)
    """
    try:
        import torchkbnufft as tkbn
    except ImportError:
        raise ImportError("torchkbnufft is required for non-Cartesian reconstruction.")

    device = device or kspace.device

    # Ensure dimensions are correct for tkbn
    # Handle potential extra batch dimension from collation (B, 1, ...)
    if kspace.ndim == 4 and kspace.shape[1] == 1:
        kspace = kspace.squeeze(1)

    if trajectory.ndim == 4 and trajectory.shape[1] == 1:
        trajectory = trajectory.squeeze(1)

    if dcf is not None and dcf.ndim == 3 and dcf.shape[1] == 1:
        dcf = dcf.squeeze(1)

    # kspace: (B, C, N)
    # traj: (B, N, 2) -> (B, 2, N) [Transpose needed if loaded as (N, 2)]
    if trajectory.ndim == 3 and trajectory.shape[-1] == 2:
        ktraj = trajectory.transpose(1, 2)  # (B, 2, N)
    else:
        ktraj = trajectory

    # Apply Density Compensation Function if available
    # This pre-weights k-space for better conditioning
    kspace_input = kspace
    if dcf is not None:
        # Ensure broadcasting: kspace (B, C, N) * dcf (B, 1, N)
        if dcf.ndim == 2:
            dcf = dcf.unsqueeze(1)
        kspace_input = kspace * dcf

    # Perform Adjoint Operation
    # Create operator (could be cached if performance critical, but this function is stateless)
    # Note: Creating KbNufftAdjoint is relatively cheap compared to operation
    adj_op = tkbn.KbNufftAdjoint(im_size=im_size, device=device)

    # Move inputs to device
    if kspace_input.device != device:
        kspace_input = kspace_input.to(device)
    if ktraj.device != device:
        ktraj = ktraj.to(device)

    # (B, C, H, W)
    image = adj_op(kspace_input, ktraj)

    return image
