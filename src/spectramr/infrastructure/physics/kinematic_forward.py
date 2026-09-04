"""Differentiable Kinematic Forward Operator for MRI Motion Simulation.

Implements rigid-body motion corruption in k-space using the Fourier Shift Theorem.
A spatial translation Δr manifests as a linear phase ramp: S(k) = exp(-i·2π·k·Δr).
Rotation manifests as a rigid rotation of the k-space sampling coordinates k(t).

This module enables end-to-end differentiable motion simulation for:
1. Meta-training of motion correction networks on clean M4Raw data
2. Test-Time Optimization (TTO) where only θ(t) is updated

Reference:
    Batchelor et al., "Matrix description of general motion correction
    applied to multishot images," MRM 2005.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

try:
    import torchkbnufft as tkbn

    HAS_NUFFT = True
except ImportError:
    HAS_NUFFT = False
    tkbn = None

logger = logging.getLogger(__name__)


class KinematicForwardOperator(nn.Module):
    """Differentiable rigid-body motion simulation in k-space.

    Applies per-readout-line rigid body transformations (dx, dy, dθ)
    to a clean image via NUFFT with rotated/shifted trajectories.

    The key physics:
    - Translation → linear phase ramp in k-space (Fourier Shift Theorem)
    - Rotation → inverse rotation of sampling coordinates

    Args:
        im_size: Image dimensions (H, W).
        num_readout_lines: Number of k-space readout lines (phase encodes).
            Defaults to im_size[0] for Cartesian sampling.

    Example:
        >>> op = KinematicForwardOperator(im_size=(256, 256))
        >>> clean_image = torch.randn(1, 1, 256, 256, dtype=torch.complex64)
        >>> theta = torch.zeros(1, 3, 256)  # [dx, dy, dθ] per line
        >>> corrupted_kspace = op(clean_image, theta)
    Mathematical Formulation:
    .. math::

        \\mathcal{O}_{Kinematic}(x, \theta) = \\mathcal{F}(R_\theta x \\odot S) \\odot M"""

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        num_readout_lines: int | None = None,
    ) -> None:
        """Initialize KinematicForwardOperator.

        Args:
            im_size: Image dimensions (H, W).
            num_readout_lines: Number of phase encode lines.
        """
        super().__init__()

        if not HAS_NUFFT:
            raise ImportError(
                "torchkbnufft is required for KinematicForwardOperator. "
                "Install with: pip install torchkbnufft"
            )

        self.im_size = im_size
        self.num_readout_lines = num_readout_lines or im_size[0]

        # NUFFT forward operator for off-grid sampling
        self.nufft = tkbn.KbNufft(im_size=im_size)

        # Pre-compute base Cartesian k-space trajectory [-π, π]
        # Shape: (2, num_readout_lines * num_freq_encodes)
        base_traj = self._build_cartesian_trajectory()
        self.register_buffer("base_traj", base_traj)

        logger.info(
            "[KinematicForwardOperator] Initialized: im_size=%s, readout_lines=%d, base_traj=%s",
            im_size,
            self.num_readout_lines,
            tuple(base_traj.shape),
        )

    def _build_cartesian_trajectory(self) -> torch.Tensor:
        """Build base Cartesian sampling trajectory in [-π, π].

        Returns:
            Trajectory tensor of shape (2, N) where N = H * W.
            Row 0 = kx (readout), Row 1 = ky (phase encode).
        """
        H, W = self.im_size
        # kx: readout direction (along columns) — fully sampled per line
        kx = torch.linspace(-math.pi, math.pi, W)
        # ky: phase encode direction (along rows) — one value per line
        ky = torch.linspace(-math.pi, math.pi, H)

        # Create grid: each (kx_j, ky_i) pair
        grid_kx, grid_ky = torch.meshgrid(kx, ky, indexing="xy")
        # Flatten: (H*W,) for each coordinate
        traj = torch.stack([grid_kx.reshape(-1), grid_ky.reshape(-1)], dim=0)  # (2, H*W)
        return traj

    def forward(
        self,
        image: torch.Tensor,
        theta_motion: torch.Tensor,
        smaps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply rigid-body motion corruption to a clean image in k-space.

        Args:
            image: Clean complex image [B, C, H, W] (complex64).
            theta_motion: Per-readout motion parameters [B, 3, N_lines].
                Channel 0: dx (x-translation in pixels)
                Channel 1: dy (y-translation in pixels)
                Channel 2: dtheta (rotation in radians)
            smaps: Optional coil sensitivity maps [B, N_coils, H, W] (complex64).

        Returns:
            Corrupted k-space [B, C, H, W] or [B, N_coils, H, W] (complex64).

        forward method for KinematicForwardOperator.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            theta_motion (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            smaps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = image.shape[0]
        H, W = self.im_size
        device = image.device

        # Ensure base_traj is on the correct device
        base_traj = self.base_traj.to(device)  # (2, H*W)

        # Expand theta from per-line to per-point:
        # theta_motion: [B, 3, N_lines] → expand to [B, 3, N_lines * W]
        # Each line has W readout samples
        dx = theta_motion[:, 0, :]  # [B, N_lines]
        dy = theta_motion[:, 1, :]  # [B, N_lines]
        dtheta = theta_motion[:, 2, :]  # [B, N_lines]

        # Repeat each line's parameters W times (for each readout sample)
        dx_expanded = dx.repeat_interleave(W, dim=1)  # [B, H*W]
        dy_expanded = dy.repeat_interleave(W, dim=1)  # [B, H*W]
        dtheta_expanded = dtheta.repeat_interleave(W, dim=1)  # [B, H*W]

        # 1. Rotational duality: rotate k-space trajectory inversely
        cos_t = torch.cos(-dtheta_expanded)  # [B, H*W]
        sin_t = torch.sin(-dtheta_expanded)  # [B, H*W]

        kx = base_traj[0].unsqueeze(0).expand(B, -1)  # [B, H*W]
        ky = base_traj[1].unsqueeze(0).expand(B, -1)  # [B, H*W]

        # Apply 2D rotation to each k-space point
        rotated_kx = cos_t * kx - sin_t * ky  # [B, H*W]
        rotated_ky = sin_t * kx + cos_t * ky  # [B, H*W]

        # Stack into trajectory format: [B, 2, H*W]
        rotated_traj = torch.stack([rotated_kx, rotated_ky], dim=1)

        # 2. Multi-coil NUFFT forward (handles off-grid rotated coordinates)
        if smaps is not None:
            # Apply sensitivity maps before NUFFT: [B, N_coils, H, W]
            image_multicoil = image * smaps
        else:
            image_multicoil = image

        # Forward NUFFT: image → k-space at rotated coordinates
        sampled_kspace = self.nufft(image_multicoil, rotated_traj)  # [B, C, H*W]

        # 3. Fourier Shift Theorem: translation → linear phase shift.
        # The trajectory is in torchkbnufft's radian convention (ω ∈ [-π, π]),
        # so a shift of d pixels multiplies k-space by exp(-i·ω·d) — no extra
        # 2π/N factor. The previous exp(-i·2π·ω·d/W) carried a spurious 2π/W
        # scale and under-scaled every translation by ~W/(2π) (verified: a 6-px
        # shift was recovered as 1 px at N=32).
        phase_shift = torch.exp(
            -1j * (rotated_kx * dx_expanded + rotated_ky * dy_expanded)
        )  # [B, H*W]

        # Apply phase shift to each coil
        corrupted_kspace = sampled_kspace * phase_shift.unsqueeze(1)

        # 4. Reshape to grid: [B, C, H*W] → [B, C, H, W]
        C = corrupted_kspace.shape[1]
        corrupted_kspace = corrupted_kspace.view(B, C, H, W)

        return corrupted_kspace

    def generate_random_motion(
        self,
        batch_size: int,
        max_translation: float = 5.0,
        max_rotation: float = 0.05,
        motion_type: str = "smooth",
        device: torch.device = torch.device("cpu"),
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate random motion trajectories for training.

        Args:
            batch_size: Number of trajectories to generate.
            max_translation: Maximum translation in pixels.
            max_rotation: Maximum rotation in radians.
            motion_type: 'smooth' for low-frequency motion, 'sudden' for step changes.
            device: Device for output tensor.
            generator: Optional scoped RNG. When given, ALL draws use it (on
                the generator's own device, then moved to ``device``) and the
                global CPU/CUDA RNG streams are untouched — the repo
                convention for deterministic validation, replacing global
                ``torch.manual_seed`` + save/restore brackets at call sites.
                When ``None``, draws use the global RNG on ``device``
                (byte-identical to the historical behaviour).

        Returns:
            Motion parameters [B, 3, N_lines].
        """
        N = self.num_readout_lines

        def _rand(*shape: int) -> torch.Tensor:
            if generator is None:
                return torch.rand(*shape, device=device)
            return torch.rand(*shape, generator=generator, device=generator.device).to(device)

        def _randint(low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
            # These draws are host-side (indices), so the generator (CPU by
            # convention) is used directly.
            if generator is None:
                return torch.randint(low, high, shape)
            return torch.randint(low, high, shape, generator=generator)

        if motion_type == "smooth":
            # Low-frequency smooth motion (respiratory-like)
            t = torch.linspace(0, 2 * math.pi, N, device=device)
            # Random frequencies and phases per batch
            freq = _rand(batch_size, 1) * 2 + 0.5
            phase = _rand(batch_size, 1) * 2 * math.pi

            dx = max_translation * torch.sin(freq * t.unsqueeze(0) + phase)
            dy = max_translation * 0.5 * torch.cos(freq * 1.3 * t.unsqueeze(0) + phase + 0.5)
            dtheta = max_rotation * torch.sin(freq * 0.7 * t.unsqueeze(0) + phase + 1.0)
        elif motion_type == "sudden":
            # Sudden head movement (cough/swallow)
            motion = torch.zeros(batch_size, 3, N, device=device)
            # Random onset time
            onset = _randint(N // 4, 3 * N // 4, (batch_size,))
            duration = _randint(N // 20, N // 5, (batch_size,))

            for b in range(batch_size):
                start = onset[b].item()
                end = min(start + duration[b].item(), N)
                motion[b, 0, start:end] = max_translation * (_rand(1) * 2 - 1)
                motion[b, 1, start:end] = max_translation * 0.5 * (_rand(1) * 2 - 1)
                motion[b, 2, start:end] = max_rotation * (_rand(1) * 2 - 1)
            return motion
        else:
            raise ValueError(f"Unknown motion_type '{motion_type}'. Supported: 'smooth', 'sudden'.")

        return torch.stack([dx, dy, dtheta], dim=1)  # [B, 3, N]


__all__ = ["KinematicForwardOperator"]
