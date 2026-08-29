"""K-Space Kinematic Trajectory Generator for MRI Acquisition Simulation.

Generates physically-accurate 1D k-space readout trajectories that preserve
the **chronological Arrow of Time** during the MRI scan. These trajectories
are plugged directly into differentiable NUFFT operators (TorchKbNufft).

**CRITICAL DOMAIN RULE:**
Never apply spatial space-filling curves (Hilbert, Morton) to k-space.
K-space is an acoustic time-series governed by gradient waveforms.
Spatial permutation destroys the temporal continuity needed for motion tracking.

Supported trajectories:

- ``archimedean_spiral``: Center-out spiral with configurable arms.
  Low-frequencies sampled first → natural temporal high-pass.
- ``radial_golden_angle``: Golden-angle-spaced spokes through DC.
  SOTA for motion robustness — every spoke passes through k-space center.
- ``epi_boustrophedon``: Echo Planar Imaging serpentine readout.
  Alternating readout direction per PE line captures B0 phase errors.
- ``cartesian_sequential``: Standard Cartesian phase-encode ordering.
  Baseline reference trajectory.

Reference:
    Pipe, J.G. (1999). "Motion correction with PROPELLER MRI."
    Magnetic Resonance in Medicine.

    Winkelmann et al. (2007). "An optimal radial profile order based
    on the Golden Ratio." IEEE TMI.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def archimedean_spiral(
    num_arms: int = 12,
    points_per_arm: int = 2048,
    max_k: float = 0.5,
    num_rotations: float = 10.0,
) -> torch.Tensor:
    """Generate Archimedean spiral k-space trajectory.

    The gradients oscillate as sine/cosine with linearly increasing
    amplitude, sampling low-frequencies first (center-out).

    Args:
        num_arms: Number of interleaved spiral arms.
        points_per_arm: Readout samples per arm.
        max_k: Maximum k-space radius (cycles/FOV).
        num_rotations: Full rotations per arm.

    Returns:
        Trajectory tensor ``[2, num_arms * points_per_arm]`` — (kx, ky).
    """
    t = torch.linspace(0, 1, points_per_arm)
    trajectories = []

    for i in range(num_arms):
        theta_offset = i * (2 * math.pi / num_arms)
        r = max_k * t
        theta = num_rotations * 2 * math.pi * t + theta_offset

        kx = r * torch.cos(theta)
        ky = r * torch.sin(theta)
        trajectories.append(torch.stack([kx, ky], dim=0))

    traj = torch.cat(trajectories, dim=1)
    logger.debug(
        "[archimedean_spiral] arms=%d, pts/arm=%d, total=%d",
        num_arms,
        points_per_arm,
        traj.shape[1],
    )
    return traj


def radial_golden_angle(
    num_spokes: int = 400,
    points_per_spoke: int = 512,
    max_k: float = 0.5,
) -> torch.Tensor:
    """Generate radial golden-angle k-space trajectory.

    Each spoke passes through the k-space center at angle increments of
    π × (√5 - 1)/2 ≈ 111.246°. This guarantees near-uniform coverage
    and maximal motion robustness (every spoke samples DC).

    Args:
        num_spokes: Number of radial spokes.
        points_per_spoke: Readout samples per spoke.
        max_k: Maximum k-space radius.

    Returns:
        Trajectory tensor ``[2, num_spokes * points_per_spoke]``.
    """
    t = torch.linspace(-max_k, max_k, points_per_spoke)
    golden_angle = math.pi * (math.sqrt(5) - 1) / 2  # ~111.246°

    trajectories = []
    for i in range(num_spokes):
        theta = i * golden_angle
        kx = t * math.cos(theta)
        ky = t * math.sin(theta)
        trajectories.append(torch.stack([kx, ky], dim=0))

    traj = torch.cat(trajectories, dim=1)
    logger.debug(
        "[radial_golden_angle] spokes=%d, pts/spoke=%d, total=%d",
        num_spokes,
        points_per_spoke,
        traj.shape[1],
    )
    return traj


def epi_boustrophedon(
    num_lines: int = 128,
    points_per_line: int = 128,
    max_k: float = 0.5,
) -> torch.Tensor:
    """Generate EPI (Echo Planar Imaging) boustrophedon trajectory.

    The scanner reads left→right on even lines and right→left on odd
    lines, with blipped phase-encode steps between lines. This zig-zag
    preserves the alternating B0 phase error structure needed for
    Nyquist ghost correction.

    Args:
        num_lines: Number of phase-encode lines.
        points_per_line: Frequency-encode samples per line.
        max_k: Maximum k-space radius.

    Returns:
        Trajectory tensor ``[2, num_lines * points_per_line]``.
    """
    ky_lines = torch.linspace(-max_k, max_k, num_lines)
    kx_base = torch.linspace(-max_k, max_k, points_per_line)

    trajectories = []
    for i in range(num_lines):
        ky = torch.full((points_per_line,), ky_lines[i].item())
        kx = kx_base if i % 2 == 0 else torch.flip(kx_base, dims=[0])
        trajectories.append(torch.stack([kx, ky], dim=0))

    traj = torch.cat(trajectories, dim=1)
    logger.debug(
        "[epi_boustrophedon] lines=%d, pts/line=%d, total=%d",
        num_lines,
        points_per_line,
        traj.shape[1],
    )
    return traj


def cartesian_sequential(
    num_pe_lines: int = 256,
    num_fe_samples: int = 256,
    max_k: float = 0.5,
) -> torch.Tensor:
    """Generate standard Cartesian sequential trajectory.

    Phase-encode lines acquired sequentially from -k_max to +k_max.
    Baseline reference — no special temporal structure.

    Args:
        num_pe_lines: Number of phase-encode lines.
        num_fe_samples: Frequency-encode samples per line.
        max_k: Maximum k-space radius.

    Returns:
        Trajectory tensor ``[2, num_pe_lines * num_fe_samples]``.
    """
    ky_lines = torch.linspace(-max_k, max_k, num_pe_lines)
    kx_base = torch.linspace(-max_k, max_k, num_fe_samples)

    trajectories = []
    for i in range(num_pe_lines):
        ky = torch.full((num_fe_samples,), ky_lines[i].item())
        trajectories.append(torch.stack([kx_base, ky], dim=0))

    return torch.cat(trajectories, dim=1)


# ---------------------------------------------------------------------------
# Registry for programmatic access
# ---------------------------------------------------------------------------

TRAJECTORY_REGISTRY = {
    "archimedean_spiral": archimedean_spiral,
    "radial_golden_angle": radial_golden_angle,
    "epi_boustrophedon": epi_boustrophedon,
    "cartesian_sequential": cartesian_sequential,
}


def create_trajectory(name: str, **kwargs):  # type: ignore[no-untyped-def]
    """Create a k-space trajectory by name.

    Args:
        name: Trajectory type from ``TRAJECTORY_REGISTRY``.
        **kwargs: Trajectory-specific parameters.

    Returns:
        Trajectory tensor ``[2, total_points]``.

    Raises:
        ValueError: If name is not in registry.
    """
    if name not in TRAJECTORY_REGISTRY:
        raise ValueError(
            f"Unknown trajectory '{name}'. Available: {list(TRAJECTORY_REGISTRY.keys())}"
        )
    return TRAJECTORY_REGISTRY[name](**kwargs)
