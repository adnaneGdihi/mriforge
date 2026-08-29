"""Multiband EPI forward operator + 4D Cartesian masks (fMRI §1).

Plan-reference: ``TODO/audit_plan_novel_fmri.md`` §1 explicitly calls
out that since raw multiband k-space is essentially unavailable, the
training pipeline applies a *learned-or-fixed* EPI forward operator
to the reconstructed BOLD volumes. This module ships the standard
fixed forward operator (slice-aliasing + CAIPIRINHA inter-slice
phase encoding + Cartesian undersampling along the in-plane PE
direction) as a deterministic stand-in for the learned variant.

References:
    [1] D. A. Feinberg et al., "Multiplexed EPI for sub-second whole
        brain fMRI", *PLOS ONE*, 5(12), 2010.
    [2] K. Setsompop et al., "Blipped-CAIPIRINHA for SMS-EPI",
        *Magn. Reson. Med.*, 67(5), 2012, 1210-1224.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c


@dataclass(frozen=True)
class MultibandEPIConfig:
    """Configuration for the multiband-EPI forward operator.

    Attributes:
        multiband_factor: Number of slices simultaneously excited (1
            disables aliasing).
        caipi_shift: Per-slice phase increment (cycles), 0.0 disables
            CAIPIRINHA.
        in_plane_R: In-plane Cartesian acceleration along the
            phase-encode direction (1 = full sampling).
        center_lines: Number of fully-sampled centre k-space lines
            (auto-calibration).
    """

    multiband_factor: int = 1
    caipi_shift: float = 0.0
    in_plane_R: int = 1
    center_lines: int = 0


def cartesian_mask_4d(
    *,
    n_time: int,
    n_slice: int,
    height: int,
    width: int,
    in_plane_R: int = 1,
    center_lines: int = 0,
    phase_encode_axis: int = -2,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a 4-D (T, S, H, W) Cartesian k-space sampling mask.

    The same in-plane mask is applied to every (t, s); production-grade
    schedules randomise per-timepoint to break temporal aliasing — the
    knob is exposed via ``in_plane_R`` and ``center_lines`` per the
    plan's "extend ``sampling.py::MaskGenerator`` to 4D" requirement.
    """
    if in_plane_R < 1:
        raise ValueError(f"in_plane_R must be ≥ 1, got {in_plane_R}")
    axis_len = height if phase_encode_axis == -2 else width
    mask_1d = torch.zeros(axis_len, device=device, dtype=dtype)
    mask_1d[::in_plane_R] = 1.0
    if center_lines > 0:
        c0 = max(0, axis_len // 2 - center_lines // 2)
        c1 = min(axis_len, c0 + center_lines)
        mask_1d[c0:c1] = 1.0
    if phase_encode_axis == -2:
        mask_2d = mask_1d.unsqueeze(-1).expand(height, width)
    else:
        mask_2d = mask_1d.unsqueeze(0).expand(height, width)
    return mask_2d.unsqueeze(0).unsqueeze(0).expand(n_time, n_slice, height, width).contiguous()


def simulated_multiband_epi(
    volume: torch.Tensor,
    *,
    cfg: MultibandEPIConfig,
    coil_sensitivities: torch.Tensor | None = None,
    sampling_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simulate multiband-EPI k-space from a 4-D BOLD volume.

    Forward model (per timepoint):

    1. Multiply by coil sensitivities (single-coil if absent).
    2. FFT each slice.
    3. Apply CAIPIRINHA phase ramp across the multiband-aliased
       slice group.
    4. Sum the aliased slice contributions.
    5. Apply the Cartesian undersampling mask.

    Args:
        volume: ``[B, T, S, H, W]`` real-valued BOLD volume (one batch
            element per subject).
        cfg: :class:`MultibandEPIConfig`.
        coil_sensitivities: Optional ``[B, C, S, H, W]`` complex coil
            maps. Single-coil identity when absent.
        sampling_mask: Optional ``[T, S_mb, H, W]`` real mask. Built
            automatically when absent.

    Returns:
        Aliased multi-coil k-space tensor of shape
        ``[B, C, T, S_mb, H, W]`` (complex).
    """
    if volume.dim() != 5:
        raise ValueError(f"expected [B, T, S, H, W] volume, got {volume.shape}")
    B, T, S, H, W = volume.shape
    if S % cfg.multiband_factor != 0:
        raise ValueError(
            f"slice count {S} not divisible by multiband_factor {cfg.multiband_factor}"
        )
    S_mb = S // cfg.multiband_factor
    device = volume.device

    if coil_sensitivities is None:
        coil_sensitivities = torch.ones(B, 1, S, H, W, dtype=torch.complex64, device=device)
    C = coil_sensitivities.shape[1]

    # Per-slice FFT applied independently of aliasing.
    vol_c = volume.unsqueeze(1).to(torch.complex64) * coil_sensitivities.unsqueeze(
        2
    )  # [B, C, T, S, H, W]
    kspace = fft2c(vol_c.reshape(B * C * T * S, 1, H, W)).reshape(B, C, T, S, H, W)

    # CAIPIRINHA phase ramp per slice within each multiband group.
    if cfg.caipi_shift != 0.0:
        slice_indices = torch.arange(S, device=device, dtype=torch.float32)
        # within-group index = i mod multiband_factor
        within = slice_indices % cfg.multiband_factor
        # Phase ramp along the height axis (phase-encode direction).
        ph = torch.exp(
            1j
            * 2.0
            * torch.pi
            * cfg.caipi_shift
            * within.view(-1, 1, 1)
            * torch.linspace(0, 1, H, device=device).view(1, -1, 1)
        )
        kspace = kspace * ph.view(1, 1, 1, S, H, 1)

    # Multiband aliasing: sum every multiband_factor slices.
    kspace_aliased = kspace.view(B, C, T, S_mb, cfg.multiband_factor, H, W).sum(dim=4)

    if sampling_mask is None:
        sampling_mask = cartesian_mask_4d(
            n_time=T,
            n_slice=S_mb,
            height=H,
            width=W,
            in_plane_R=cfg.in_plane_R,
            center_lines=cfg.center_lines,
            device=device,
        )
    return kspace_aliased * sampling_mask.unsqueeze(0).unsqueeze(0)


def adjoint_multiband_epi(
    kspace_aliased: torch.Tensor,
    *,
    cfg: MultibandEPIConfig,
    coil_sensitivities: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adjoint of :func:`simulated_multiband_epi` for SENSE-style
    zero-filled reconstruction.

    Returns:
        ``[B, T, S, H, W]`` real-valued image-domain magnitude.
    """
    B, C, T, S_mb, H, W = kspace_aliased.shape
    S = S_mb * cfg.multiband_factor
    device = kspace_aliased.device
    if coil_sensitivities is None:
        coil_sensitivities = torch.ones(B, 1, S, H, W, dtype=torch.complex64, device=device)
    # Replicate aliased k-space across the multiband stack and inverse-FFT.
    rep = kspace_aliased.unsqueeze(4).expand(B, C, T, S_mb, cfg.multiband_factor, H, W)
    rep = rep.reshape(B, C, T, S, H, W)
    img = ifft2c(rep.reshape(B * C * T * S, 1, H, W)).reshape(B, C, T, S, H, W)
    coil_combined = (img * coil_sensitivities.conj().unsqueeze(2)).sum(dim=1)  # [B, T, S, H, W]
    return coil_combined.abs()


__all__ = [
    "MultibandEPIConfig",
    "adjoint_multiband_epi",
    "cartesian_mask_4d",
    "simulated_multiband_epi",
]
