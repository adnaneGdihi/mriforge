"""NUFFT-based Forward Operator Implementation with B0 Field Correction.

This module provides the NUFFT-based forward operator implementation
with Time-Segmented B0 correction for Non-Cartesian MRI reconstruction.

Physics Reference:
- Noll DC et al., "A homogeneity correction method for magnetic resonance
  imaging with time-varying gradients" IEEE TMI 1991
- Sutton BP et al., "Fast, iterative image reconstruction for MRI in the
  presence of field inhomogeneities" IEEE TMI 2003
"""

import logging

import numpy as np
import torch

from spectramr.infrastructure.physics.registry import BaseForwardOperator, register_operator

try:
    import torchkbnufft as tkbn

    TORCHKBNUFFT_AVAILABLE = True
except ImportError:
    TORCHKBNUFFT_AVAILABLE = False
    tkbn = None

logger = logging.getLogger(__name__)


class NUFFTOperator(BaseForwardOperator):
    """NUFFT-based forward operator with B0 field correction.

    Implements Time-Segmented reconstruction to correct for B0 inhomogeneity
    in Non-Cartesian (Spiral, Radial) acquisitions.

    The signal equation with off-resonance is:
        s(k(t)) = integral{ m(r) * exp(-i * k(t) * r) * exp(-i * 2π * Δf(r) * t) dr }

    Time-segmentation approximates this by splitting the readout into L segments
    and demodulating the phase error at each segment's center time.
    """

    def __init__(
        self,
        traj: torch.Tensor,
        dcf: torch.Tensor | None = None,
        im_size: tuple[int, int] = (256, 256),
        normalized: bool = True,
        n_segments: int = 6,
        dt: float = 4e-6,  # Dwell time in seconds (4 µs typical)
    ):
        """Initialize NUFFT operator.

        Args:
            traj: Trajectory tensor [2, N] in [-π, π]
            dcf: Density Compensation Function [N]
            im_size: Image size (H, W)
            normalized: Whether to use normalized FFT
            n_segments: Number of time segments for B0 correction (6-10 typical)
            dt: Dwell time in seconds (4e-6 = 4µs typical for clinical)
        """
        super().__init__()
        if not TORCHKBNUFFT_AVAILABLE:
            raise ImportError("torchkbnufft is required for NUFFTOperator")

        self.traj = traj
        self.dcf = dcf
        self.im_size = im_size
        self.normalized = normalized
        self.n_segments = n_segments
        self.dt = dt

        # Create NUFFT objects
        self.nufft_obj = tkbn.KbNufft(im_size=im_size)
        self.adj_nufft_obj = tkbn.KbNufftAdjoint(im_size=im_size)

    def _get_time_vector(self, n_points: int, device: torch.device) -> torch.Tensor:
        """Create readout time vector.

        Args:
            n_points: Number of k-space samples
            device: Target device

        Returns:
            Time vector [N] from 0 to (N-1)*dt
        """
        return torch.arange(n_points, device=device, dtype=torch.float32) * self.dt

    def _validate_setup(self) -> None:
        """Validate operator setup."""
        if not TORCHKBNUFFT_AVAILABLE:
            raise ImportError("torchkbnufft is required for NUFFTOperator")

    def forward(
        self,
        img: torch.Tensor,
        smaps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        b0_map: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward operator: image -> k-space.

        Physics Simulation: S = NUFFT(I * Sens)

        Note: B0 corruption in forward is complex (integral over time).
        For DL training, we typically generate clean k-space and let the
        network learn to correct B0 effects during reconstruction.

        Args:
            img: Image [B, C, H, W]
            smaps: Sensitivity maps [B, Coils, H, W]
            mask: K-space mask (for Cartesian fallback)
            b0_map: Field map in Hz [B, 1, H, W] (currently unused in forward)
        """
        if img.ndim > 4:
            raise RuntimeError(f"Input image must have at most 4 dimensions, got {img.ndim}")

        if smaps is not None:
            # Multi-coil forward model: image * coil_sens -> NUFFT
            img_coil = img * smaps
        else:
            img_coil = img

        # Apply NUFFT
        kspace = self.nufft_obj(img_coil, self.traj)

        if mask is not None:
            kspace = kspace * mask

        return kspace

    def adjoint(
        self,
        kspace: torch.Tensor,
        smaps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        dcf: torch.Tensor | None = None,
        b0_map: torch.Tensor | None = None,
        combine_rss: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Adjoint operator with Time-Segmented B0 correction.

        Approximates: I = F^H * (D * S) with B0 demodulation.

        Args:
            kspace: K-space data [B, Coils, N]
            smaps: Sensitivity maps [B, Coils, H, W]
            mask: K-space mask
            dcf: Density Compensation Function [B, N] or [N]
            b0_map: Field map in Hz [B, 1, H, W]
            combine_rss: Use Root-Sum-of-Squares coil combination

        Returns:
            Reconstructed image [B, 1, H, W]
        """
        if kspace.ndim > 4:
            raise RuntimeError(f"Input k-space must have at most 4 dimensions, got {kspace.ndim}")

        if mask is not None:
            kspace = kspace * mask

        # Apply Density Compensation Function (DCF)
        density = dcf if dcf is not None else self.dcf
        if density is not None:
            if density.ndim < kspace.ndim:
                while density.ndim < kspace.ndim:
                    density = density.unsqueeze(1)
            kspace = kspace * density

        # =====================================================================
        # TIME-SEGMENTED B0 CORRECTION
        # =====================================================================
        if b0_map is not None:
            img = self._adjoint_b0_corrected(kspace, smaps, b0_map)
        else:
            # Standard adjoint (no B0 correction)
            img_coil = self.adj_nufft_obj(kspace, self.traj)

            if smaps is not None:
                if combine_rss:
                    img = torch.sqrt(
                        torch.sum(torch.abs(img_coil * smaps.conj()) ** 2, dim=1, keepdim=True)
                    )
                else:
                    img = torch.sum(img_coil * smaps.conj(), dim=1, keepdim=True)
            else:
                img = img_coil

        return img

    def _adjoint_b0_corrected(
        self,
        kspace: torch.Tensor,
        smaps: torch.Tensor | None,
        b0_map: torch.Tensor,
    ) -> torch.Tensor:
        """Time-Segmented B0-corrected adjoint reconstruction.

        Strategy: Split readout into L segments, reconstruct each with
        phase demodulation, then sum.

        Args:
            kspace: K-space [B, Coils, N]
            smaps: Sensitivity maps [B, Coils, H, W]
            b0_map: Field map in Hz [B, 1, H, W]

        Returns:
            B0-corrected image [B, 1, H, W]
        """
        n_points = kspace.shape[-1]
        if self.n_segments > n_points:
            raise ValueError(
                f"Time-segmented B0 adjoint requires n_segments <= n_points, "
                f"but got n_segments={self.n_segments} and n_points={n_points}. "
                f"With n_segments > n_points, seg_len floors to 0 and all but the "
                f"final segment become empty-slice NUFFT calls, silently collapsing "
                f"the time-segmentation to a single, mis-centered segment."
            )
        seg_len = n_points // self.n_segments
        device = kspace.device

        # Accumulator for final image
        final_image = None

        for seg_idx in range(self.n_segments):
            # Segment indices
            start = seg_idx * seg_len
            end = (seg_idx + 1) * seg_len if seg_idx < self.n_segments - 1 else n_points

            # Slice k-space and trajectory for this segment
            kspace_seg = kspace[..., start:end]
            traj_seg = self.traj[..., start:end]

            # Reconstruct this segment
            img_seg = self.adj_nufft_obj(kspace_seg, traj_seg)

            # Calculate center time for this segment (in seconds)
            t_center = ((start + end) / 2.0) * self.dt

            # B0 Correction: Demodulate by exp(+i * 2π * Δf * t)
            # Note: PLUS sign to cancel the -i in the signal equation
            # b0_map is in Hz, t_center is in seconds
            phase_correction = 2 * np.pi * b0_map * t_center
            correction_phasor = torch.exp(1j * phase_correction)

            # Apply correction (handle real/complex)
            if torch.is_complex(img_seg):
                img_seg_corrected = img_seg * correction_phasor
            else:
                # Convert to complex if needed
                img_seg_complex = torch.complex(img_seg, torch.zeros_like(img_seg))
                img_seg_corrected = img_seg_complex * correction_phasor

            # Coil combination for this segment
            if smaps is not None:
                img_seg_combined = torch.sum(img_seg_corrected * smaps.conj(), dim=1, keepdim=True)
            else:
                img_seg_combined = img_seg_corrected

            # Accumulate
            if final_image is None:
                final_image = img_seg_combined
            else:
                final_image = final_image + img_seg_combined

        return final_image

    @property
    def img_size(self) -> tuple[int, int]:
        """Get image size (H, W)."""
        return self.im_size

    def get_operator_type(self) -> str:
        """Get string identifier for operator type."""
        return "nufft"


# Register the operator
register_operator(
    name="nufft",
    operator_class=NUFFTOperator,
    config={"im_size": (256, 256), "normalized": True, "n_segments": 6},
)
