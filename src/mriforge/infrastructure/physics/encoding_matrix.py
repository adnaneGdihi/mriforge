r"""Explicit Encoding Matrix Builder for Model-Based Reconstruction.

Constructs the encoding matrix **E** that maps image-space spin densities
to k-space measurements, incorporating per-readout-sample B₀ phase:

.. math::

    E_{(j \cdot N + i), \, m} = e^{-2\pi i \Delta B_0(\mathbf{x}_m)(t_i + t_{\text{shift}})}
        \cdot e^{-2\pi i \mathbf{k}_{j,i} \cdot \mathbf{x}_m}

where :math:`j` is the phase-encoding index, :math:`i` is the frequency-
encoding index, and :math:`m` is the voxel index.

The matrix **E** ∈ ℂ^(N²×M) is the total encoding matrix in which the
encoding matrices for each phase-encoding step are stacked vertically.

For the paper's standard 128×128 case:
- N² = 16384 rows (measurements)
- M = 16384 columns (voxels)
- Memory: ~2 GB for complex64

This module provides both:
1. **Explicit matrix** construction (for small problems / validation)
2. **Matrix-free** forward/adjoint operators (for GPU-efficient CG)

Reference:
    Koolstra K et al., MAGMA (2021) 34:631–642, §2.5.
"""

from __future__ import annotations

import math

import torch


class EncodingMatrixBuilder:
    r"""Builds and applies the MRI encoding matrix with B₀ phase.

    The encoding matrix incorporates both spatial encoding (gradient
    encoding via standard DFT) and temporal B₀ phase accumulation
    during the readout window.

    Args:
        n_readout: Number of frequency-encoding samples.
        n_phase: Number of phase-encoding steps.
        fov_m: Field of view in metres.
        bw_hz: Readout bandwidth in Hz.
    """

    def __init__(
        self,
        n_readout: int = 128,
        n_phase: int = 128,
        fov_m: float = 0.225,
        bw_hz: float = 20_000.0,
    ) -> None:
        self.n_ro = n_readout
        self.n_pe = n_phase
        self.fov = fov_m
        self.bw = bw_hz
        self.dwell = 1.0 / bw_hz

    def _build_coordinates(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build all required coordinate vectors.

        Returns:
            (x_coords, y_coords, kx, ky, t_readout)
        """
        # Spatial coordinates
        x = torch.linspace(-self.fov / 2, self.fov / 2, self.n_ro, device=device)
        y = torch.linspace(-self.fov / 2, self.fov / 2, self.n_pe, device=device)

        # K-space coordinates
        dk = 1.0 / self.fov
        kx = (torch.arange(self.n_ro, device=device) - self.n_ro // 2) * dk
        ky = (torch.arange(self.n_pe, device=device) - self.n_pe // 2) * dk

        # Readout time
        total = self.n_ro * self.dwell
        t_ro = torch.linspace(-total / 2, total / 2, self.n_ro, device=device)

        return x, y, kx, ky, t_ro

    def build_explicit(
        self,
        b0_map: torch.Tensor,
        t_shift: float = 0.0,
    ) -> torch.Tensor:
        r"""Build the explicit encoding matrix E.

        .. warning::
            For 128×128, E is a 16384×16384 complex matrix (~2 GB).
            Use :meth:`forward` / :meth:`adjoint` for memory-efficient
            matrix-free operations on larger problems.

        Args:
            b0_map: B₀ deviation map [1, 1, H, W] or [H, W] in Hz.
            t_shift: Readout time-shift in seconds.

        Returns:
            Encoding matrix E [N², M] complex64.
        """
        device = b0_map.device
        b0_2d = b0_map.squeeze()  # [H, W]

        x, y, kx, ky, t_ro = self._build_coordinates(device)

        # Spatial grids
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx_flat = xx.reshape(-1)  # [M]
        yy_flat = yy.reshape(-1)  # [M]
        b0_flat = b0_2d.reshape(-1)  # [M]

        M = xx_flat.shape[0]
        N2 = self.n_pe * self.n_ro

        E = torch.zeros(N2, M, dtype=torch.complex64, device=device)

        for j in range(self.n_pe):
            row_start = j * self.n_ro

            # B₀ phase: exp(-i * 2π * ΔB₀(x) * (t + t_shift))
            b0_phase = (
                -2.0 * math.pi * b0_flat.unsqueeze(0) * (t_ro.unsqueeze(1) + t_shift)
            )  # [N_ro, M]

            # Spatial phase: exp(-i * 2π * (kx·x + ky·y))
            spatial_phase = (
                -2.0
                * math.pi
                * (kx.unsqueeze(1) * xx_flat.unsqueeze(0) + ky[j] * yy_flat.unsqueeze(0))
            )  # [N_ro, M]

            E[row_start : row_start + self.n_ro] = torch.exp(1j * (b0_phase + spatial_phase))

        return E

    def forward(
        self,
        image: torch.Tensor,
        b0_map: torch.Tensor,
        t_shift: float = 0.0,
    ) -> torch.Tensor:
        r"""Matrix-free forward operation: E·m.

        Computes the encoding without explicitly building the full matrix.
        Computes one PE line at a time for memory efficiency.

        Args:
            image: Spin density [1, 1, H, W] or [H, W], real or complex.
            b0_map: B₀ deviation [1, 1, H, W] or [H, W] in Hz.
            t_shift: Readout time-shift in seconds.

        Returns:
            K-space measurements [1, 1, N_pe, N_ro] complex.
        """
        device = image.device
        m = image.squeeze().reshape(-1).to(torch.complex64)  # [M]
        b0_flat = b0_map.squeeze().reshape(-1)  # [M]

        x, y, kx, ky, t_ro = self._build_coordinates(device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        kspace = torch.zeros(self.n_pe, self.n_ro, dtype=torch.complex64, device=device)

        for j in range(self.n_pe):
            b0_phase = -2.0 * math.pi * b0_flat.unsqueeze(0) * (t_ro.unsqueeze(1) + t_shift)
            spatial_phase = (
                -2.0
                * math.pi
                * (kx.unsqueeze(1) * xx_flat.unsqueeze(0) + ky[j] * yy_flat.unsqueeze(0))
            )
            E_row = torch.exp(1j * (b0_phase + spatial_phase))  # [N_ro, M]
            kspace[j] = E_row @ m  # [N_ro]

        return kspace.unsqueeze(0).unsqueeze(0)

    def adjoint(
        self,
        kspace: torch.Tensor,
        b0_map: torch.Tensor,
        t_shift: float = 0.0,
    ) -> torch.Tensor:
        r"""Matrix-free adjoint operation: E†·y.

        Computes the conjugate-transpose encoding without building the
        full matrix. Accumulates contributions one PE line at a time.

        Args:
            kspace: K-space measurements [1, 1, N_pe, N_ro] complex.
            b0_map: B₀ deviation [1, 1, H, W] or [H, W] in Hz.
            t_shift: Readout time-shift in seconds.

        Returns:
            Image estimate [1, 1, H, W] complex.
        """
        device = kspace.device
        kspace_2d = kspace.squeeze()  # [N_pe, N_ro]
        b0_2d = b0_map.squeeze()  # [H, W]
        H, W = b0_2d.shape
        b0_flat = b0_2d.reshape(-1)  # [M]
        M = b0_flat.shape[0]

        x, y, kx, ky, t_ro = self._build_coordinates(device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        image_flat = torch.zeros(M, dtype=torch.complex64, device=device)

        for j in range(self.n_pe):
            b0_phase = -2.0 * math.pi * b0_flat.unsqueeze(0) * (t_ro.unsqueeze(1) + t_shift)
            spatial_phase = (
                -2.0
                * math.pi
                * (kx.unsqueeze(1) * xx_flat.unsqueeze(0) + ky[j] * yy_flat.unsqueeze(0))
            )
            # E†: conjugate transpose → conjugate of each row, then multiply
            E_row_H = torch.exp(-1j * (b0_phase + spatial_phase))  # [N_ro, M]

            # E†·y for this PE line: sum over readout samples
            image_flat += E_row_H.T @ kspace_2d[j]  # [M]

        return image_flat.reshape(H, W).unsqueeze(0).unsqueeze(0)

    def build_with_sampling(
        self,
        b0_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        t_shift: float = 0.0,
    ) -> torch.Tensor:
        """Build encoding matrix with undersampling (R·E).

        For compressed sensing: only rows corresponding to sampled
        PE lines are included.

        Args:
            b0_map: B₀ deviation map [H, W] or [1, 1, H, W] in Hz.
            sampling_mask: Binary mask [N_pe] indicating sampled lines.
            t_shift: Readout time-shift in seconds.

        Returns:
            Reduced encoding matrix [n_sampled * N_ro, M] complex64.
        """
        E_full = self.build_explicit(b0_map, t_shift=t_shift)

        # Build row selection
        mask_1d = sampling_mask.squeeze().bool()
        sampled_indices = torch.where(mask_1d)[0]

        rows = []
        for j in sampled_indices:
            row_start = j * self.n_ro
            rows.append(E_full[row_start : row_start + self.n_ro])

        return torch.cat(rows, dim=0)


__all__ = ["EncodingMatrixBuilder"]
