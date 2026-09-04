r"""Conjugate Phase Reconstruction (CPR) via Multifrequency Interpolation.

Implements the CPR algorithm from Koolstra et al. (2021) §2.4, based on
the multi-frequency interpolation (MFI) method of Man et al. (1997).

The inverse of the signal equation is approximated as:

.. math::

    m(\mathbf{x}) \approx \int y(\mathbf{k}(t))
        e^{+2\pi \Delta B_0(\mathbf{x}) i (t + t_{\text{shift}})}
        e^{+\mathbf{k}(t) \cdot \mathbf{x} i} \, dt

This is implemented efficiently via FFTs by decomposing the B₀-dependent
phase into a sum of contributions at discrete frequencies, then
interpolating:

1. For each discrete frequency :math:`f_l`:
   a. Demodulate k-space by :math:`f_l` in the time domain.
   b. Apply standard IFFT to obtain a frequency-specific image.
2. For each voxel, interpolate among the frequency-specific images
   using the local B₀ value.

Default parameters (paper):
- Frequency range: −4000 to +4000 Hz
- Frequency step: 3 Hz  (2667 bins)

References:
    - Man LC, Pauly JM, Macovski A. "Multifrequency interpolation for fast
      off-resonance correction." MRM 37(5):785–792, 1997.
    - Koolstra K et al., MAGMA (2021) 34:631–642, §2.4.
"""

from __future__ import annotations

import math

import torch

from spectramr.infrastructure.physics.fft_ops import ifft2c


class ConjugatePhaseReconstructor:
    r"""Conjugate Phase Reconstruction engine.

    Corrects for B₀-induced phase distortion during readout using
    multifrequency interpolation (MFI). The correction is applied
    in the frequency domain via standard FFTs, making it
    computationally efficient compared to model-based reconstruction.

    Args:
        freq_range_hz: (min, max) frequency range for the MFI bins.
        freq_step_hz: Frequency discretisation step in Hz.
        bw_hz: Readout bandwidth in Hz.
        n_readout: Number of frequency-encoding samples.
    """

    def __init__(
        self,
        freq_range_hz: tuple[float, float] = (-4000.0, 4000.0),
        freq_step_hz: float = 3.0,
        bw_hz: float = 20_000.0,
        n_readout: int = 128,
    ) -> None:
        self.freq_min, self.freq_max = freq_range_hz
        self.freq_step = freq_step_hz
        self.bw = bw_hz
        self.n_ro = n_readout

        # Discrete frequency bins
        self.n_bins = int((self.freq_max - self.freq_min) / self.freq_step) + 1

        # Dwell time
        self.dwell = 1.0 / bw_hz

    def _build_frequency_bins(self, device: torch.device) -> torch.Tensor:
        """Build vector of discrete MFI frequency bins.

        Returns:
            Frequency bins [L] in Hz.
        """
        return torch.linspace(self.freq_min, self.freq_max, self.n_bins, device=device)

    def _build_time_vector(self, device: torch.device) -> torch.Tensor:
        """Build readout time vector centred at echo-top.

        Returns:
            Time vector [N_ro] in seconds.
        """
        total = self.n_ro * self.dwell
        return torch.linspace(-total / 2, total / 2, self.n_ro, device=device)

    def reconstruct(
        self,
        kspace: torch.Tensor,
        b0_map: torch.Tensor,
        t_shift: float = 0.0,
    ) -> torch.Tensor:
        r"""Perform CPR reconstruction.

        For each discrete MFI frequency :math:`f_l`:

        1. Apply time-domain demodulation:
           :math:`y_l[j, i] = y[j, i] \cdot e^{+2\pi i f_l (t_i + t_{\text{shift}})}`
        2. IFFT → frequency-specific image :math:`I_l(x, y)`.

        Then for each voxel, select/interpolate the image at the local
        :math:`\Delta B_0` frequency using linear interpolation.

        Args:
            kspace: Complex k-space [1, 1, N_pe, N_ro].
            b0_map: B₀ deviation map [1, 1, H, W] in Hz.
            t_shift: Readout time-shift in seconds (0 for unshifted).

        Returns:
            Reconstructed complex image [1, 1, H, W].
        """
        device = kspace.device
        kspace_2d = kspace.squeeze()  # [N_pe, N_ro]
        N_pe, N_ro = kspace_2d.shape
        b0_2d = b0_map.squeeze()  # [H, W]
        H, W = b0_2d.shape

        # Time vector sized to actual k-space readout dimension
        dwell = 1.0 / self.bw
        total_ro = N_ro * dwell
        t_ro = torch.linspace(-total_ro / 2, total_ro / 2, N_ro, device=device)
        freq_bins = self._build_frequency_bins(device)  # [L]

        # ── Step 1: Build frequency-specific images ──────────────────
        # For memory efficiency, we process bins in chunks and keep only
        # the interpolation-relevant bins per voxel. However, for the
        # paper's 128² case this is tractable directly.

        # Quantise B₀ values to bin indices for interpolation
        b0_flat = b0_2d.reshape(-1)  # [M]

        # Bin index (continuous, for linear interpolation)
        bin_idx_cont = (b0_flat - self.freq_min) / self.freq_step  # [M]
        bin_idx_lo = bin_idx_cont.floor().long().clamp(0, self.n_bins - 2)
        bin_idx_hi = bin_idx_lo + 1
        frac = (bin_idx_cont - bin_idx_lo.float()).clamp(0, 1)  # [M]

        # Identify unique bin indices needed (avoid computing all L bins)
        needed_bins = torch.unique(torch.cat([bin_idx_lo, bin_idx_hi])).sort().values

        # Pre-allocate images for needed bins
        bin_images: dict[int, torch.Tensor] = {}

        for idx in needed_bins.tolist():
            f_l = freq_bins[idx]

            # Demodulate each readout line: exp(+2πi·f_l·(t + t_shift))
            demod_phase = 2.0 * math.pi * f_l * (t_ro + t_shift)  # [N_ro]
            demod = torch.exp(1j * demod_phase).unsqueeze(0)  # [1, N_ro]

            kspace_demod = kspace_2d * demod  # [N_pe, N_ro]

            # IFFT to get frequency-specific image
            img = ifft2c(kspace_demod.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]
            bin_images[idx] = img.squeeze()  # [H, W]

        # ── Step 2: Interpolate at each voxel's B₀ value (vectorized) ──
        # Stack the per-bin images into [num_bins, M], remap each voxel's lo/hi
        # bin index to a row, then gather per voxel. This replaces a per-voxel
        # Python loop with two ``.item()`` host syncs each (~2·H·W GPU stalls).
        M = b0_flat.shape[0]
        needed = list(bin_images.keys())
        stacked = torch.stack(
            [bin_images[idx].reshape(-1) for idx in needed], dim=0
        )  # [num_bins, M]
        needed_t = torch.tensor(needed, dtype=torch.long, device=device)
        lut = torch.zeros(int(needed_t.max()) + 1, dtype=torch.long, device=device)
        lut[needed_t] = torch.arange(needed_t.numel(), device=device)

        cols = torch.arange(M, device=device)
        img_lo = stacked[lut[bin_idx_lo.long()], cols]  # [M]
        img_hi = stacked[lut[bin_idx_hi.long()], cols]  # [M]
        result_flat = (1.0 - frac) * img_lo + frac * img_hi

        return result_flat.reshape(H, W).unsqueeze(0).unsqueeze(0)


__all__ = ["ConjugatePhaseReconstructor"]
