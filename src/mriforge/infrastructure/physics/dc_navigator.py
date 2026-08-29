"""DC Navigator — Endogenous Virtual Marker from K-Space Center.

Extracts the DC component k₀(t) = k(0,0,t) from sequential k-space
readouts to create a free, always-available physiological motion signal.

Physics:
    The center of k-space is the spatial integral of the image energy:

    .. math::

        k_0(t) = \\int\\!\\int I(x, y, t)\\,dx\\,dy

    Respiratory motion modulates the **magnitude** (changing tissue volume
    in the slice), while cardiac pulsation modulates the **phase** (bulk
    flow velocity encoding).  Radial and spiral trajectories sample k₀
    on every single readout, giving continuous 500+ Hz tracking for free.

Spectral Decomposition:
    Band-pass filtering separates the respiratory component
    (f ≈ 0.15–0.5 Hz) from the cardiac component (f ≈ 0.8–2.0 Hz).

Reference:
    Feng et al., "XD-GRASP: Golden-angle radial MRI with reconstruction
    of extra motion dimensions," *MRM*, 2016.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DCNavigatorExtractor(nn.Module):
    """Extract and decompose the DC navigator signal from k-space data.

    Args:
        sample_rate: Acquisition sample rate in Hz (readouts per second).
        resp_band: Respiratory frequency band ``(f_lo, f_hi)`` in Hz.
        card_band: Cardiac frequency band ``(f_lo, f_hi)`` in Hz.
        center_radius: Radius (in k-space pixels) around DC to average.
            Use 0 for a single-point DC, >0 for a more robust average.
    """

    def __init__(
        self,
        sample_rate: float = 500.0,
        resp_band: tuple[float, float] = (0.15, 0.5),
        card_band: tuple[float, float] = (0.8, 2.0),
        center_radius: int = 0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.resp_band = resp_band
        self.card_band = card_band
        self.center_radius = center_radius

        logger.info(
            "[DCNavigatorExtractor] fs=%.0f Hz, resp=(%.2f–%.2f), card=(%.2f–%.2f), center_r=%d",
            sample_rate,
            *resp_band,
            *card_band,
            center_radius,
        )

    # ------------------------------------------------------------------
    # DC Extraction
    # ------------------------------------------------------------------

    def extract_dc(self, kspace_sequence: torch.Tensor) -> torch.Tensor:
        """Extract the DC signal from a sequence of k-space frames.

        Args:
            kspace_sequence: K-space tensor.  Supported shapes:
                - ``[T, B, C, H, W]`` — temporal sequence of 2-D k-space
                - ``[B, C, H, W]`` — single frame (T=1)

                The k-space **must be centered**, i.e. as produced by
                :func:`mriforge.infrastructure.physics.fft_ops.fft2c`.  The DC
                bin is read at ``[H // 2, W // 2]``.  This is the exact
                ``fft2c`` DC location **only for even ``H``/``W``**; for *odd*
                sizes ``fft2c`` (via ``fftshift``) places DC at the ceil-rounded
                centre (e.g. index 4 for ``H == 7``), so ``H // 2`` (= 3) reads
                the bin one short of true DC — a known off-by-one for odd grids
                (production k-space is even-sized, so this is benign in practice).
                A non-centered k-space (raw ``torch.fft.fft2``, DC-at-corner)
                would silently read a high-frequency corner bin instead of DC —
                this method does not re-center or validate the input.

        Returns:
            Complex DC signal ``[T]`` (magnitude + phase).
        """
        if kspace_sequence.dim() == 4:
            kspace_sequence = kspace_sequence.unsqueeze(0)  # add T

        T, B, C, H, W = kspace_sequence.shape
        cy, cx = H // 2, W // 2
        r = self.center_radius

        if r == 0:
            # Single-point DC
            dc = kspace_sequence[:, :, :, cy, cx]  # [T, B, C]
        else:
            # Average within radius
            y_idx = torch.arange(max(0, cy - r), min(H, cy + r + 1))
            x_idx = torch.arange(max(0, cx - r), min(W, cx + r + 1))
            yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
            y_sel = yy[mask].long()
            x_sel = xx[mask].long()
            dc = kspace_sequence[:, :, :, y_sel, x_sel].mean(dim=-1)

        # Average across batch and channels → single complex scalar per time
        dc = dc.mean(dim=(1, 2))  # [T]
        return dc

    # ------------------------------------------------------------------
    # Spectral Decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def _bandpass(
        signal: torch.Tensor,
        sample_rate: float,
        f_lo: float,
        f_hi: float,
    ) -> torch.Tensor:
        """Apply ideal band-pass filter via FFT.

        Args:
            signal: 1-D complex or real signal ``[T]``.
            sample_rate: Sampling rate in Hz.
            f_lo: Low cutoff frequency.
            f_hi: High cutoff frequency.

        Returns:
            Filtered signal ``[T]``.
        """
        T = signal.shape[0]
        freqs = torch.fft.fftfreq(T, d=1.0 / sample_rate, device=signal.device)
        spectrum = torch.fft.fft(signal)

        # Zero out frequencies outside the band
        mask = (freqs.abs() >= f_lo) & (freqs.abs() <= f_hi)
        filtered_spectrum = spectrum * mask.to(spectrum.dtype)

        return torch.fft.ifft(filtered_spectrum)

    def decompose(
        self,
        dc_signal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decompose DC signal into respiratory and cardiac components.

        Args:
            dc_signal: Complex 1-D signal ``[T]`` from :meth:`extract_dc`.

        Returns:
            Dictionary with keys:
            - ``"respiratory"``: band-passed respiratory component
            - ``"cardiac"``: band-passed cardiac component
            - ``"residual"``: everything outside both bands
        """
        resp = self._bandpass(dc_signal, self.sample_rate, *self.resp_band)
        card = self._bandpass(dc_signal, self.sample_rate, *self.card_band)
        residual = dc_signal - resp - card

        return {
            "respiratory": resp,
            "cardiac": card,
            "residual": residual,
        }

    def forward(self, kspace_sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract DC and decompose into physiological components.

        Args:
            kspace_sequence: ``[T, B, C, H, W]`` k-space sequence.

        Returns:
            Dictionary with ``"dc"``, ``"respiratory"``, ``"cardiac"``,
            ``"residual"`` keys.

        forward method for DCNavigatorExtractor.

        Executes PyTorch tensor operations.

        Args:
            kspace_sequence (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        dc = self.extract_dc(kspace_sequence)
        components = self.decompose(dc)
        components["dc"] = dc
        return components


__all__ = ["DCNavigatorExtractor"]
