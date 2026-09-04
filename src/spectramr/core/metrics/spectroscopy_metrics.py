"""Magnetic Resonance Spectroscopy (MRS) Quality Metrics.

Implements metrics for spectroscopy data:
- Spectral Linewidth (FWHM)
- CRLB (Cramér-Rao Lower Bound)
- Frequency Domain SNR

Follows unit-test.md rules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spectramr.config.schemas.enums import Regime
from spectramr.core.metrics.registry import register_metric

try:
    from beartype import beartype
    from jaxtyping import Float, jaxtyped

    JAXTYPING_AVAILABLE = True
except ImportError:
    JAXTYPING_AVAILABLE = False

    def jaxtyped(fn):
        """jaxtyped.

        Args:
            fn (Any): Description.
        Returns:
            Any: Description.
        """
        return fn

    def beartype(fn):
        """beartype.

        Args:
            fn (Any): Description.
        Returns:
            Any: Description.
        """
        return fn


@register_metric(
    "spectral_linewidth", aliases=["FWHM"], workflows=frozenset({Regime.SPECTROSCOPIC})
)
class SpectralLinewidth(nn.Module):
    """Spectral Linewidth (FWHM) metric.

    Measures the Full Width at Half Maximum of spectral peaks.
    Narrower linewidth indicates better B0 homogeneity and higher quality.

    Typical values:
    - Good quality: < 0.1 ppm (brain at 3T)
    - Clinical threshold: < 0.15 ppm
    """

    def __init__(
        self,
        peak_threshold: float = 0.5,  # Fraction of max for peak detection
    ) -> None:
        """__init__.

        Args:
            peak_threshold (float): Description.
        """
        super().__init__()
        self.peak_threshold = peak_threshold

    def _find_peaks(
        self,
        spectrum: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Find local maxima in spectrum."""
        # Simple peak detection: points higher than neighbors
        # Manual padding for 1D tensor (F.pad doesn't support 1D replicate)
        padded = torch.cat(
            [
                spectrum[0:1],  # Replicate first
                spectrum,
                spectrum[-1:],  # Replicate last
            ]
        )

        is_peak = (
            (spectrum > padded[:-2])  # Greater than left
            & (spectrum > padded[2:])  # Greater than right
            & (spectrum > spectrum.max() * self.peak_threshold)  # Above threshold
        )

        peak_indices = torch.where(is_peak)[0]
        peak_values = spectrum[peak_indices]

        return peak_indices, peak_values

    def _compute_fwhm(
        self,
        spectrum: Tensor,
        peak_idx: int,
        freq_axis: Tensor,
    ) -> float:
        """Compute FWHM for a single peak."""
        peak_val = spectrum[peak_idx]
        half_max = peak_val * 0.5

        # Find left crossing
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > half_max:
            left_idx -= 1

        # Find right crossing
        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
            right_idx += 1

        # FWHM in frequency units
        if left_idx < right_idx:
            fwhm = freq_axis[right_idx] - freq_axis[left_idx]
        else:
            fwhm = 0.0

        return abs(fwhm.item()) if isinstance(fwhm, Tensor) else abs(fwhm)

    @jaxtyped
    @beartype
    def forward(
        self,
        spectrum: Float[Tensor, "batch freq"],
        freq_axis: Float[Tensor, freq] | None = None,
        ppm_range: tuple[float, float] = (0.0, 10.0),
    ) -> Float[Tensor, ""]:
        """Compute mean spectral linewidth.

        Args:
            spectrum: Magnitude spectrum [B, F]
            freq_axis: Frequency/ppm axis (created if None)
            ppm_range: PPM range covered by spectrum

        Returns:
            Mean FWHM across peaks (in ppm units)

        forward method for SpectralLinewidth.

        Executes PyTorch tensor operations.

        Args:
            spectrum (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            freq_axis (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ppm_range (tuple[float, float]): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, F = spectrum.shape
        device = spectrum.device

        if freq_axis is None:
            freq_axis = torch.linspace(ppm_range[0], ppm_range[1], F, device=device)

        all_fwhms = []

        for b in range(B):
            spec = spectrum[b]
            peak_indices, _ = self._find_peaks(spec)

            if len(peak_indices) == 0:
                continue

            for peak_idx in peak_indices:
                fwhm = self._compute_fwhm(spec, peak_idx.item(), freq_axis)
                if fwhm > 0:
                    all_fwhms.append(fwhm)

        if len(all_fwhms) > 0:
            return torch.tensor(sum(all_fwhms) / len(all_fwhms), device=device)
        else:
            return torch.tensor(0.0, device=device)


@register_metric("crlb", aliases=["CRLB"], workflows=frozenset({Regime.SPECTROSCOPIC}))
class CRLB(nn.Module):
    """Cramér-Rao Lower Bound (CRLB) estimation.

    Estimates the theoretical lower bound on variance of metabolite
    concentration estimates. Lower CRLB indicates more reliable quantification.

    Typically reported as %CRLB = CRLB / concentration * 100

    Thresholds:
    - Reliable: %CRLB < 20%
    - Acceptable: %CRLB < 50%
    """

    def __init__(
        self,
        noise_estimation: str = "residual",  # 'residual' or 'baseline'
    ) -> None:
        """__init__.

        Args:
            noise_estimation (str): Description.
        """
        super().__init__()
        self.noise_estimation = noise_estimation

    def _estimate_noise(
        self,
        spectrum: Tensor,
        model_spectrum: Tensor | None = None,
    ) -> Tensor:
        """Estimate noise standard deviation."""
        if self.noise_estimation == "residual" and model_spectrum is not None:
            residual = spectrum - model_spectrum
            noise_std = residual.std()
        else:
            # Use noise floor from high-ppm region (typically > 9 ppm is noise)
            noise_region = spectrum[:, -int(spectrum.shape[1] * 0.1) :]
            noise_std = noise_region.std()

        return noise_std.clamp(min=1e-8)

    @jaxtyped
    @beartype
    def forward(
        self,
        spectrum: Float[Tensor, "batch freq"],
        peak_amplitude: Float[Tensor, batch],
        model_spectrum: Float[Tensor, "batch freq"] | None = None,
    ) -> Float[Tensor, ""]:
        """Compute %CRLB for metabolite quantification.

        Args:
            spectrum: Measured spectrum
            peak_amplitude: Estimated metabolite amplitude
            model_spectrum: Fitted model spectrum (for residual noise)

        Returns:
            Mean %CRLB across batch

        forward method for CRLB.

        Executes PyTorch tensor operations.

        Args:
            spectrum (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            peak_amplitude (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            model_spectrum (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        noise_std = self._estimate_noise(spectrum, model_spectrum)

        # Simplified CRLB: the Cramer-Rao lower bound on a peak amplitude scales
        # as the noise standard deviation over the amplitude (i.e. 1/SNR), so the
        # percentage CRLB is ``100 * sigma_noise / A``. (True CRLB needs the full
        # Fisher information matrix.) The previous form divided by ``A`` twice,
        # giving a non-physical ``sigma_noise / A^1.5`` that contradicted the
        # documented "proportional to noise/signal".
        percent_crlb = (noise_std / (peak_amplitude + 1e-8)) * 100.0

        return percent_crlb.mean().clamp(0.0, 200.0)


@register_metric(
    "freq_domain_snr", aliases=["FrequencyDomainSNR"], workflows=frozenset({Regime.SPECTROSCOPIC})
)
class FrequencyDomainSNR(nn.Module):
    """Frequency Domain SNR for spectroscopy.

    Measures signal-to-noise ratio in the frequency domain,
    typically for specific metabolite peaks.
    """

    def __init__(
        self,
        signal_ppm_range: tuple[float, float] = (1.8, 2.2),  # NAA range
        noise_ppm_range: tuple[float, float] = (9.0, 10.0),  # Noise region
    ) -> None:
        """__init__.

        Args:
            signal_ppm_range (tuple[float, float]): Description.
            noise_ppm_range (tuple[float, float]): Description.
        """
        super().__init__()
        self.signal_ppm_range = signal_ppm_range
        self.noise_ppm_range = noise_ppm_range

    def _ppm_to_indices(
        self,
        ppm_range: tuple[float, float],
        freq_axis: Tensor,
    ) -> tuple[int, int]:
        """Convert PPM range to indices."""
        start_idx = (freq_axis >= ppm_range[0]).nonzero(as_tuple=True)[0]
        end_idx = (freq_axis <= ppm_range[1]).nonzero(as_tuple=True)[0]

        if len(start_idx) > 0 and len(end_idx) > 0:
            return start_idx[0].item(), end_idx[-1].item()
        else:
            return 0, freq_axis.shape[0] - 1

    @jaxtyped
    @beartype
    def forward(
        self,
        spectrum: Float[Tensor, "batch freq"],
        freq_axis: Float[Tensor, freq] | None = None,
        ppm_range: tuple[float, float] = (0.0, 10.0),
    ) -> Float[Tensor, ""]:
        """Compute frequency domain SNR.

        Args:
            spectrum: Magnitude spectrum
            freq_axis: PPM axis
            ppm_range: Full PPM range of spectrum

        Returns:
            SNR (higher is better)

        forward method for FrequencyDomainSNR.

        Executes PyTorch tensor operations.

        Args:
            spectrum (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            freq_axis (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ppm_range (tuple[float, float]): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, F = spectrum.shape
        device = spectrum.device

        if freq_axis is None:
            freq_axis = torch.linspace(ppm_range[0], ppm_range[1], F, device=device)

        # Signal region (e.g., NAA peak at ~2.0 ppm)
        sig_start, sig_end = self._ppm_to_indices(self.signal_ppm_range, freq_axis)
        signal_region = spectrum[:, sig_start : sig_end + 1]
        signal_max = signal_region.max(dim=1)[0]

        # Noise region (high ppm, typically no metabolites)
        noise_start, noise_end = self._ppm_to_indices(self.noise_ppm_range, freq_axis)
        noise_region = spectrum[:, noise_start : noise_end + 1]
        noise_std = noise_region.std(dim=1)

        # SNR
        snr = signal_max / (noise_std + 1e-8)

        return snr.mean()
