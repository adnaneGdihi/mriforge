"""Diffusion MRI (dMRI) Quality Metrics.

Implements metrics specific to diffusion-weighted imaging:
- CC SNR: Corpus Callosum Signal-to-Noise Ratio

Follows unit-test.md rules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spectramr.core.metrics.registry import register_metric

try:
    from beartype import beartype
    from jaxtyping import Float, Int, jaxtyped

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


@register_metric("cc_snr", aliases=["CCSNR"])
class CCSNR(nn.Module):
    """Corpus Callosum Signal-to-Noise Ratio (CC SNR).

    Measures SNR specifically in the corpus callosum region,
    which is a white matter structure commonly used as a
    reference for diffusion MRI quality.

    CC is preferred because:
    1. High FA values (well-aligned fibers)
    2. Reproducible anatomy across subjects
    3. Sensitive to noise and artifacts
    """

    def __init__(
        self,
        noise_method: str = "background",  # 'background' or 'difference'
    ) -> None:
        """
        Args:
            noise_method: Method to estimate noise
                - 'background': Use air/background region
                - 'difference': Use difference of two acquisitions
        """
        super().__init__()
        # No-silent-fallback (CLAUDE.md #9): an unknown noise_method must
        # raise at construction rather than quietly resolving to 'background'
        # in forward().
        if noise_method not in ("background", "difference"):
            raise ValueError(
                f"CCSNR noise_method={noise_method!r} not in ('background', 'difference')."
            )
        self.noise_method = noise_method

    def _estimate_noise_background(
        self,
        image: Tensor,
        background_mask: Tensor | None = None,
    ) -> Tensor:
        """Estimate noise from background region."""
        if background_mask is None:
            # Auto-detect background (low intensity corners)
            threshold = image.mean() * 0.1
            background_mask = image < threshold

        if background_mask.sum() > 0:
            # Background has Rayleigh distribution for magnitude images
            # std = signal / sqrt(2/pi) ≈ signal / 0.655
            background_signal = image[background_mask]
            noise_std = background_signal.std() / 0.655
        else:
            noise_std = image.std() * 0.1  # Fallback

        return noise_std.clamp(min=1e-8)

    def _estimate_noise_difference(
        self,
        image1: Tensor,
        image2: Tensor,
    ) -> Tensor:
        """Estimate noise from difference of two acquisitions."""
        diff = image1 - image2
        # For difference image, noise std = sqrt(2) * single noise std
        noise_std = diff.std() / 1.414
        return noise_std.clamp(min=1e-8)

    @jaxtyped
    @beartype
    def forward(
        self,
        dwi: Float[Tensor, "batch channels height width"],
        cc_mask: Int[Tensor, "batch 1 height width"],
        background_mask: Int[Tensor, "batch 1 height width"] | None = None,
        dwi_repeat: Float[Tensor, "batch channels height width"] | None = None,
    ) -> Float[Tensor, ""]:
        """Compute CC SNR.

        Args:
            dwi: Diffusion-weighted image
            cc_mask: Binary mask of corpus callosum
            background_mask: Binary mask of background (for noise estimation)
            dwi_repeat: Repeat acquisition (for difference-based noise)

        Returns:
            CC SNR value (higher is better)

        forward method for CCSNR.

        Executes PyTorch tensor operations.

        Args:
            dwi (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            cc_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            background_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            dwi_repeat (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Signal in CC region
        cc_mask_bool = cc_mask.bool()
        signal_mean = dwi[cc_mask_bool.expand_as(dwi)].mean()

        # Noise estimation. No-silent-fallback (CLAUDE.md #9): a 'difference'
        # request with no repeat acquisition must raise, not quietly switch to
        # the background estimator (which would report a different SNR).
        if self.noise_method == "difference":
            if dwi_repeat is None:
                raise ValueError(
                    "CCSNR(noise_method='difference') requires a `dwi_repeat` "
                    "acquisition; got None. Pass the repeat acquisition or "
                    "use noise_method='background'."
                )
            noise_std = self._estimate_noise_difference(dwi, dwi_repeat)
        else:
            background_bool = background_mask.bool() if background_mask is not None else None
            noise_std = self._estimate_noise_background(dwi, background_bool)

        # SNR
        snr = signal_mean / noise_std

        # NaN safety
        if not torch.isfinite(snr):
            return torch.tensor(0.0, device=dwi.device)

        return snr


# requires_reference=False: NDC scores the internal consistency of ONE dwi stack
# across its diffusion directions -- ``forward(dwi_stack)`` takes no target. The
# default is True, so leaving it undeclared made every caller that honours the
# registry pass a reference and get a TypeError, which a sweep turns into a
# swallowed NaN (#607).
@register_metric("ndc_diffusion", aliases=["NDC"], requires_reference=False)
class NDC(nn.Module):
    """Neighboring DWI Correlation (NDC).

    Measures correlation between adjacent diffusion directions,
    which should be high for quality data (smooth angular sampling).
    """

    @jaxtyped
    @beartype
    def forward(
        self,
        dwi_stack: Float[Tensor, "batch directions height width"],
    ) -> Float[Tensor, ""]:
        """Compute NDC across diffusion directions.

        Args:
            dwi_stack: Stack of DWI volumes along directions

        Returns:
            Mean neighboring correlation (higher is better, max 1.0)

        forward method for NDC.

        Executes PyTorch tensor operations.

        Args:
            dwi_stack (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = dwi_stack.shape[0]
        D = dwi_stack.shape[1]
        H, W = dwi_stack.shape[-2], dwi_stack.shape[-1]

        correlations = []
        for i in range(D - 1):
            vol1 = dwi_stack[:, i].flatten(1)
            vol2 = dwi_stack[:, i + 1].flatten(1)

            # Pearson correlation
            v1_centered = vol1 - vol1.mean(dim=1, keepdim=True)
            v2_centered = vol2 - vol2.mean(dim=1, keepdim=True)

            corr = (v1_centered * v2_centered).sum(dim=1) / (
                v1_centered.norm(dim=1) * v2_centered.norm(dim=1) + 1e-8
            )
            correlations.append(corr.mean())

        ndc = torch.stack(correlations).mean()
        return ndc.clamp(0.0, 1.0)


# requires_reference=False -- ``forward(dwi_stack)``; see the NDC note above (#607).
@register_metric("spike_percentage", aliases=["SpikePercentage"], requires_reference=False)
class SpikePercentage(nn.Module):
    """Spike Percentage for dMRI.

    Detects outlier volumes in diffusion data that may indicate
    motion artifacts or hardware issues.
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        """__init__.

        Args:
            z_threshold (float): Description.
        """
        super().__init__()
        self.z_threshold = z_threshold

    @jaxtyped
    @beartype
    def forward(
        self,
        dwi_stack: Float[Tensor, "batch directions height width"],
    ) -> Float[Tensor, ""]:
        """Compute spike percentage.

        Returns:
            Percentage of volumes flagged as spikes (lower is better)

        forward method for SpikePercentage.

        Executes PyTorch tensor operations.

        Args:
            dwi_stack (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = dwi_stack.shape[0]
        D = dwi_stack.shape[1]
        H, W = dwi_stack.shape[-2], dwi_stack.shape[-1]

        # Compute mean intensity per volume
        vol_means = dwi_stack.mean(dim=(2, 3))  # [B, D]

        # Z-score across directions
        mean_of_means = vol_means.mean(dim=1, keepdim=True)
        std_of_means = vol_means.std(dim=1, keepdim=True)
        z_scores = (vol_means - mean_of_means) / (std_of_means + 1e-8)

        # Count spikes
        spike_count = (z_scores.abs() > self.z_threshold).float().sum()
        total_vols = B * D

        spike_pct = spike_count / total_vols * 100.0

        return spike_pct
