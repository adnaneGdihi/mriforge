"""Perfusion MRI Quality Metrics.

Implements metrics for ASL and DCE-MRI:
- Wash-in / Wash-out Slope
- K-trans / Ve (Tofts model parameters)
- BAT (Bolus Arrival Time)

Follows unit-test.md rules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mriforge.config.schemas.enums import Regime
from mriforge.core.metrics.registry import register_metric

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


@register_metric("wash_slope", aliases=["WashSlope"], workflows=frozenset({Regime.PERFUSION}))
class WashSlope(nn.Module):
    """Wash-in and Wash-out Slope metrics for DCE-MRI.

    Measures the rate of contrast agent uptake (wash-in) and
    elimination (wash-out) from tissue.
    """

    def __init__(
        self,
        temporal_resolution: float = 1.0,  # seconds per frame
    ) -> None:
        """__init__.

        Args:
            temporal_resolution (float): Description.
        """
        super().__init__()
        self.temporal_resolution = temporal_resolution

    def _find_peak(self, curve: Tensor) -> tuple[int, Tensor]:
        """Find peak enhancement time and value."""
        peak_idx = curve.argmax()
        peak_val = curve[peak_idx]
        return peak_idx.item(), peak_val

    @jaxtyped
    @beartype
    def forward(
        self,
        dce_curve: Float[Tensor, "batch voxels time"],
        baseline_frames: int = 5,
    ) -> dict:
        """Compute wash-in and wash-out slopes.

        Args:
            dce_curve: Time-intensity curves [B, V, T]
            baseline_frames: Number of pre-contrast frames

        Returns:
            Dictionary with wash_in_slope, wash_out_slope, peak_enhancement

        forward method for WashSlope.

        Executes PyTorch tensor operations.

        Args:
            dce_curve (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            baseline_frames (int): Expected input tensor.

        Returns:
            dict: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, V, T = dce_curve.shape

        # Baseline
        baseline = dce_curve[:, :, :baseline_frames].mean(dim=-1, keepdim=True)

        # Enhancement curve
        enhancement = dce_curve - baseline

        # Find peak (per voxel)
        peak_vals, peak_times = enhancement.max(dim=-1)

        # Wash-in slope (baseline to peak)
        wash_in_slopes = []
        wash_out_slopes = []

        for b in range(B):
            for v in range(V):
                peak_t = peak_times[b, v].item()
                peak_v = peak_vals[b, v]

                # Wash-in: from baseline to peak
                if peak_t > baseline_frames:
                    wash_in = peak_v / ((peak_t - baseline_frames) * self.temporal_resolution)
                else:
                    wash_in = torch.tensor(0.0, device=dce_curve.device)

                # Wash-out: from peak to end
                if peak_t < T - 1:
                    end_val = enhancement[b, v, -1]
                    wash_out = (end_val - peak_v) / ((T - 1 - peak_t) * self.temporal_resolution)
                else:
                    wash_out = torch.tensor(0.0, device=dce_curve.device)

                wash_in_slopes.append(wash_in)
                wash_out_slopes.append(wash_out)

        return {
            "wash_in_slope": torch.stack(wash_in_slopes).mean(),
            "wash_out_slope": torch.stack(wash_out_slopes).mean(),
            "peak_enhancement": peak_vals.mean(),
        }


@register_metric("ktrans", aliases=["ToftsModel"], workflows=frozenset({Regime.PERFUSION}))
class ToftsModel(nn.Module):
    """Extended Tofts Model for DCE-MRI pharmacokinetics.

    Estimates K-trans (transfer constant) and Ve (extracellular volume fraction).

    .. math::

        C_t(t) = K^{trans} \\int_0^t C_p(\tau) e^{-\frac{K^{trans}}{v_e}(t-\tau)} d\tau

    Where:
    - :math:`C_t(t)`: Tissue concentration
    - :math:`C_p(t)`: Plasma concentration (AIF)
    - :math:`K^{trans}`: Volume transfer constant
    - :math:`v_e`: Extracellular volume fraction
    """

    def __init__(
        self,
        temporal_resolution: float = 1.0,
        fit_iterations: int = 100,
    ) -> None:
        """__init__.

        Args:
            temporal_resolution (float): Description.
            fit_iterations (int): Description.
        """
        super().__init__()
        self.temporal_resolution = temporal_resolution
        self.fit_iterations = fit_iterations

    def _population_aif(
        self,
        t: Tensor,
        dose: float = 0.1,  # mmol/kg
    ) -> Tensor:
        """Parker population-averaged arterial input function.

        AIF(t) = sum of two Gaussians + exponential decay
        """
        # Parker model parameters (simplified)
        A1, A2 = 0.809, 0.330
        T1, T2 = 0.170, 0.365
        sigma1, sigma2 = 0.055, 0.134
        alpha, beta = 1.050, 0.169
        s, tau = 38.078, 0.483

        # Two Gaussian components
        g1 = A1 / (sigma1 * 2.5066) * torch.exp(-0.5 * ((t - T1) / sigma1) ** 2)
        g2 = A2 / (sigma2 * 2.5066) * torch.exp(-0.5 * ((t - T2) / sigma2) ** 2)

        # Exponential modulated sigmoid
        exp_term = alpha * torch.exp(-beta * t) / (1 + torch.exp(-s * (t - tau)))

        aif = dose * (g1 + g2 + exp_term)
        return aif.clamp(min=0.0)

    @jaxtyped
    @beartype
    def forward(
        self,
        tissue_curve: Float[Tensor, "batch time"],
        aif: Float[Tensor, time] | None = None,
    ) -> dict:
        """Fit Tofts model to tissue curve.

        Args:
            tissue_curve: Tissue concentration curve
            aif: Arterial input function (uses population AIF if None)

        Returns:
            Dictionary with Ktrans, Ve, Kep, and fit quality

        forward method for ToftsModel.

        Executes PyTorch tensor operations.

        Args:
            tissue_curve (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            aif (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, T = tissue_curve.shape
        device = tissue_curve.device

        # Time axis
        t = torch.arange(T, device=device).float() * self.temporal_resolution

        # AIF
        if aif is None:
            aif = self._population_aif(t)

        # Grid search for Ktrans and Ve (simplified fitting)
        ktrans_range = torch.linspace(0.01, 1.0, 20, device=device)
        ve_range = torch.linspace(0.05, 0.8, 20, device=device)

        best_ktrans = torch.zeros(B, device=device)
        best_ve = torch.zeros(B, device=device)
        best_error = torch.full((B,), float("inf"), device=device)

        for kt in ktrans_range:
            for ve in ve_range:
                kep = kt / ve

                # Convolution: Ct(t) = Ktrans * (Cp * exp(-kep*t))
                kernel = kt * torch.exp(-kep * t)
                ct_pred = (
                    torch.nn.functional.conv1d(
                        aif.unsqueeze(0).unsqueeze(0),
                        kernel.flip(0).unsqueeze(0).unsqueeze(0),
                        padding=T - 1,
                    )[:, :, :T].squeeze()
                    * self.temporal_resolution
                )

                # Fit error
                error = ((tissue_curve - ct_pred.unsqueeze(0)) ** 2).mean(dim=1)

                # Update best
                improved = error < best_error
                best_ktrans[improved] = kt
                best_ve[improved] = ve
                best_error[improved] = error[improved]

        return {
            "Ktrans": best_ktrans.mean(),
            "Ve": best_ve.mean(),
            "Kep": (best_ktrans / best_ve).mean(),
            "fit_error": best_error.mean(),
        }


@register_metric("bat", aliases=["BAT"], workflows=frozenset({Regime.PERFUSION}))
class BAT(nn.Module):
    """Bolus Arrival Time (BAT) estimation.

    Detects when contrast agent first arrives at tissue.
    """

    def __init__(
        self,
        threshold_factor: float = 2.0,  # Multiples of baseline std
    ) -> None:
        """__init__.

        Args:
            threshold_factor (float): Description.
        """
        super().__init__()
        self.threshold_factor = threshold_factor

    @jaxtyped
    @beartype
    def forward(
        self,
        dce_curve: Float[Tensor, "batch time"],
        baseline_frames: int = 5,
        temporal_resolution: float = 1.0,
    ) -> Float[Tensor, batch]:
        """Estimate bolus arrival time.

        Args:
            dce_curve: Time-intensity curve
            baseline_frames: Pre-contrast frames
            temporal_resolution: Seconds per frame

        Returns:
            BAT in seconds for each curve

        forward method for BAT.

        Executes PyTorch tensor operations.

        Args:
            dce_curve (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            baseline_frames (int): Expected input tensor.
            temporal_resolution (float): Expected input tensor.

        Returns:
            Float[Tensor, batch]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, T = dce_curve.shape
        device = dce_curve.device

        # Baseline statistics
        baseline = dce_curve[:, :baseline_frames]
        baseline_mean = baseline.mean(dim=1, keepdim=True)
        baseline_std = baseline.std(dim=1, keepdim=True)

        # Threshold
        threshold = baseline_mean + self.threshold_factor * baseline_std

        # Find first crossing
        above_threshold = dce_curve > threshold

        bat = torch.zeros(B, device=device)
        for b in range(B):
            crossings = torch.where(above_threshold[b])[0]
            if len(crossings) > 0:
                bat[b] = crossings[0].float() * temporal_resolution
            else:
                bat[b] = T * temporal_resolution  # No arrival detected

        return bat


# requires_reference=False: a count of negative voxels in ONE perfusion map is a
# no-reference quality measure -- ``forward`` takes no target. The registry default
# is True, so the undeclared contract produced a TypeError for any caller that
# trusted it (#607).
@register_metric(
    "negative_voxels",
    aliases=["NegativeVoxelCount"],
    requires_reference=False,
    workflows=frozenset({Regime.PERFUSION}),
)
class NegativeVoxelCount(nn.Module):
    """Count of negative voxels in perfusion maps.

    Negative values in CBF/CBV maps indicate fitting errors or artifacts.
    """

    @jaxtyped
    @beartype
    def forward(
        self,
        perfusion_map: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, ""]:
        """Count percentage of negative voxels.

        Returns:
            Percentage of negative voxels (lower is better)

        forward method for NegativeVoxelCount.

        Executes PyTorch tensor operations.

        Args:
            perfusion_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        negative_count = (perfusion_map < 0).float().sum()
        total_count = perfusion_map.numel()

        return (negative_count / total_count) * 100.0


@register_metric("cbf_rmse", aliases=["CBFRMSE"], workflows=frozenset({Regime.PERFUSION}))
class CBFRMSE(nn.Module):
    """RMSE of cerebral blood flow maps (ASL / DSC), lower is better."""

    def forward(self, cbf_pred: Tensor, cbf_target: Tensor) -> Tensor:
        """Root-mean-square error between predicted and reference CBF maps."""
        return torch.sqrt(torch.mean((cbf_pred - cbf_target) ** 2))


@register_metric("att_mae", aliases=["ATTMae"], workflows=frozenset({Regime.PERFUSION}))
class ATTMae(nn.Module):
    """Mean-absolute error of arterial transit time (ASL), lower is better."""

    def forward(self, att_pred: Tensor, att_target: Tensor) -> Tensor:
        """Mean absolute error between predicted and reference ATT maps."""
        return (att_pred - att_target).abs().mean()
