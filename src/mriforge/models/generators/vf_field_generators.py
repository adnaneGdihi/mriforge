"""Task 3: B0 and B1 Field Extraction Generators.

Specialised generator architectures for extracting quantitative
B0/B1 field maps from the synthetic marker's known physical properties.

All models accept ``marker_prior`` via ``**kwargs`` from the training
strategy.  When available, field extraction is anchored to the marker's
known analytic properties; otherwise falls back to learned pathways.

Models
------
- ``resnet_unwrap``           Phase unwrapping U-Net (marker-seeded)
- ``bloch_siegert_algebraic`` Pure pixel-wise Bloch-Siegert arithmetic
- ``afi_ratio_cnn``           AFI ratio with analytical arccos gate
- ``dam_trigfit``             Differentiable trig-fit LM solver
- ``siamese_epi_tracker``    Dual 1D CNN + NCC for EPI shift
- ``phase_tracking_lstm``    LSTM on FID traces for temporal B0
- ``reciprocity_divisor``    Channel division + Gaussian LPF (marker anchored)
- ``mrf_dict_matcher``       Dictionary matching via batched matmul
- ``bssfp_peak_finder``      2D FFT peak extraction (marker-derived B0)
- ``graph_cut_unwrap``       Phase unwrapping with marker boundary seed

Reference
---------
    Blueprint Task 3 §21–30
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.infrastructure.physics.multi_acquisition import invert_bssfp_banding
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


def _ensure_real_stacked(x: torch.Tensor) -> torch.Tensor:
    """Guarantee real-stacked [B, 2C, H, W] from any input (complex or real)."""
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=1)
    return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared mixin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _Task3Mixin:
    """Default IGenerator abstract method implementations."""

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(x, **kwargs)  # type: ignore[attr-defined]

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # type: ignore[attr-defined]


def _zero_init_conv(layer: nn.Module) -> None:
    """Zero-initialize a Conv2d/Linear so residual identity dominates at init."""
    if hasattr(layer, "weight"):
        nn.init.zeros_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to 2-channel real/imag."""
    return torch.cat([x.real, x.imag], dim=1)


def _extract_marker_phase(marker_prior: torch.Tensor) -> torch.Tensor:
    """Extract phase from marker prior [B, 2, H, W] → [B, 1, H, W]."""
    if torch.is_complex(marker_prior):
        return torch.angle(marker_prior)
    return torch.atan2(marker_prior[:, 1:2], marker_prior[:, 0:1])


class _ConvBlock(nn.Module):
    """Conv-BN-ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ConvBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.block(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 21. ResNet Unwrap (exp_vf_21) — Phase unwrapping U-Net
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="resnet_unwrap", training_mode="reconstruction")
class ResNetUnwrapGenerator(_Task3Mixin, IGenerator, nn.Module):
    r"""Phase unwrapping U-Net with marker-seeded boundary (PRELUDE-Net).

    When marker_prior is available, prepends the unwrapped marker phase
    as a seed channel to guide the U-Net wrap-count estimation.

    .. math::

        \Delta B_0 = \frac{1}{\gamma \Delta TE}
            (\Delta\phi_{wrap} + 2\pi \cdot \text{UNet}(\cos\phi, \sin\phi, \phi_{marker}))

    Args:
        in_channels: Input channels (2 = cos + sin of wrapped phase).
        out_channels: Output channels.
        chans: U-Net feature width.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        chans: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # +1 for optional marker phase channel
        self.enc1 = _ConvBlock(in_channels + 1, chans)
        self.enc2 = _ConvBlock(chans, chans * 2)
        self.enc3 = _ConvBlock(chans * 2, chans * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(chans * 4, chans * 2, 2, stride=2)
        self.dec2 = _ConvBlock(chans * 4, chans * 2)
        self.up1 = nn.ConvTranspose2d(chans * 2, chans, 2, stride=2)
        self.dec1 = _ConvBlock(chans * 2, chans)

        self.head = nn.Conv2d(chans, out_channels, 1)
        _zero_init_conv(self.head)

    @property
    def name(self) -> str:
        return "resnet_unwrap"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for ResNetUnwrapGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")

        # Build seed channel from marker (unwrapped phase or zeros)
        if marker_prior is not None:
            seed = _extract_marker_phase(marker_prior)  # [B, 1, H, W]
            if seed.shape[0] != x.shape[0]:
                seed = seed.expand(x.shape[0], -1, -1, -1)
        else:
            seed = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

        x_in = torch.cat([x, seed], dim=1)

        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        dy, dx = e2.shape[2] - d2.shape[2], e2.shape[3] - d2.shape[3]
        d2 = F.pad(d2, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        dy, dx = e1.shape[2] - d1.shape[2], e1.shape[3] - d1.shape[3]
        d1 = F.pad(d1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.head(d1) + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 22. Bloch-Siegert Algebraic (exp_vf_22)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="bloch_siegert_algebraic", training_mode="reconstruction")
class BlochSiegertAlgebraicGenerator(_Task3Mixin, IGenerator, nn.Module):
    r"""Pure pixel-wise Bloch-Siegert B1+ estimator.

    When marker_prior is available, calibrates K_BS from the marker's
    known flip angle, anchoring the analytical formula.

    .. math::

        B_1^+(\mathbf{r}) = \sqrt{\angle(y_{BS} \cdot y_{base}^*) / K_{BS}}

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.k_bs = nn.Parameter(torch.tensor(1.0))
        # Init alpha to large negative → sigmoid ≈ 0 → b1_raw branch inactive at init
        self.alpha = nn.Parameter(torch.tensor(-5.0))
        self.bias_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1),
        )
        _zero_init_conv(self.bias_net[-1])

    @property
    def name(self) -> str:
        return "bloch_siegert_algebraic"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for BlochSiegertAlgebraicGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")

        if marker_prior is not None:
            # Calibrate K_BS from marker: known phase / known B1²
            marker_phase = _extract_marker_phase(marker_prior)
            marker_mag = (
                marker_prior.abs().mean(dim=1, keepdim=True)
                if torch.is_complex(marker_prior)
                else (marker_prior[:, 0:1].pow(2) + marker_prior[:, 1:2].pow(2)).sqrt()
            )
            k_bs_cal = marker_phase.abs().mean() / (marker_mag.pow(2).mean() + 1e-8)
            k_bs = k_bs_cal.clamp(min=0.1)
        else:
            k_bs = self.k_bs.clamp(min=0.1)

        phase_est = torch.atan2(
            x[:, 1:2] if x.shape[1] > 1 else x[:, 0:1],
            x[:, 0:1],
        )
        b1_raw = torch.sqrt(F.relu(phase_est / k_bs))
        # Normalize b1_raw to have unit RMS so correction magnitude stays O(1)
        b1_rms = b1_raw.pow(2).mean().sqrt().clamp(min=1e-6)
        b1_norm = b1_raw / b1_rms
        bias = self.bias_net(x)
        # Scale B1 correction by 0.1 to keep it as a small residual even at high alpha
        return 0.1 * self.alpha.sigmoid() * b1_norm.expand_as(bias) + bias + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 23. AFI Ratio CNN (exp_vf_23)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="afi_ratio_cnn", training_mode="reconstruction")
class AFIRatioCNN(_Task3Mixin, IGenerator, nn.Module):
    """AFI ratio with analytical arccos Bloch equation gate.

    When marker_prior is available, computes α analytically:
        r = S_TR2 / S_TR1
        α = arccos((r·n - 1) / (n - r))
    and uses the MLP only to refine residual errors.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        hidden_dim: MLP width.
        tr_ratio: TR2/TR1 ratio (n).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 64,
        tr_ratio: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n = nn.Parameter(torch.tensor(tr_ratio))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_dim, 1),  # +1 for α_analytical
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1),
        )
        # Zero-init output layer for identity passthrough at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @property
    def name(self) -> str:
        return "afi_ratio_cnn"

    def _compute_analytical_alpha(self, x: torch.Tensor) -> torch.Tensor:
        """Compute α from AFI ratio: arccos((r·n - 1)/(n - r))."""
        s1 = x[:, 0:1]
        s2 = x[:, 1:2] if x.shape[1] > 1 else x[:, 0:1]
        r = s2 / (s1 + 1e-8)
        n = self.n.abs().clamp(min=1.01)
        arg = (r * n - 1.0) / (n - r + 1e-8)
        arg = arg.clamp(-1.0, 1.0)
        return torch.arccos(arg)  # [B, 1, H, W]

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for AFIRatioCNN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")
        alpha_analytic = self._compute_analytical_alpha(x)
        # Concatenate analytical estimate with input for MLP refinement
        x_aug = torch.cat([x, alpha_analytic], dim=1)

        result = self.mlp(x_aug) + x
        if marker_prior is not None and self.training:
            self._last_marker_anchor = _ensure_real_stacked(marker_prior)

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 24. Double Angle Method Trig-Fit (exp_vf_24)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="dam_trigfit", training_mode="reconstruction")
class DAMTrigFitGenerator(_Task3Mixin, IGenerator, nn.Module):
    r"""Differentiable trigonometric fit for Double Angle Method.

    Uses learnable Levenberg-Marquardt-style optimisation:

    .. math::

        \hat{\alpha} = \arg\min_{\alpha}
            \|y_{2\alpha} - 2 y_\alpha \cos(\alpha)\|_2^2

    When marker_prior is available, initialises α from the marker's
    known flip angle.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_iters: LM optimisation iterations.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_iters: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.alpha_init = nn.Parameter(torch.tensor(math.radians(60.0)))
        self.damping = nn.Parameter(torch.tensor(0.01))
        self.output_proj = nn.Conv2d(1, out_channels, 1)
        _zero_init_conv(self.output_proj)

    @property
    def name(self) -> str:
        return "dam_trigfit"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for DAMTrigFitGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")
        B, C, H, W = x.shape

        s_alpha = x[:, 0:1]
        s_2alpha = x[:, 1:2] if C > 1 else x[:, 0:1]

        # Initialise α from marker if available
        if marker_prior is not None:
            marker_phase = _extract_marker_phase(marker_prior)
            alpha = marker_phase.abs().clamp(min=0.01, max=math.pi)
        else:
            alpha = self.alpha_init.expand(B, 1, H, W).clone()

        for _ in range(self.num_iters):
            r = s_2alpha - 2 * s_alpha * torch.cos(alpha)
            J = 2 * s_alpha * torch.sin(alpha)
            delta = J * r / (J.pow(2) + self.damping.abs() + 1e-8)
            alpha = alpha + delta

        return self.output_proj(alpha) + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 25. Siamese EPI Tracker (exp_vf_25)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="siamese_epi_tracker", training_mode="reconstruction")
class SiameseEPITracker(_Task3Mixin, IGenerator, nn.Module):
    """Dual 1D CNN for EPI geometric shift with marker-derived NCC.

    When marker_prior is available, computes the PE-direction pixel
    shift via normalised cross-correlation between the marker's PE
    profiles.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        features: Feature extraction width.
    """

    # The correction path is a ZERO-INIT residual (``correction(...) + x`` with
    # the last conv zero-initialised), so the UNTRAINED model is exactly identity
    # — by design (residual learning; the shift_head still has live weights and
    # trains). The Tier-2 synthetic-forward probe's identity-collapse heuristic
    # false-positives on this at init, so opt that one check out (CLAUDE.md #10
    # sanctioned per-arm skip). All other probe checks still apply.
    synthetic_forward_probe_skip = {"identity_collapse"}

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels, features, (7, 1), padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, (5, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, (3, 1), padding=(1, 0)),
        )
        self.shift_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(features, 1),
        )
        self.correction = nn.Sequential(
            nn.Conv2d(in_channels + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )
        _zero_init_conv(self.correction[-1])

    @property
    def name(self) -> str:
        return "siamese_epi_tracker"

    def _ncc_shift(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Compute PE shift via NCC along columns."""
        # Average across readout dimension
        x_pe = x.mean(dim=-1)  # [B, C, H]
        r_pe = ref.mean(dim=-1)  # [B, C, H]

        # Cross-correlation via FFT
        X = torch.fft.fft(x_pe, dim=-1)
        R = torch.fft.fft(r_pe, dim=-1)
        cc = torch.fft.ifft(X * R.conj(), dim=-1).real
        # Peak location → shift
        shift_idx = cc.mean(dim=1).argmax(dim=-1).float()  # [B]
        H = x_pe.shape[-1]
        shift = (shift_idx - H // 2) / H * 2  # normalise to [-1, 1]
        return shift

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for SiameseEPITracker.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")

        if marker_prior is not None:
            shift = self._ncc_shift(x, marker_prior)
            shift_map = shift.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        else:
            feat = self.feature_net(x)
            shift = self.shift_head(feat).squeeze(1)  # [B]
            shift_map = (
                shift.unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(x.shape[0], 1, x.shape[2], x.shape[3])
            )

        # Expose the estimated PE shift so the VF strategy can SCORE it against
        # the digital twin's applied displacement (field self-consistency), not
        # just grade the corrected image. Normalised to [-1, 1] like the twin's
        # grid units; the strategy converts both to the same reference.
        self.last_shift_estimate = shift.detach()

        # Identity-skip: correction learns a residual conditioned on the shift estimate.
        # The shift_map encodes the estimated PE displacement for the correction network.
        return self.correction(torch.cat([x, shift_map], dim=1)) + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 26. Phase Tracking LSTM (exp_vf_26)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="phase_tracking_lstm", training_mode="reconstruction")
class PhaseTrackingLSTM(_Task3Mixin, IGenerator, nn.Module):
    """LSTM for temporal B0 drift tracking from FID navigators.

    When marker_prior is available, extracts the per-row temporal
    phase difference as a navigated B0 estimate before RNN processing.

    FIXED: operates along rows (temporal PE lines) not columns.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        hidden_dim: LSTM hidden size.
        num_layers: LSTM depth.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Conv2d(hidden_dim * 2, out_channels, 1)
        _zero_init_conv(self.head)
        # Output gate: init=0 → GRU branch contributes nothing at init
        # (bidirectional GRU hidden states are random at init, leaking through head)
        self._output_gate = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self) -> str:
        return "phase_tracking_lstm"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for PhaseTrackingLSTM.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")
        B, C, H, W = x.shape

        if marker_prior is not None:
            # Extract temporal phase drift from marker along PE (rows)
            marker_phase = _extract_marker_phase(marker_prior)  # [B, 1, H, W]
            input_phase = _extract_marker_phase(x)
            phase_diff = input_phase - marker_phase
            # Concatenate phase_diff with features
            augmented = torch.cat([x, phase_diff.expand_as(x[:, :1])], dim=1)
            h = self.stem(augmented[:, :C])
        else:
            h = self.stem(x)

        # Process each ROW as a temporal sequence (fixed from column)
        h_rnn = h.permute(0, 2, 3, 1).reshape(B * H, W, -1)
        h_rnn, _ = self.rnn(h_rnn)
        h_rnn = h_rnn.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Identity-skip: correction learns a residual conditioned on GRU temporal features.
        # The output gate prevents the randomly-initialized GRU from corrupting the output at init.
        phase_correction = self.head(h_rnn) * self._output_gate.sigmoid()

        # Expose the estimated B0-proportional phase field for VF field-scoring
        # (_score_field_structure grades it against a real B0's spatial structure).
        self.last_field_estimate = phase_correction.detach()
        return phase_correction + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 27. Reciprocity Divisor (exp_vf_27)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="reciprocity_divisor", training_mode="reconstruction")
class ReciprocityDivisor(_Task3Mixin, IGenerator, nn.Module):
    """Channel-wise division + Gaussian LPF for B1- estimation.

    When marker_prior is available, uses the marker's smooth field
    as the reference denominator for exact normalisation.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        sigma: Gaussian kernel sigma.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        sigma: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        k_size = int(6 * sigma + 1) | 1
        self.register_buffer("_kernel", self._build_gaussian_kernel(k_size, sigma))
        self.eps = 1e-3
        self.refine = nn.Conv2d(in_channels, out_channels, 1)
        _zero_init_conv(self.refine)
        # Output gate: init=0 → ratio branch contributes nothing at init
        # (channel-mean denominator creates non-uniform ratios across channels)
        self._output_gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _build_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords.pow(2) / (2 * sigma**2))
        kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        k = self._kernel.to(x.device)
        pad = k.shape[-1] // 2
        channels = []
        for c in range(x.shape[1]):
            sc = F.conv2d(x[:, c : c + 1], k, padding=pad)
            channels.append(sc)
        return torch.cat(channels, dim=1)

    @property
    def name(self) -> str:
        return "reciprocity_divisor"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for ReciprocityDivisor.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")

        x_smooth = self._smooth(x)

        if marker_prior is not None:
            # Use marker as exact reference denominator (ensure real-stacked)
            body_ref = self._smooth(_ensure_real_stacked(marker_prior))
            # Channel count may differ if marker has different resolution; fallback
            if body_ref.shape[1] != x_smooth.shape[1]:
                body_ref = x_smooth.mean(dim=1, keepdim=True).expand_as(x_smooth)
        else:
            body_ref = x_smooth.mean(dim=1, keepdim=True).expand_as(x_smooth)

        ratio = x_smooth / (body_ref + self.eps)
        ratio = (ratio - 1.0).clamp(-5.0, 5.0)
        return self.refine(ratio) * self._output_gate.sigmoid() + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 28. MRF Dictionary Matcher (exp_vf_28)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="mrf_dict_matcher", training_mode="reconstruction")
class MRFDictMatcher(_Task3Mixin, IGenerator, nn.Module):
    """Differentiable MRF dictionary matching via batched matmul.

    Learnable dictionary D is matched against input signal
    via normalised inner product.

    Args:
        in_channels: Input channels (signal length).
        out_channels: Output channels (parameter maps).
        dict_size: Number of dictionary atoms.
        atom_length: Length of each dictionary atom.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dict_size: int = 500,
        atom_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        al = atom_length or in_channels

        # Build physics-grounded Bloch dictionary instead of random noise
        with torch.no_grad():
            bloch_dict = self._build_bloch_dictionary(dict_size, al)

        self.dictionary = nn.Parameter(bloch_dict)
        self.param_map = nn.Linear(dict_size, out_channels)
        # Zero-init param_map so dictionary matching contributes nothing at init
        nn.init.zeros_(self.param_map.weight)
        nn.init.zeros_(self.param_map.bias)
        self._in_ch = in_channels
        self._out_ch = out_channels

    def _build_bloch_dictionary(self, dict_size: int, atom_length: int) -> torch.Tensor:
        """Builds an MRF dictionary using DifferentiableBlochLayer with varying flip angles."""
        import math

        from mriforge.infrastructure.physics.differentiable_bloch import (
            DifferentiableBlochLayer,
        )

        # Log-uniform sampling of T1 and T2 typical ranges
        t1_vals = torch.exp(torch.linspace(math.log(100), math.log(3000), dict_size)).view(
            dict_size, 1, 1, 1
        )
        t2_vals = torch.exp(torch.linspace(math.log(20), math.log(500), dict_size)).view(
            dict_size, 1, 1, 1
        )
        pd_map = torch.ones_like(t1_vals)

        bloch = DifferentiableBlochLayer(sequence_type="SPGR")

        # Transient-like fingerprint via sinusoidally varying flip angles
        fa_schedule = torch.sin(torch.linspace(0.1, torch.pi, atom_length)) * (
            60.0 * math.pi / 180.0
        )

        signals = []
        for fa in fa_schedule:
            sig = bloch(
                t1_map=t1_vals,
                t2_map=t2_vals,
                pd_map=pd_map,
                tr=10.0,
                te=2.0,
                flip_angle_rad=fa.item(),
            )
            signals.append(sig.view(dict_size))

        dict_tensor = torch.stack(signals, dim=1)  # [dict_size, atom_length]
        # Avoid zero vectors
        dict_tensor = dict_tensor + 1e-6
        return dict_tensor

    @property
    def name(self) -> str:
        return "mrf_dict_matcher"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for MRFDictMatcher.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        D_norm = F.normalize(self.dictionary, dim=1)
        x_norm = F.normalize(x_flat, dim=-1)

        scores = torch.matmul(x_norm, D_norm.T)  # [B, HW, dict_size]
        weights = F.softmax(scores * 3.0, dim=-1)
        params = self.param_map(weights)

        result = params.permute(0, 2, 1).reshape(B, self._out_ch, H, W) + x
        if marker_prior is not None and self.training:
            self._last_marker_anchor = _ensure_real_stacked(marker_prior)

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 29. bSSFP Peak Finder (exp_vf_29)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="bssfp_peak_finder", training_mode="reconstruction")
class BSSFPPeakFinder(_Task3Mixin, IGenerator, nn.Module):
    """2D FFT magnitude peak extraction for bSSFP banding B0.

    When marker_prior is available, extracts the dominant banding
    frequency from the marker's power spectrum to estimate ΔB0.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.spec_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )
        _zero_init_conv(self.spec_net[-1])

    @property
    def name(self) -> str:
        return "bssfp_peak_finder"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for BSSFPPeakFinder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")

        if marker_prior is not None:
            # Use marker power spectrum for B0 estimation
            M_fft = fft2c(marker_prior)
            power_real = M_fft.real.pow(2)
            power_imag = M_fft.imag.pow(2)
            power = torch.cat([power_real, power_imag], dim=-3)
        else:
            X_fft = fft2c(x)
            power_real = X_fft.real.pow(2)
            power_imag = X_fft.imag.pow(2)
            power = torch.cat([power_real, power_imag], dim=-3)

        if power.shape[1] != x.shape[1]:
            power = power[:, : x.shape[1]]

        # spec_net processes k-space peaks. Must invert to image space to prevent sinc wave artifacts.
        k_filtered = self.spec_net(power)

        # Convert the 2 output channels into a complex tensor for inversion.
        k_complex = torch.complex(k_filtered[:, 0:1], k_filtered[:, 1:2])

        # Bring the filtered banding peaks back to spatial geometry.
        spatial_band = ifft2c(k_complex)

        # Resplit into 2-channel real/imag format.
        spatial_band_real = torch.cat([spatial_band.real, spatial_band.imag], dim=1)

        return spatial_band_real + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 30. Graph Cut Unwrap (exp_vf_30)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(
    name="graph_cut_unwrap",
    training_mode="reconstruction",
    # ``forward`` unpacks ``B, C, H, W`` unconditionally, so rank 2 is not a
    # preference but a contract: a 3-D input raises on the unpack. Declaring it
    # is what lets the Tier-2 probe constrain the ranks it tries
    # (forward_probe.py) and the spec card compare against the data's rank --
    # both of which SKIP entirely while spatial_dims is None (#1106).
    spatial_dims=(2,),
    # ``_ensure_real_stacked`` coerces complex64 -> [B, 2, H, W] real at the top
    # of forward, so complex input is supported. (graph_unet had the mirror-image
    # bug until 2026-05-12: a wrong accepts_complex=False rejected valid pre_model
    # iFFT chains. An absent declaration gives no signal in EITHER direction.)
    accepts_complex=True,
    # Supervised unwrapping against a phase reference target; both committed arms
    # score a wrap-error metric on unwrapped phase vs that reference.
    requires_paired_data=True,
    # input/output_domain intentionally left unannotated, exactly as
    # ``bssfp_b0_regressor`` below and every other multi-acquisition VF field
    # model in this file. ConcreteVirtualFiducialStrategy IFFTs k-space -> image
    # BEFORE the backbone runs, so this model consumes the strategy's synthesised
    # stack rather than the raw dataset domain, and the static data->model domain
    # check does not apply to it. `domain_inference.KNOWN_IMAGE_OUTPUT_MODELS`
    # is the deliberate SSOT for its default output domain -- that entry is by
    # design, not misplaced knowledge, and moving it here would ALSO make P2 fire
    # and outrank an arm's own `data.domain.output` (#986).
    #
    # output_field_units is likewise omitted ON PURPOSE. ``last_field_estimate``
    # is unwrapped phase, which is only PROPORTIONAL to B0 -- never Hz. The
    # metric<->units lock reads this field to gate ``b0_field_rmse``, so
    # declaring "Hz" would unlock a metric that would then grade a unit mismatch;
    # ``_score_field_structure`` exists precisely because the comparison has to
    # be scale-free (Pearson + scale-fit NRMSE).
)
class GraphCutUnwrapGenerator(_Task3Mixin, IGenerator, nn.Module):
    r"""Graph-based phase unwrapping with marker-seeded boundary conditions.

    Constructs a pixel graph over the wrapped phase image and performs
    differentiable message passing to estimate integer wrap counts.

    Graph structure:
      - **Nodes**: Non-overlapping patch embeddings from the wrapped phase
      - **Edges**: k-nearest-neighbor graph (k=4 ≈ 4-connected neighbors)
      - **Message passing**: ``DifferentiableGraphModule`` with KAN layers
      - **Boundary seed**: Marker phase injected as node features at corners

    Energy minimisation:
      - Data cost: ``D(n) = |φ_wrapped − φ_unwrapped − 2πn|²``
      - Smoothness: learned via graph message passing (replaces explicit Potts)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        chans: Feature width for node embeddings.
        patch_size: Patch size for graph nodes.
        k: k-NN neighbors for graph construction.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        chans: int = 32,
        patch_size: int = 8,
        k: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        # use_marker=False zeroes the phase seed (unseeded PRELUDE — the model
        # already does this when ``marker_prior is None``) → paired baseline is
        # one config knob (pitfall #17).
        self.use_marker = bool(kwargs.get("use_marker", True))
        # zero-init residual ⇒ identity at init by design (learns the residual);
        # opt out of the Tier-2 identity-collapse FP (other probe checks apply).
        self.synthetic_forward_probe_skip = {"identity_collapse"}

        # Node embedding: each patch → feature vector (+1 for marker phase)
        node_dim = (in_channels + 1) * patch_size * patch_size
        self.node_embed = nn.Linear(node_dim, chans)

        # Graph construction + message passing layers
        from mriforge.models.topological.graph_cuts_neural import DifferentiableGraphModule

        self.graph_layers = nn.ModuleList(
            [
                DifferentiableGraphModule(chans, chans, k=k),
                DifferentiableGraphModule(chans, chans, k=k),
            ]
        )
        self.graph_norms = nn.ModuleList(
            [
                nn.LayerNorm(chans),
                nn.LayerNorm(chans),
            ]
        )

        # Readout: node features → per-patch wrap count
        self.wrap_readout = nn.Linear(chans, patch_size * patch_size)

        # Output projection with residual
        self.output_proj = nn.Conv2d(in_channels + 1, out_channels, 1)
        _zero_init_conv(self.output_proj)

    @property
    def name(self) -> str:
        return "graph_cut_unwrap"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for GraphCutUnwrapGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        B, C, H, W = x.shape
        marker_prior = kwargs.get("marker_prior")
        if not self.use_marker:
            marker_prior = None  # unseeded-PRELUDE baseline (no marker seed)

        # 1. Extract marker phase seed
        if marker_prior is not None:
            seed = _extract_marker_phase(marker_prior)
            if seed.shape[0] != B:
                seed = seed.expand(B, -1, -1, -1)
        else:
            seed = torch.zeros(B, 1, H, W, device=x.device)

        # 2. Concatenate input with marker phase seed
        x_in = torch.cat([x, seed], dim=1)  # [B, C+1, H, W]
        C_in = x_in.shape[1]

        # 3. Patchify → graph nodes
        p = self.patch_size
        # Pad if not evenly divisible
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x_in, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x_in.shape[2], x_in.shape[3]
        h_patches = Hp // p
        w_patches = Wp // p
        num_patches = h_patches * w_patches

        patches = x_in.unfold(2, p, p).unfold(3, p, p)  # [B,C+1,h,w,p,p]
        nodes = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches, -1)

        # 4. Embed nodes
        h_feat = self.node_embed(nodes)  # [B, N, chans]

        # 5. Graph message passing with residual connections
        for layer, norm in zip(self.graph_layers, self.graph_norms, strict=False):
            h_feat = norm(layer(h_feat) + h_feat)

        # 6. Readout: wrap counts per pixel
        wraps_flat = self.wrap_readout(h_feat)  # [B, N, p*p]
        wraps = wraps_flat.view(B, h_patches, w_patches, 1, p, p)
        wraps = wraps.permute(0, 3, 1, 4, 2, 5).contiguous()
        wraps = wraps.view(B, 1, Hp, Wp)
        # Remove padding
        wraps = wraps[:, :, :H, :W]

        # Expose the estimated wrap-count field (∝ B0 structure) for VF
        # field-scoring against a real B0 (_score_field_structure).
        self.last_field_estimate = wraps.detach()

        # 7. Output: combine input with learned wraps
        return self.output_proj(torch.cat([x, wraps], dim=1)) + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bssfp_b0_regressor — phase-cycled bSSFP → absolute B0 (Hz)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(
    name="bssfp_b0_regressor",
    training_mode="reconstruction",
    spatial_dims=(2,),
    # input/output_domain intentionally left unannotated (like every sibling
    # multi-acquisition VF field model): this model consumes the strategy's
    # *synthesised* phase-cycled stack, not the raw dataset domain, so the static
    # data→model domain check does not apply. output_field_units IS declared so
    # the metric↔units lock (pitfall #16) still binds.
    accepts_complex=True,
    requires_paired_data=True,
    output_field_units="Hz",
)
class BSSFPB0Regressor(_Task3Mixin, IGenerator, nn.Module):
    """Recover the absolute off-resonance field ΔB0 (Hz) from a phase-cycled
    balanced-SSFP stack.

    A frozen **elliptical-signal-model phase-cycle DFT** prior
    (:func:`invert_bssfp_banding`) extracts ``ΔB0`` analytically from the
    complex stack ``[B, N, H, W]``; a zero-initialised CNN residual then
    refines it (and absorbs the receive-phase calibration constant). At init
    the residual is identity so the prior dominates — an untrained model cannot
    underperform the closed form (the analytic limit). ``last_field_estimate``
    is a B0 **field in Hz**, never a corrected image (the ``bssfp_peak_finder``
    facade this arm replaces — pitfall #16). The acquisition parameters
    (TR, T1, T2, flip) feed the DFT calibration exactly as the synthesis used
    them; a small mismatch only shifts a global constant the residual learns.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        *,
        tr_ms: float = 5.0,
        t1_ms: float = 400.0,
        t2_ms: float = 80.0,
        flip_deg: float = 60.0,
        hidden: int = 32,
        use_dft_prior: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tr_ms = tr_ms
        self.t1_ms = t1_ms
        self.t2_ms = t2_ms
        self.flip_rad = math.radians(flip_deg)
        # The Δf=0 / no-mechanism baseline (pitfall #17): with the phase-cycle
        # DFT prior off, the model sees only magnitude (no banding phase) and
        # cannot resolve absolute off-resonance (identifiability §2) — the
        # control the real arm must beat. A genuinely wired knob (pitfall #15).
        self.use_dft_prior = use_dft_prior
        # Residual head over [ΔB0_dft, mean-magnitude] (both real, 2 channels).
        self.residual = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )
        _zero_init_conv(self.residual[-1])  # untrained → prior dominates (Rec 3)
        self.last_field_estimate: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "bssfp_b0_regressor"

    @staticmethod
    def _to_complex_stack(x: torch.Tensor) -> torch.Tensor:
        """Coerce input to a complex phase-cycled stack ``[B, N, H, W]``.

        Accepts a complex stack directly, or a real interleaved ``[B, 2N, H, W]``
        (first N real channels, next N imag — the 2C-real = C-complex convention).
        """
        if torch.is_complex(x):
            return x
        c = x.shape[1]
        if c % 2 != 0:
            raise ValueError(
                f"bssfp_b0_regressor expects a complex stack or an even real "
                f"channel count (2N interleaved), got {c} real channels."
            )
        n = c // 2
        return torch.complex(x[:, :n], x[:, n:])

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stack = self._to_complex_stack(x)
        mag = stack.abs().mean(dim=1, keepdim=True)
        if self.use_dft_prior:
            # Frozen ESM phase-cycle DFT prior — absolute ΔB0 in Hz (no params).
            b0_dft = invert_bssfp_banding(
                stack,
                tr_ms=self.tr_ms,
                t1_ms=self.t1_ms,
                t2_ms=self.t2_ms,
                flip_rad=self.flip_rad,
            )
        else:
            # No-mechanism baseline: zero prior → only magnitude reaches the head.
            b0_dft = torch.zeros_like(mag)
        est = b0_dft + self.residual(torch.cat([b0_dft, mag], dim=1))
        self.last_field_estimate = est.detach()
        return est


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# __all__
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    "AFIRatioCNN",
    "BSSFPB0Regressor",
    "BSSFPPeakFinder",
    "BlochSiegertAlgebraicGenerator",
    "DAMTrigFitGenerator",
    "GraphCutUnwrapGenerator",
    "MRFDictMatcher",
    "PhaseTrackingLSTM",
    "ReciprocityDivisor",
    "ResNetUnwrapGenerator",
    "SiameseEPITracker",
]
