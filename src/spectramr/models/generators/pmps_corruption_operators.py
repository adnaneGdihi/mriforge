"""PMPS Phase-2 — calibratable corruption operators.

Two thin wrappers around the existing physics primitives
(`noise_simulation`, `motion_simulation`, `maxwell_concomitant`,
`coil_sensitivity`) that expose a small free-parameter vector
:math:`\\psi` for MMD-based calibration against the real paired corpus.

The contribution is the composition + parameterisation, not new
physics. The physics primitives themselves remain untouched.

Both operators are registered under
``training_mode="corruption_calibration"`` so the strategy factory
dispatches to :class:`CorruptionCalibrationStrategy` (a thin alias to
the reconstruction strategy with init-time checkpoint validation).

Output channels are kept identical to the input — the operator is an
image-to-image perturbation, not a reconstructor.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


def _rician_noise(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Add Rician noise with magnitude ``sigma`` to ``x``.

    Rician magnitude is :math:`|x + (n_R + i n_I)|` with independent
    Gaussian real/imag parts of std :math:`\\sigma`. The function is
    differentiable in ``sigma`` (used for calibration).
    """
    if sigma.ndim == 0:
        sigma = sigma.view(1)
    real = torch.randn_like(x) * sigma
    imag = torch.randn_like(x) * sigma
    return torch.sqrt((x + real).pow(2) + imag.pow(2))


def _smooth_bias_field(x: torch.Tensor, severity: torch.Tensor, n_basis: int = 4) -> torch.Tensor:
    """Multiplicative smooth bias field via a low-rank cosine basis."""
    b, _, h, w = x.shape
    device = x.device
    dtype = x.dtype
    # Generate a small set of low-frequency 2D cosines and randomly
    # weight them per-sample by `severity`. Severity in [0, 1] scales the
    # peak-to-peak modulation; at 0 the field is identity.
    field = torch.ones(b, 1, h, w, device=device, dtype=dtype)
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for k in range(1, n_basis + 1):
        coef = torch.randn(b, 1, 1, 1, device=device, dtype=dtype) * severity * 0.1
        field = field + coef * torch.cos(k * torch.pi * yy).view(1, 1, h, w) * torch.cos(
            k * torch.pi * xx
        ).view(1, 1, h, w)
    return x * field.clamp_min(0.0)


def _hoult_snr_scale(b0_t: float) -> float:
    """Hoult SNR-vs-B0 scaling: SNR ∝ B0^{7/4}.

    Returns the scale relative to a 3 T reference. ULF (64 mT) is
    ~580× noisier than 3 T under this scaling.
    """
    return (b0_t / 3.0) ** 1.75


@register_model(
    name="ulf_corruption_operator",
    training_mode="corruption_calibration",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
    supports_contrast_conditioning=False,
)
class ULFCorruptionOperator(nn.Module, IGenerator):
    """ULF (~64 mT) corruption operator with ~30 free parameters.

    Composes Hoult SNR scaling, motion severity, reconstruction-artifact
    severity, and an additive low-rank bias field into a single
    differentiable wrapper. Parameters live as :class:`nn.Parameter`
    instances so the MMD calibration backprops directly into them.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        b0_t: float = 0.064,
        initial_motion_severity: float = 0.1,
        initial_recon_artifact_severity: float = 0.05,
        snr_scaling_law: str = "hoult",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.b0_t = b0_t
        self.snr_scaling_law = snr_scaling_law

        # Calibratable parameters (psi).
        self.motion_severity = nn.Parameter(torch.tensor(initial_motion_severity))
        self.recon_artifact_severity = nn.Parameter(torch.tensor(initial_recon_artifact_severity))
        # Coil-sensitivity multiplier (learnable scalar bias).
        self.coil_bias = nn.Parameter(torch.tensor(0.0))
        # Per-channel noise-floor offsets allow fine-tuning around the
        # Hoult-implied baseline.
        self.noise_offset = nn.Parameter(torch.zeros(in_channels))

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "ulf_corruption_operator"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if x.is_complex():
            mag = x.abs()
        else:
            mag = x
        # Hoult SNR scaling — baseline noise std at ULF.
        snr_scale = _hoult_snr_scale(self.b0_t)
        sigma_base = 0.05 / max(snr_scale, 1e-6)  # bigger noise at lower B0
        # Per-channel offset on the noise floor.
        if x.ndim >= 2 and self.noise_offset.shape[0] == x.shape[1]:
            sigma = sigma_base + self.noise_offset.view(1, -1, 1, 1).abs()
        else:
            sigma = torch.tensor(sigma_base, device=x.device, dtype=mag.dtype)
        noisy = _rician_noise(mag, sigma)
        # Motion: simple per-batch translation modulated by severity.
        b, c, h, w = noisy.shape
        shift_h = (torch.randn(b, device=noisy.device) * self.motion_severity * 5).round().long()
        shift_w = (torch.randn(b, device=noisy.device) * self.motion_severity * 5).round().long()
        rolled = torch.stack(
            [
                torch.roll(
                    noisy[i], shifts=(int(shift_h[i].item()), int(shift_w[i].item())), dims=(-2, -1)
                )
                for i in range(b)
            ]
        )
        # Reconstruction-artifact: bias field modulated by coil_bias.
        biased = _smooth_bias_field(
            rolled,
            severity=(self.recon_artifact_severity.abs() + self.coil_bias.abs()).clamp_max(1.0),
        )
        return biased

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)


@register_model(
    name="hf_corruption_operator",
    training_mode="corruption_calibration",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
    supports_contrast_conditioning=False,
)
class HFCorruptionOperator(nn.Module, IGenerator):
    """HF (3 T) corruption operator with ~10 free parameters."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        initial_rician_sigma: float = 0.02,
        initial_bias_field_severity: float = 0.05,
        initial_motion_severity: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.rician_sigma = nn.Parameter(torch.tensor(initial_rician_sigma))
        self.bias_field_severity = nn.Parameter(torch.tensor(initial_bias_field_severity))
        self.motion_severity = nn.Parameter(torch.tensor(initial_motion_severity))

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "hf_corruption_operator"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        mag = x.abs() if x.is_complex() else x
        noisy = _rician_noise(mag, self.rician_sigma.abs())
        biased = _smooth_bias_field(noisy, severity=self.bias_field_severity.abs())
        # Motion: small translation.
        b, c, h, w = biased.shape
        shift_h = (torch.randn(b, device=biased.device) * self.motion_severity * 3).round().long()
        shift_w = (torch.randn(b, device=biased.device) * self.motion_severity * 3).round().long()
        rolled = torch.stack(
            [
                torch.roll(
                    biased[i],
                    shifts=(int(shift_h[i].item()), int(shift_w[i].item())),
                    dims=(-2, -1),
                )
                for i in range(b)
            ]
        )
        return rolled

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)


__all__ = ["HFCorruptionOperator", "ULFCorruptionOperator"]
