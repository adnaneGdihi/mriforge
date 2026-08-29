"""K-Space-Native Virtual Fiducial Operators.

Five Fourier-domain correction operators that diagnose and repair
scanner-level physics degradations *natively* in k-space using complex-valued
convolutions and the synthetic marker as an absolute calibration standard.

Models
------
- ``kspace_phase_ramp``       Phase-Ramp Conjugator (inter-shot translation)
- ``mtf_deapodization``       MTF Transversal De-Apodization (T2* decay)
- ``phase_navigator_lock``    0th-Order Phase Navigator Lock (B0 drift)
- ``marker_grappa_conv``      Marker-Anchored k-Space GRAPPA (aliasing)
- ``diff_trajectory_opt``     Differentiable Trajectory Optimizer (grad delays)

Reference
---------
    Blueprint: k-Space-native marker-anchored correction operators.
    All operators use the Convolution Theorem to isolate the marker
    in the Fourier domain via W(k) ⊛ Y_total(k).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.layers.complex_conv import ComplexConv2d
from mriforge.models.layers.complex_layers import ComplexBatchNorm2d, ComplexReLU
from mriforge.models.registry import register_model


def _to_ri(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split input into (real, imag) pair.

    Accepts:
      - Complex [B, C, H, W] → split into real/imag
      - Real-stacked [B, 2C, H, W] → split channels evenly
    """
    if torch.is_complex(x):
        return x.real, x.imag
    # Real-stacked: first half real, second half imag
    C = x.shape[1]
    return x[:, : C // 2], x[:, C // 2 :]


def _from_ri(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Merge (real, imag) back to 2-channel real-stacked [B, 2C, H, W]."""
    return torch.cat([real, imag], dim=1)


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert any input format to native complex64."""
    if torch.is_complex(x):
        return x
    C = x.shape[1]
    return torch.complex(x[:, : C // 2], x[:, C // 2 :])


def _zero_init_complex_conv(layer: ComplexConv2d) -> None:
    """Zero-initialize a ComplexConv2d so residual identity dominates at init."""
    nn.init.zeros_(layer.weight_real)
    nn.init.zeros_(layer.weight_imag)
    if hasattr(layer, "bias_real") and layer.bias_real is not None:
        nn.init.zeros_(layer.bias_real)
    if hasattr(layer, "bias_imag") and layer.bias_imag is not None:
        nn.init.zeros_(layer.bias_imag)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared IGenerator mixin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _KSpaceMixin:
    """Default IGenerator abstract method implementations."""

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(x, **kwargs)  # type: ignore[attr-defined]

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # type: ignore[attr-defined]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared: k-Space Marker Isolation via Convolution Theorem
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _KSpaceMarkerIsolator(nn.Module):
    r"""Isolate the corrupted marker frequency spectrum from total k-space.

    Uses the Convolution Theorem: multiplying by a spatial bounding-box
    mask w(r) in image space ≡ convolving with W(k) = F{w} in k-space.

    .. math::

        \hat{Y}_m(k) = W(k) \circledast Y_{total}(k)

    For computational efficiency, this is implemented as:
    W(k) ⊛ Y(k) = F{ F^{-1}{Y(k)} · w(r) }

    Args:
        corner_offset: Normalised corner position in [-1, 1].
        window_sigma: Gaussian window sigma for smooth extraction.
    """

    def __init__(
        self,
        corner_offset: float = 0.8,
        window_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.corner_offset = corner_offset
        self.window_sigma = window_sigma
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_window(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or create Gaussian corner window for given dimensions."""
        key = (H, W)
        if key not in self._cache or self._cache[key].device != device:
            yy = torch.linspace(-1, 1, H, device=device)
            xx = torch.linspace(-1, 1, W, device=device)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

            mask = torch.zeros(1, 1, H, W, device=device)
            for sy in [-1, 1]:
                for sx in [-1, 1]:
                    cy = sy * self.corner_offset
                    cx = sx * self.corner_offset
                    dist_sq = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
                    mask += torch.exp(-dist_sq / (2 * self.window_sigma**2))
            mask = mask / mask.max()
            self._cache[key] = mask
        return self._cache[key]

    def isolate_marker_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        """Extract marker-only k-space via convolution theorem.

        Accepts complex [B, C, H, W] or real-stacked [B, 2C, H, W].
        Returns same format as input.

        Args:
            kspace: Total k-space [B, C, H, W].

        Returns:
            Marker-isolated k-space [B, C, H, W].
        """
        is_real = not torch.is_complex(kspace)
        if is_real:
            kspace_c = _to_complex(kspace)
        else:
            kspace_c = kspace

        H, W = kspace_c.shape[-2:]
        window_mask = self._get_window(H, W, kspace_c.device)

        # W(k) ⊛ Y(k) = F{ F^{-1}{Y(k)} · w(r) }
        image = ifft2c(kspace_c)
        masked_image = image * window_mask
        marker_kspace = fft2c(masked_image)

        if is_real:
            return _from_ri(marker_kspace.real, marker_kspace.imag)
        return marker_kspace


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared: Complex Convolution Block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _ComplexConvBlock(nn.Module):
    """ComplexConv2d + ComplexBatchNorm2d + ComplexReLU (modReLU)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.conv = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            **kwargs,
        )
        self.norm = ComplexBatchNorm2d(out_channels)
        self.act = ComplexReLU()

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """forward method for _ComplexConvBlock.

        Executes PyTorch tensor operations.

        Args:
            real (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            imag (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        real, imag = self.conv(real, imag)
        real, imag = self.norm(real, imag)
        real, imag = self.act(real, imag)
        return real, imag


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 31. k-Space Phase-Ramp Conjugator (exp_vf_31)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="kspace_phase_ramp", training_mode="reconstruction")
class KSpacePhaseRampConjugator(_KSpaceMixin, IGenerator, nn.Module):
    r"""k-Space Phase-Ramp Conjugator for inter-shot translation correction.

    Extracts the 2D phase difference between the isolated marker k-space
    and the analytical prior, fits a linear phase plane to recover the
    translation vector Δr, then applies the conjugate phase ramp:

    .. math::

        \mathcal{O}_{mot}(Y) = Y(k) \odot \exp(+i 2\pi k \cdot \Delta r)

    Uses complex-valued convolutions throughout to preserve phase structure.
    The ``_KSpaceMarkerIsolator`` extracts the marker's frequency spectrum
    via the Convolution Theorem for shift estimation.

    Reference: Batchelor et al., MRM 2005.

    Args:
        in_channels: Input channels (per complex component).
        out_channels: Output channels (per complex component).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        ch = in_channels // 2 if in_channels > 1 else 1
        self.isolator = _KSpaceMarkerIsolator()
        # Complex shift estimator: extract Δr from marker k-space
        self.shift_block = _ComplexConvBlock(ch, 16, 3)
        self.shift_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 2),  # (Δy, Δx) in normalised coords
        )
        # Gate for gradual activation (init=0 → identity)
        self.gate = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self) -> str:
        return "kspace_phase_ramp"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for KSpacePhaseRampConjugator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_real, x_imag = _to_ri(x)
        B, C, H, W = x_real.shape

        # 1. Isolate marker k-space
        marker_prior = kwargs.get("marker_prior")
        if marker_prior is not None:
            mp_complex = _to_complex(marker_prior)
            mp_k = fft2c(mp_complex)
            # Phase difference: corrupted_marker / marker_prior
            x_k = fft2c(torch.complex(x_real, x_imag))
            mk_k = self.isolator.isolate_marker_kspace(x_k)
            # Feed phase difference magnitude to shift estimator
            phase_diff = mk_k * mp_k.conj()
            shift_input_r, shift_input_i = phase_diff.real, phase_diff.imag
        else:
            shift_input_r, shift_input_i = x_real, x_imag

        # 2. Complex conv to extract shift parameters
        feat_r, feat_i = self.shift_block(shift_input_r, shift_input_i)
        # Use magnitude for pooling (real-valued shift)
        feat_mag = torch.sqrt(feat_r**2 + feat_i**2 + 1e-8)
        shift = self.shift_head(feat_mag)  # [B, 2]
        alpha = self.gate.sigmoid()
        delta_y = shift[:, 0:1] * alpha  # gated
        delta_x = shift[:, 1:2] * alpha

        # 3. Construct conjugate phase ramp
        ky = torch.linspace(-0.5, 0.5, H, device=x_real.device)
        kx = torch.linspace(-0.5, 0.5, W, device=x_real.device)
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

        phase = (
            2
            * math.pi
            * (
                kx_grid.unsqueeze(0) * delta_x.unsqueeze(-1)
                + ky_grid.unsqueeze(0) * delta_y.unsqueeze(-1)
            )
        )  # [B, H, W]
        phase = phase.unsqueeze(1)  # [B, 1, H, W]

        # 4. Apply complex rotation: x * exp(+iφ)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        out_real = x_real * cos_p - x_imag * sin_p
        out_imag = x_real * sin_p + x_imag * cos_p

        return _from_ri(out_real, out_imag)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 32. MTF Transversal De-Apodization Operator (exp_vf_32)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="mtf_deapodization", training_mode="reconstruction")
class MTFDeApodizationOperator(_KSpaceMixin, IGenerator, nn.Module):
    r"""Modulation Transfer Function Equalizer in k-space.

    Computes the amplitude decay envelope H_MTF(k) by comparing the
    isolated marker magnitude against the analytical prior. Applies a
    Tikhonov-regularized inverse to boost suppressed high frequencies:

    .. math::

        \mathcal{O}_{MTF}(Y) = Y(k) \odot \frac{H^*_{MTF}(k)}
        {|H_{MTF}(k)|^2 + \lambda_{noise}}

    Uses complex convolutions to estimate H_MTF from marker k-space.

    Reference: Campisi et al., Blind Image Deconvolution, CRC Press 2007.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        lambda_noise: Tikhonov regularisation parameter.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        lambda_noise: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # use_marker=False forces the no-MTF-estimate path (no de-apodization —
        # the model already passes k-space through when ``marker_prior is None``)
        # → paired baseline is one config knob (pitfall #17).
        self.use_marker = bool(kwargs.get("use_marker", True))
        # zero-init residual ⇒ identity at init by design (learns the residual);
        # opt out of the Tier-2 identity-collapse FP (other probe checks apply).
        self.synthetic_forward_probe_skip = {"identity_collapse"}
        ch = in_channels // 2 if in_channels > 1 else 1
        self.lambda_noise = nn.Parameter(torch.tensor(lambda_noise))
        self.isolator = _KSpaceMarkerIsolator()

        # Complex envelope estimator (readout-oriented kernels)
        self.mtf_block1 = _ComplexConvBlock(ch, 16, kernel_size=3)
        self.mtf_block2 = _ComplexConvBlock(16, 16, kernel_size=3)
        self.mtf_head = ComplexConv2d(16, ch, 1)
        _zero_init_complex_conv(self.mtf_head)

        # Complex refinement after de-apodization
        self.refine = _ComplexConvBlock(ch, ch, 3)
        self.refine_head = ComplexConv2d(ch, ch, 1)
        _zero_init_complex_conv(self.refine_head)

        # Gate (init=0 → identity)
        self.gate = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self) -> str:
        return "mtf_deapodization"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for MTFDeApodizationOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_real, x_imag = _to_ri(x)

        # 1. Isolate marker k-space for MTF estimation
        marker_prior = kwargs.get("marker_prior")
        if not self.use_marker:
            marker_prior = None  # no-de-apodization baseline (no marker MTF)
        if marker_prior is not None:
            x_k = fft2c(torch.complex(x_real, x_imag))
            mk_k = self.isolator.isolate_marker_kspace(x_k)
            mp_k = fft2c(_to_complex(marker_prior))
            mp_k_isolated = self.isolator.isolate_marker_kspace(mp_k)
            # MTF H(k) = Y_marker / M_prior  (proper complex division)
            denom = mp_k_isolated.real**2 + mp_k_isolated.imag**2 + 1e-8
            ratio_r = (mk_k.real * mp_k_isolated.real + mk_k.imag * mp_k_isolated.imag) / denom
            ratio_i = (mk_k.imag * mp_k_isolated.real - mk_k.real * mp_k_isolated.imag) / denom
        else:
            ratio_r, ratio_i = x_real, x_imag

        # 2. Estimate MTF envelope via complex conv
        h_r, h_i = self.mtf_block1(ratio_r, ratio_i)
        h_r, h_i = self.mtf_block2(h_r, h_i)
        h_r, h_i = self.mtf_head(h_r, h_i)

        # 3. Tikhonov-regularized inverse: H* / (|H|² + λ)
        h_mag_sq = h_r**2 + h_i**2
        lambda_reg = F.softplus(self.lambda_noise)
        denom = h_mag_sq + lambda_reg
        inv_r = h_r / denom
        inv_i = -h_i / denom  # conjugate

        # 4. Apply de-apodization (gated complex multiplication)
        alpha = self.gate.sigmoid()
        # x * (1 + α*(inv - 1))
        boost_r = 1.0 + alpha * (inv_r - 1.0)
        boost_i = alpha * inv_i
        out_r = x_real * boost_r - x_imag * boost_i
        out_i = x_real * boost_i + x_imag * boost_r

        # 5. Complex refinement + global residual
        ref_r, ref_i = self.refine(out_r, out_i)
        ref_r, ref_i = self.refine_head(ref_r, ref_i)
        return _from_ri(ref_r + x_real, ref_i + x_imag)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 33. 0th-Order Phase Navigator Lock (exp_vf_33)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="phase_navigator_lock", training_mode="reconstruction")
class PhaseNavigatorLock(_KSpaceMixin, IGenerator, nn.Module):
    r"""0th-Order Phase Navigator Lock for B0 drift correction.

    Extracts the global phase scalar φ₀ for each shot via matched
    filtering between the isolated marker k-space and the prior:

    .. math::

        \phi_0 = \angle \left( \sum_k \hat{Y}_m(k) \cdot M^*_{prior}(k) \right)

    Demodulates the entire k-space tensor: Y(k) · exp(-iφ₀).
    Uses complex convolutions for robust phase extraction.

    Reference: Korin et al., MRM 1989.

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
        ch = in_channels // 2 if in_channels > 1 else 1
        self.isolator = _KSpaceMarkerIsolator()

        # Complex phase estimator
        self.phase_block = _ComplexConvBlock(ch, 16, 5)
        self.phase_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Tanh(),  # bound to [-1, 1], scaled by π
        )
        # Gate (init=0 → identity demodulation)
        self.phase_gate = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self) -> str:
        return "phase_navigator_lock"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for PhaseNavigatorLock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_real, x_imag = _to_ri(x)
        B, C, H, W = x_real.shape

        # 1. Extract phase from marker comparison
        marker_prior = kwargs.get("marker_prior")
        if marker_prior is not None:
            x_k = fft2c(torch.complex(x_real, x_imag))
            mk_k = self.isolator.isolate_marker_kspace(x_k)
            mp_k = fft2c(_to_complex(marker_prior))
            # Matched filter: inner product
            matched = mk_k * mp_k.conj()
            phase_input_r, phase_input_i = matched.real, matched.imag
        else:
            phase_input_r, phase_input_i = x_real, x_imag

        # 2. Complex feature extraction → magnitude for phase estimation
        feat_r, feat_i = self.phase_block(phase_input_r, phase_input_i)
        feat_mag = torch.sqrt(feat_r**2 + feat_i**2 + 1e-8)
        phi_0 = self.phase_head(feat_mag) * math.pi  # [B, 1]
        phi_0 = phi_0 * self.phase_gate.sigmoid()
        phi_0 = phi_0.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

        # 3. Global complex demodulation: exp(-iφ₀)
        cos_phi = torch.cos(phi_0)
        sin_phi = torch.sin(phi_0)
        out_real = x_real * cos_phi + x_imag * sin_phi
        out_imag = -x_real * sin_phi + x_imag * cos_phi

        return _from_ri(out_real, out_imag)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 34. Marker-Anchored k-Space GRAPPA Convolution (exp_vf_34)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="marker_grappa_conv", training_mode="reconstruction")
class MarkerGRAPPAConvolution(_KSpaceMixin, IGenerator, nn.Module):
    r"""Marker-Anchored k-Space GRAPPA Convolution Kernel.

    Learns shift-invariant k-space interpolation kernels W by minimizing
    the error between the undersampled marker and the noiseless prior:

    .. math::

        \hat{W} = \arg\min_W \| M_{prior}(k) -
        \sum_j W_{j} \circledast \hat{Y}_{m,j}(k_{sampled}) \|_2^2

    Uses **linear** complex convolutions (no activation between layers)
    because GRAPPA kernels are inherently linear operators. Non-linear
    activations in k-space would violate the shift-invariance property.

    Reference: Griswold et al., MRM 2002.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: GRAPPA convolution kernel size.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        kernel_size: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # use_marker=False forces the native-calibration path (no marker ACS —
        # the model already uses the input k-space when ``marker_prior is None``)
        # → paired baseline is one config knob (pitfall #17).
        self.use_marker = bool(kwargs.get("use_marker", True))
        # zero-init residual ⇒ identity at init by design (learns the residual);
        # opt out of the Tier-2 identity-collapse FP (other probe checks apply).
        self.synthetic_forward_probe_skip = {"identity_collapse"}
        ch = in_channels // 2 if in_channels > 1 else 1
        self.isolator = _KSpaceMarkerIsolator()

        # LINEAR complex k-space convolution kernels (no activation)
        # GRAPPA kernels are linear operators — ReLU would destroy k-space structure
        self.grappa_k1 = ComplexConv2d(
            ch,
            16,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.grappa_k2 = ComplexConv2d(
            16,
            16,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.grappa_head = ComplexConv2d(16, ch, 1)
        _zero_init_complex_conv(self.grappa_head)

        # Gate (init=0 → identity)
        self.gate = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self) -> str:
        return "marker_grappa_conv"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for MarkerGRAPPAConvolution.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_real, x_imag = _to_ri(x)

        # 1. Isolate marker for calibration target
        marker_prior = kwargs.get("marker_prior")
        if not self.use_marker:
            marker_prior = None  # native-calibration baseline (no marker ACS)
        if marker_prior is not None:
            x_k = fft2c(torch.complex(x_real, x_imag))
            mk_k = self.isolator.isolate_marker_kspace(x_k)
            # Use marker calibration as input conditioning
            calib_r, calib_i = mk_k.real, mk_k.imag
        else:
            calib_r, calib_i = x_real, x_imag

        # 2. Linear complex GRAPPA kernels (NO activation — intentional)
        g_r, g_i = self.grappa_k1(calib_r, calib_i)
        g_r, g_i = self.grappa_k2(g_r, g_i)
        synth_r, synth_i = self.grappa_head(g_r, g_i)

        # 3. Apply synthesized correction (gated)
        alpha = self.gate.sigmoid()
        out_r = x_real + alpha * synth_r
        out_i = x_imag + alpha * synth_i

        return _from_ri(out_r, out_i)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 35. Differentiable Trajectory Optimizer (exp_vf_35)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="diff_trajectory_opt", training_mode="reconstruction")
class DifferentiableTrajectoryOptimizer(_KSpaceMixin, IGenerator, nn.Module):
    r"""Differentiable Trajectory Optimizer for gradient delay correction.

    Treats the trajectory deviations Δk(t) as learnable parameters
    and optimises them to minimise the L2 distance between the measured
    marker k-space and the continuous analytical Fourier equation:

    .. math::

        \Delta\hat{k}(t) = \arg\min_{\Delta k} \sum_t
        \| \hat{Y}_m(t) - \mathcal{F}_{cont}\{m_{prior}\}
        (k_{nom}(t) + \Delta k(t)) \|_2^2

    Uses complex convolutions for spectral feature extraction and
    1D temporal CNN for per-readout-line trajectory corrections.

    Reference: Vannesjo et al., MRM 2016.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        hidden_dim: Temporal CNN hidden dimension.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        ch = in_channels // 2 if in_channels > 1 else 1
        self.isolator = _KSpaceMarkerIsolator()

        # Complex feature extractor for marker k-space
        self.marker_block = _ComplexConvBlock(ch, 16, 3)

        # Temporal 1D CNN: predict per-readout-line Δk corrections
        # Input: magnitude of complex features averaged over readout
        self.trajectory_net = nn.Sequential(
            nn.Conv1d(16, hidden_dim, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 2, 3, padding=1),  # Δky, Δkx per line
            nn.Tanh(),
        )
        # Correction amplitude gate (init=0 → no correction)
        self.correction_gate = nn.Parameter(torch.tensor(0.0))
        # Complex refinement after trajectory correction
        self.refine = _ComplexConvBlock(ch, ch, 3)
        self.refine_head = ComplexConv2d(ch, ch, 1)
        _zero_init_complex_conv(self.refine_head)

    @property
    def name(self) -> str:
        return "diff_trajectory_opt"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for DifferentiableTrajectoryOptimizer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_real, x_imag = _to_ri(x)
        B, C, H, W = x_real.shape
        alpha = self.correction_gate.sigmoid()

        # 1. Extract marker calibration features
        marker_prior = kwargs.get("marker_prior")
        if marker_prior is not None:
            x_k = fft2c(torch.complex(x_real, x_imag))
            mk_k = self.isolator.isolate_marker_kspace(x_k)
            feat_r, feat_i = self.marker_block(mk_k.real, mk_k.imag)
        else:
            feat_r, feat_i = self.marker_block(x_real, x_imag)

        # 2. Extract 1D readout profiles (magnitude, averaged over readout)
        feat_mag = torch.sqrt(feat_r**2 + feat_i**2 + 1e-8)
        readout_profile = feat_mag.mean(dim=-1)  # [B, 16, H]

        # 3. Predict trajectory deviations
        delta_k = self.trajectory_net(readout_profile)  # [B, 2, H]
        delta_k = delta_k * alpha * 0.1  # small, gated corrections

        # Expose the estimated trajectory deviation for inspection / a future
        # matched trajectory-scoring seam. NOTE this is a *Cartesian* per-readout-
        # line (kx, ky) deviation — it is NOT directly comparable to a spiral
        # acquisition trajectory (different coordinate system), so it must only be
        # graded against a Cartesian measured-trajectory reference, never a spiral
        # one (that would be a units/semantics facade — see exp_vf_35 note).
        self.last_trajectory_estimate = delta_k.detach()

        # 4. Construct per-line phase correction from trajectory deviation
        kx = torch.linspace(-0.5, 0.5, W, device=x_real.device)
        ky = torch.linspace(-0.5, 0.5, H, device=x_real.device)

        phase_correction = (
            2
            * math.pi
            * delta_k[:, 0:1, :].unsqueeze(-1)
            * kx.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        phase_correction = phase_correction + (
            2
            * math.pi
            * delta_k[:, 1:2, :].unsqueeze(-1)
            * ky.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        )  # [B, 1, H, W]

        cos_p = torch.cos(phase_correction)
        sin_p = torch.sin(phase_correction)

        corr_r = x_real * cos_p - x_imag * sin_p
        corr_i = x_real * sin_p + x_imag * cos_p

        # 5. Complex refinement + global residual
        ref_r, ref_i = self.refine(corr_r, corr_i)
        ref_r, ref_i = self.refine_head(ref_r, ref_i)
        return _from_ri(ref_r + x_real, ref_i + x_imag)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# __all__
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    "DifferentiableTrajectoryOptimizer",
    "KSpacePhaseRampConjugator",
    "MTFDeApodizationOperator",
    "MarkerGRAPPAConvolution",
    "PhaseNavigatorLock",
]
