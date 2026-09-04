r"""Complex k-Space Forward Operator for Cold Diffusion.

Implements the full MRI signal equation as a differentiable forward
degradation operator for cold diffusion training:

.. math::

    k_t = P_t \odot \mathcal{F}(C \odot x_0 \cdot e^{-i \Delta\omega(r) t_e})
          + \eta_t

Where:
    - :math:`x_0` is the ground-truth complex image
    - :math:`C` are coil sensitivity maps (multi-coil, optional)
    - :math:`\Delta\omega(r)` is the B0 off-resonance map (optional)
    - :math:`t_e` is the echo time
    - :math:`P_t` is the progressive sub-Nyquist sampling mask at timestep *t*
    - :math:`\eta_t \sim \mathcal{CN}(0, \sigma_t^2 I)` is complex Gaussian noise

Reference:
    Cole et al., "Analysis of Deep Complex-Valued CNNs for MRI
    Reconstruction", *Magnetic Resonance in Medicine* (2021).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .fft_ops import _to_complex, fft2c

# ---------------------------------------------------------------------------
# Sub-operators
# ---------------------------------------------------------------------------


def apply_coil_sensitivity(
    image: Tensor,
    sensitivity_maps: Tensor | None,
) -> Tensor:
    """Multiply image by coil sensitivity maps.

    Args:
        image: Complex image ``(B, 1, H, W)`` or ``(B, C, H, W)``.
        sensitivity_maps: Complex coil maps ``(B, N_coils, H, W)`` or *None*.

    Returns:
        Multi-coil image ``(B, N_coils, H, W)`` if maps given, else unchanged.
    """
    if sensitivity_maps is None:
        return image

    img_c = _to_complex(image)
    smaps_c = _to_complex(sensitivity_maps)

    # Broadcast single-channel image across coils
    if img_c.shape[1] == 1:
        return img_c * smaps_c  # (B, N_coils, H, W)

    return img_c * smaps_c


def apply_b0_phase(
    image: Tensor,
    b0_map: Tensor | None,
    echo_time: float = 0.0,
) -> Tensor:
    r"""Apply off-resonance phase accrual :math:`e^{-i \Delta\omega t_e}`.

    Args:
        image: Complex image ``(..., H, W)``.
        b0_map: Off-resonance map ``(B, 1, H, W)`` in rad/s, or *None*.
        echo_time: Echo time in seconds.

    Returns:
        Phase-modulated complex image, same shape as input.
    """
    if b0_map is None or echo_time == 0.0:
        return image

    # Phase = -Δω · TE  (negative sign: signal equation convention)
    phase = -b0_map * echo_time  # (B, 1, H, W) real
    phase_factor = torch.polar(
        torch.ones_like(phase),
        phase,
    )  # complex unit phasor

    return _to_complex(image) * phase_factor


def apply_sampling_mask(
    kspace: Tensor,
    mask: Tensor | None,
    density: float = 1.0,
) -> Tensor:
    """Sub-Nyquist sampling mask with optional density control.

    Args:
        kspace: Complex k-space ``(B, C, H, W)``.
        mask: Binary sampling mask broadcastable to *kspace*, or *None*.
        density: Fraction of *mask*'s 1-entries to retain (1.0 = keep all).

    Returns:
        Undersampled k-space, same dtype.
    """
    if mask is None and density >= 1.0:
        return kspace

    if mask is not None:
        m = mask.real if torch.is_complex(mask) else mask
    else:
        m = torch.ones(
            kspace.shape[0],
            1,
            kspace.shape[-2],
            kspace.shape[-1],
            device=kspace.device,
            dtype=kspace.real.dtype,
        )

    # Sub-sample: randomly drop lines from the mask to reach target density
    if 0.0 < density < 1.0:
        keep = torch.rand_like(m) < density
        m = m * keep.float()

    return kspace * m


def add_complex_noise(
    kspace: Tensor,
    sigma: float = 0.0,
) -> Tensor:
    r"""Add zero-mean circularly-symmetric complex Gaussian noise.

    :math:`\eta \sim \mathcal{CN}(0, \sigma^2 I)`.

    Args:
        kspace: Complex k-space tensor.
        sigma: Noise standard deviation (per component).

    Returns:
        Noisy k-space (complex).
    """
    if sigma <= 0.0:
        return kspace

    k = _to_complex(kspace)
    noise = torch.complex(
        torch.randn_like(k.real) * sigma,
        torch.randn_like(k.imag) * sigma,
    )
    return k + noise


# ---------------------------------------------------------------------------
# Composite forward operator
# ---------------------------------------------------------------------------


class ComplexForwardOperator:
    r"""Differentiable MRI forward degradation operator.

    Composes coil sensitivity, B0 phase, FFT, sampling, and noise
    as prescribed by the full MRI signal equation.

    Example::

        op = ComplexForwardOperator(num_timesteps=100)
        k_degraded = op(x_clean, timestep=50, mask=sampling_mask)
        x_degraded = ifft2c(k_degraded)
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{ComplexForward}(x) = \mathcal{F}(x \odot S) \odot M"""

    def __init__(
        self,
        num_timesteps: int = 100,
        max_sigma: float = 0.05,
        min_density: float = 0.05,
        density_schedule: str = "linear",
    ) -> None:
        """Initialise the forward operator.

        Args:
            num_timesteps: Total diffusion timesteps *T*.
            max_sigma: Noise σ at ``t = T`` (linearly ramped from 0).
            min_density: Minimum mask density at ``t = T``.
            density_schedule: ``'linear'`` or ``'cosine'`` density ramp.
        """
        self.num_timesteps = num_timesteps
        self.max_sigma = max_sigma
        self.min_density = min_density
        self.density_schedule = density_schedule

    # -- schedule helpers ---------------------------------------------------

    def _density_at(self, t: int) -> float:
        """Return mask density at timestep *t* (t=0 → 1.0, t=T → min_density)."""
        frac = t / max(self.num_timesteps, 1)
        if self.density_schedule == "cosine":
            # Cosine decay from 1.0 → min_density
            return self.min_density + (1.0 - self.min_density) * (
                0.5 * (1.0 + math.cos(math.pi * frac))
            )
        # Linear decay (default)
        return 1.0 - frac * (1.0 - self.min_density)

    def _sigma_at(self, t: int) -> float:
        """Return noise σ at timestep *t* (t=0 → 0, t=T → max_sigma)."""
        return self.max_sigma * (t / max(self.num_timesteps, 1))

    # -- main entry ---------------------------------------------------------

    def __call__(
        self,
        x_clean: Tensor,
        timestep: int,
        *,
        mask: Tensor | None = None,
        sensitivity_maps: Tensor | None = None,
        b0_map: Tensor | None = None,
        echo_time: float = 0.0,
    ) -> Tensor:
        """Apply the full MRI forward degradation at timestep *t*.

        Steps (in order):
            1. Coil sensitivity modulation  ``C · x``
            2. B0 off-resonance phase       ``x · exp(-iΔω·TE)``
            3. Fourier transform             ``F(·)``
            4. Sub-Nyquist sampling          ``P_t · k``
            5. Complex Gaussian noise        ``k + η_t``

        Args:
            x_clean: Ground-truth image ``(B, C, H, W)`` (complex or 2ch real).
            timestep: Current diffusion timestep (0 = clean, T = maximally
                degraded).
            mask: Binary sampling mask broadcastable to k-space.
            sensitivity_maps: Complex coil maps ``(B, N_coils, H, W)`` or *None*.
            b0_map: Off-resonance map ``(B, 1, H, W)`` in rad/s or *None*.
            echo_time: Echo time in seconds.

        Returns:
            Degraded complex k-space ``(B, C, H, W)``.
        """
        x = _to_complex(x_clean)

        # 1. Coil sensitivity
        x = apply_coil_sensitivity(x, sensitivity_maps)

        # 2. B0 phase
        x = apply_b0_phase(x, b0_map, echo_time)

        # 3. FFT → k-space
        k = fft2c(x)

        # 4. Sampling mask with density schedule
        density = self._density_at(timestep)
        k = apply_sampling_mask(k, mask, density)

        # 5. Complex Gaussian noise
        sigma = self._sigma_at(timestep)
        k = add_complex_noise(k, sigma)

        return k


__all__ = [
    "ComplexForwardOperator",
    "add_complex_noise",
    "apply_b0_phase",
    "apply_coil_sensitivity",
    "apply_sampling_mask",
]
