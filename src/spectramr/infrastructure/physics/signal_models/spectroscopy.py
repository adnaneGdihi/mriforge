r"""MR spectroscopy signal models (regime: ``mri_spectroscopy``).

An MRS acquisition measures a **free induction decay** — a complex time signal
that is a sum of damped, phased oscillations, one per resonance:

.. math::

    s(t) = \sum_m a_m \, e^{i(2\pi f_m t + \phi_m)} \, e^{-\pi \Delta\nu_m t}

Each resonance contributes a Lorentzian line at frequency ``f_m`` whose FWHM is
``Δν_m``, with area ``a_m``. This is the model AMARES and QUEST fit in the time
domain (Vanhamme et al. 1997), and it is the forward physics of the regime:
nonlinear in ``f``, ``Δν`` and ``φ``, with no adjoint — hence a **signal model**,
not an ``OperatorRegistry`` operator.

Fitting it is what MRS quantification *is*: the metabolite concentrations are the
amplitudes ``a_m``. That makes the objective self-supervised — the measured FID
supplies its own supervision — exactly as ``tofts_residual`` does for perfusion,
and for the same reason: real acquisitions have no ground-truth concentrations.

**On the Fourier transform here, and why it is NOT ``fft2c``.** ``fft2c`` does
``fftshift(fft2(ifftshift(x)))`` because an *image*'s origin sits at index
``N // 2``. An FID's origin is genuinely ``t = 0``, i.e. index 0 — the signal is
causal. ``ifftshift``-ing it would move a sample that is already at the origin,
which is the bug rather than the fix. So :func:`fid_to_spectrum` shifts only the
OUTPUT, to put 0 Hz at the centre of the spectrum, and keeps ``norm="ortho"`` and
the FP32 autocast guard that make ``fft_ops`` correct. This is a different
transform with a different origin convention, not an exemption from CLAUDE.md #2 —
which governs 2-D complex k-space round trips.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.config.schemas.enums import Regime

from .registry import register_signal_model

#: Gyromagnetic ratio of 1H in MHz/T — converts ppm to Hz at a field strength.
GAMMA_H_MHZ_PER_T: float = 42.577478518

#: A MAGNITUDE spectrum's peaks are sqrt(3) times broader than the same peaks in
#: absorption mode. This is not an approximation and not a bug — it is what a
#: one-sided decay does, and it is the single easiest way to misread an MRS
#: linewidth by 73%.
#:
#: The FT of ``exp(-pi*dnu*t)`` for ``t >= 0`` is ``1 / (pi*dnu + i*2*pi*f)``.
#: Its REAL part is a true Lorentzian, ``a / (a^2 + (2*pi*f)^2)`` with
#: ``a = pi*dnu``, and half-maximum lands at ``2*pi*f = a``, i.e. FWHM = ``dnu``.
#: Its MAGNITUDE is ``1 / sqrt(a^2 + (2*pi*f)^2)`` — the square root of a
#: Lorentzian, not a Lorentzian — and half-maximum needs
#: ``a^2 + (2*pi*f)^2 = 4*a^2``, i.e. ``2*pi*f = sqrt(3)*a`` and FWHM =
#: ``sqrt(3)*dnu``.
#:
#: Both are verified against their closed forms in
#: ``tests/unit/infrastructure/physics/signal_models/test_spectroscopy.py``
#: (absorption 3.13e-03, magnitude 3.29e-05 max deviation).
#:
#: This matters HERE because ``spectral_linewidth`` — the metric backing this
#: regime — documents its input as a **magnitude** spectrum, so it reports
#: ``sqrt(3)*dnu`` for a model fitted with linewidth ``dnu``. Grading a fitted
#: ``dnu`` directly against that metric would be a 73% error dressed as a
#: disagreement (pitfall #18).
MAGNITUDE_FWHM_FACTOR: float = math.sqrt(3.0)


@register_signal_model(
    name="mrs_lorentzian",
    regime=Regime.SPECTROSCOPIC,
    parameters=("amplitudes", "frequencies_hz", "linewidths_hz", "phases_rad"),
    signal="free_induction_decay",
    reference="Vanhamme et al., J Magn Reson 129(1):35-43, 1997 (AMARES)",
)
def lorentzian_fid(
    t_s: Tensor,
    amplitudes: Tensor,
    frequencies_hz: Tensor,
    linewidths_hz: Tensor,
    phases_rad: Tensor,
) -> Tensor:
    r"""Sum of damped complex exponentials — the MRS forward model.

    ``s(t) = sum_m a_m exp(i(2*pi*f_m*t + phi_m)) exp(-pi*dnu_m*t)``

    The damping is written ``exp(-pi*dnu*t)`` rather than ``exp(-t/T2star)`` so
    that ``dnu`` is the **absorption-mode** Lorentzian FWHM in Hz (the two are
    related by ``dnu = 1/(pi*T2star)``). MRS reports linewidth as a FWHM, so this
    keeps the fitted parameter in the units the field actually uses.

    **Read `dnu` as the ABSORPTION-mode FWHM, never the magnitude one.** A
    magnitude spectrum's peaks are ``sqrt(3)`` times broader — see
    :data:`MAGNITUDE_FWHM_FACTOR` for the derivation and for why it matters when
    comparing against the ``spectral_linewidth`` metric, which takes a magnitude
    spectrum.

    Args:
        t_s: ``[T]`` acquisition time axis in seconds, starting at 0.
        amplitudes: ``[..., M]`` resonance areas (concentration, arbitrary units).
        frequencies_hz: ``[..., M]`` resonance offsets from the carrier, Hz.
        linewidths_hz: ``[..., M]`` Lorentzian FWHM, Hz. Must be positive; see
            below.
        phases_rad: ``[..., M]`` per-resonance phase, radians.

    Returns:
        ``[..., T]`` complex FID.

    Raises:
        ValueError: if ``t_s`` is not 1-D, or the four parameter tensors do not
            share a shape.

    Note:
        ``linewidths_hz`` is clamped positive here, and that clamp is numerical,
        not cosmetic: a negative linewidth flips ``exp(-pi*dnu*t)`` into a growing
        exponential and the FID overflows to ``inf``. An untrained generator emits
        negatives from step 0, which is why the strategy applies softplus to this
        parameter — the same trap as ``Ktrans`` in the extended-Tofts kernel.
    """
    if t_s.ndim != 1:
        raise ValueError(f"t_s must be a 1-D [T] axis; got {tuple(t_s.shape)}.")
    shapes = {
        "amplitudes": amplitudes.shape,
        "frequencies_hz": frequencies_hz.shape,
        "linewidths_hz": linewidths_hz.shape,
        "phases_rad": phases_rad.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(
            "the four resonance parameters must share a shape [..., M]; got "
            + ", ".join(f"{k}={tuple(v)}" for k, v in shapes.items())
            + ". Broadcasting them against each other would silently pair the "
            "wrong amplitude with the wrong frequency."
        )

    # [..., M, 1] against [T] broadcasts to [..., M, T]; sum over M gives [..., T].
    t = t_s.reshape(*([1] * amplitudes.ndim), -1)  # [1, ..., 1, T]
    amp = amplitudes.unsqueeze(-1)
    freq = frequencies_hz.unsqueeze(-1)
    lw = linewidths_hz.unsqueeze(-1).clamp_min(1e-6)
    phase = phases_rad.unsqueeze(-1)

    decay = torch.exp(-math.pi * lw * t)
    angle = 2.0 * math.pi * freq * t + phase
    oscillation = torch.polar(torch.ones_like(angle), angle)
    return (amp * decay * oscillation).sum(dim=-2)


def fid_to_spectrum(fid: Tensor) -> Tensor:
    """Complex FID -> complex spectrum, 0 Hz at the centre.

    Only the OUTPUT is shifted. See the module docstring: an FID's origin really
    is index 0, so ``ifftshift``-ing the input (as ``fft2c`` does for an image,
    whose origin is at ``N // 2``) would mis-centre it.

    ``norm="ortho"`` and the disabled autocast mirror ``fft_ops`` — the FP32
    context is what stops AMP producing NaN gradients through the transform.
    """
    with torch.amp.autocast("cuda", enabled=False):
        signal = fid if torch.is_complex(fid) else fid.to(torch.complex64)
        spectrum = torch.fft.fft(signal.to(torch.complex64), dim=-1, norm="ortho")
        return torch.fft.fftshift(spectrum, dim=-1)


def spectral_frequency_axis(
    num_points: int, dwell_time_s: float, *, device: torch.device | None = None
) -> Tensor:
    """``[N]`` frequency axis in Hz matching :func:`fid_to_spectrum`'s output order.

    ``fftshift``-ed to sit alongside a centred spectrum, so ``axis[i]`` is the
    frequency of ``spectrum[..., i]``.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1; got {num_points}")
    if dwell_time_s <= 0:
        raise ValueError(f"dwell_time_s must be > 0; got {dwell_time_s}")
    freqs = torch.fft.fftfreq(num_points, d=dwell_time_s, device=device)
    return torch.fft.fftshift(freqs)


def ppm_to_hz(ppm: Tensor | float, field_strength_t: float) -> Tensor:
    """Chemical shift (ppm) -> frequency offset (Hz) at ``field_strength_t``.

    ``f = ppm * 1e-6 * gamma * B0``, with gamma in Hz/T. ppm exists precisely so
    that chemical shift is field-independent, so this conversion is the only
    place the field strength enters the resonance positions.
    """
    if field_strength_t <= 0:
        raise ValueError(f"field_strength_t must be > 0; got {field_strength_t}")
    ppm_t = torch.as_tensor(ppm)
    return ppm_t * 1e-6 * (GAMMA_H_MHZ_PER_T * 1e6) * field_strength_t


__all__ = [
    "GAMMA_H_MHZ_PER_T",
    "MAGNITUDE_FWHM_FACTOR",
    "fid_to_spectrum",
    "lorentzian_fid",
    "ppm_to_hz",
    "spectral_frequency_axis",
]
