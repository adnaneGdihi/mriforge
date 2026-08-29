r"""Respiratory-gating adapter (idea 9 §9.2).

Extracts a continuous-phase respiratory signal from a 1-D respiratory-
belt trace via Hilbert transform + band-pass filtering. The output
phase :math:`\phi(t) \in [-\pi, \pi]` lets the SCAS / multi-sensor
strategies bin acquisitions by respiratory state without re-running
self-gating on the k-space data.

Mathematics
-----------
Given the analytic signal
:math:`z(t) = x(t) + i\,\mathcal H[x](t)` where :math:`\mathcal H`
is the Hilbert transform, the instantaneous phase is
:math:`\phi(t) = \arg z(t)`.

Implemented via FFT-domain analytic-signal construction (Marple 1999)
which avoids any external SciPy dependency. Band-pass pre-filtering
(0.15–0.5 Hz) suppresses heart-rate harmonics and DC drift before
the phase extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RespiratoryGate:
    """Output of :func:`extract_respiratory_phase`.

    Attributes:
        phase: ``[T]`` instantaneous respiratory phase in ``[-π, π]``.
        amplitude: ``[T]`` instantaneous amplitude (envelope of the
            analytic signal).
        sample_rate_hz: forwarded from the input.
    """

    phase: torch.Tensor
    amplitude: torch.Tensor
    sample_rate_hz: float


def _bandpass(
    signal: torch.Tensor,
    sample_rate: float,
    f_lo: float,
    f_hi: float,
) -> torch.Tensor:
    """FFT-domain ideal band-pass filter."""
    T = signal.shape[0]
    freqs = torch.fft.fftfreq(T, d=1.0 / sample_rate, device=signal.device)
    spectrum = torch.fft.fft(signal)
    mask = (freqs.abs() >= f_lo) & (freqs.abs() <= f_hi)
    return torch.fft.ifft(spectrum * mask.to(spectrum.dtype)).real


def _analytic_signal(signal: torch.Tensor) -> torch.Tensor:
    """Hilbert-transform-based analytic signal via FFT (Marple 1999)."""
    T = signal.shape[0]
    spectrum = torch.fft.fft(signal)
    h = torch.zeros(T, device=signal.device, dtype=spectrum.dtype)
    if T % 2 == 0:
        h[0] = 1.0
        h[T // 2] = 1.0
        h[1 : T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (T + 1) // 2] = 2.0
    return torch.fft.ifft(spectrum * h)


def extract_respiratory_phase(
    belt_signal: torch.Tensor,
    sample_rate_hz: float,
    band: tuple[float, float] = (0.15, 0.5),
) -> RespiratoryGate:
    """Extract the instantaneous respiratory phase.

    Args:
        belt_signal: 1-D real respiratory-belt trace.
        sample_rate_hz: Sample rate in Hertz (must be > 0).
        band: ``(f_lo, f_hi)`` band-pass cut-offs. Defaults to
            ``(0.15, 0.5)`` Hz — matches the
            :class:`DCNavigatorExtractor` defaults.

    Returns:
        :class:`RespiratoryGate` with phase + amplitude time series.

    Raises:
        ValueError: invalid sample rate / band / shape.
    """
    if belt_signal.dim() != 1:
        raise ValueError(f"belt_signal must be 1-D; got {tuple(belt_signal.shape)}")
    if belt_signal.numel() < 2:
        raise ValueError("belt_signal too short")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}")
    f_lo, f_hi = band
    if not (0 <= f_lo < f_hi):
        raise ValueError(f"band must satisfy 0 ≤ f_lo < f_hi; got {band}")

    signal = belt_signal.to(torch.float32)
    filtered = _bandpass(signal, sample_rate_hz, f_lo, f_hi)
    analytic = _analytic_signal(filtered)
    phase = torch.angle(analytic)
    amplitude = analytic.abs()
    return RespiratoryGate(
        phase=phase,
        amplitude=amplitude,
        sample_rate_hz=float(sample_rate_hz),
    )


__all__ = ["RespiratoryGate", "extract_respiratory_phase"]
