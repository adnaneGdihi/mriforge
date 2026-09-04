r"""Tests for ``extract_respiratory_phase`` (idea 9 §9.2).

Targets ``spectramr.infrastructure.sensors.adapters.respiratory``.

Categories:

- 1-D / non-positive-fs / inverted band rejected
- A pure 0.3 Hz sinusoid is recovered with a near-constant amplitude
  envelope and a phase ramp wrapping every 1/0.3 ≈ 3.3 s
- The DC component is suppressed by the band-pass
- Output phase lies in ``[-π, π]``
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.sensors.adapters.respiratory import (
    RespiratoryGate,
    extract_respiratory_phase,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_2d_signal() -> None:
    """Belt signal must be 1-D."""
    with pytest.raises(ValueError, match="1-D"):
        extract_respiratory_phase(torch.zeros(8, 8), sample_rate_hz=10.0)


def test_rejects_short_signal() -> None:
    """Single-sample input rejected."""
    with pytest.raises(ValueError, match="too short"):
        extract_respiratory_phase(torch.zeros(1), sample_rate_hz=10.0)


def test_rejects_non_positive_sample_rate() -> None:
    """Sample rate must be > 0."""
    with pytest.raises(ValueError, match="sample_rate_hz"):
        extract_respiratory_phase(torch.zeros(64), sample_rate_hz=0.0)


def test_rejects_inverted_band() -> None:
    """``f_lo ≥ f_hi`` is invalid."""
    with pytest.raises(ValueError, match="band"):
        extract_respiratory_phase(
            torch.zeros(64),
            sample_rate_hz=100.0,
            band=(0.5, 0.2),
        )


# ---------------------------------------------------------------------------
# Pure-sinusoid recovery
# ---------------------------------------------------------------------------


def test_pure_resp_tone_amplitude_near_constant() -> None:
    """A pure 0.3 Hz sinusoid produces a near-constant amplitude envelope."""
    fs = 100.0
    duration_s = 30.0
    T = int(duration_s * fs)
    t = torch.linspace(0, duration_s, T)
    signal = torch.sin(2 * math.pi * 0.3 * t)

    gate = extract_respiratory_phase(signal, sample_rate_hz=fs)
    # Trim transients at the very beginning and end of the trace.
    centre = gate.amplitude[T // 4 : 3 * T // 4]
    rel_std = (centre.std() / centre.mean()).item()
    assert rel_std < 0.10  # ≤ 10% relative variation


def test_phase_in_pi_range() -> None:
    """Output phase always lies in ``[-π, π]``."""
    torch.manual_seed(0)
    fs = 100.0
    signal = torch.randn(2048)
    gate = extract_respiratory_phase(signal, sample_rate_hz=fs)
    assert (gate.phase >= -math.pi - 1e-5).all()
    assert (gate.phase <= math.pi + 1e-5).all()


def test_dc_suppressed_by_bandpass() -> None:
    """A pure DC bias is rejected by the band-pass — amplitude ≈ 0."""
    fs = 100.0
    signal = torch.full((2048,), 5.0)  # pure DC
    gate = extract_respiratory_phase(signal, sample_rate_hz=fs)
    # Centre amplitude is essentially zero.
    centre = gate.amplitude[fs.__int__() : -int(fs)]
    assert centre.max().item() < 0.05


def test_returns_dataclass_with_all_fields() -> None:
    """The output is a :class:`RespiratoryGate` with all fields populated."""
    fs = 100.0
    gate = extract_respiratory_phase(
        torch.randn(512), sample_rate_hz=fs, band=(0.15, 0.5)
    )
    assert isinstance(gate, RespiratoryGate)
    assert gate.phase.shape == (512,)
    assert gate.amplitude.shape == (512,)
    assert gate.sample_rate_hz == fs


def test_phase_unwraps_at_expected_period() -> None:
    """Pure 0.3 Hz tone wraps the phase every ~3.33 s."""
    fs = 100.0
    duration_s = 10.0
    T = int(duration_s * fs)
    t = torch.linspace(0, duration_s, T)
    signal = torch.sin(2 * math.pi * 0.3 * t)

    gate = extract_respiratory_phase(signal, sample_rate_hz=fs)
    # The phase trace should have ~ duration / period = 10/3.33 ≈ 3 wraps.
    diffs = gate.phase[1:] - gate.phase[:-1]
    wraps = int((diffs.abs() > math.pi).sum().item())
    # 3 wraps within ±2 of the expected count due to transient region.
    assert 1 <= wraps <= 6
