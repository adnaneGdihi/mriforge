"""MRS signal model — analytic physics, verifiable with zero real data.

The whole point of an analytic forward model is that its claims are checkable in
closed form. Every test here compares against a closed form or a known identity,
never against the function itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.config.schemas.enums import Regime
from mriforge.infrastructure.physics.signal_models.registry import get_signal_model
from mriforge.infrastructure.physics.signal_models.spectroscopy import (
    GAMMA_H_MHZ_PER_T,
    MAGNITUDE_FWHM_FACTOR,
    fid_to_spectrum,
    lorentzian_fid,
    ppm_to_hz,
    spectral_frequency_axis,
)

_DWELL = 5e-4  # 2 kHz bandwidth


def _axis(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float64) * _DWELL


def _single(lw: float = 4.0, f: float = 0.0, n: int = 65536) -> tuple:
    t = _axis(n)
    fid = lorentzian_fid(
        t,
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([f], dtype=torch.float64),
        torch.tensor([lw], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    )
    return fid, spectral_frequency_axis(n, _DWELL).double()


def _fwhm(spectrum: torch.Tensor, freqs: torch.Tensor) -> float:
    """FWHM by half-max bin crossing.

    Counts bins at or above half maximum, so it UNDER-reads by up to one bin at
    each edge — the true crossings lie between samples. At 65536 points and a
    2 kHz bandwidth the bin is 0.031 Hz, which bounds the error at ~0.06 Hz;
    that is why the tolerances below are ~3% rather than exact.
    """
    idx = (spectrum >= spectrum.max() / 2).nonzero().flatten()
    return float(freqs[idx[-1]] - freqs[idx[0]])


class TestLorentzianFID:
    def test_fid_at_time_zero_is_the_phased_amplitude_sum(self) -> None:
        """s(0) = sum_m a_m * exp(i*phi_m): no decay, no oscillation yet."""
        t = _axis(64)
        a = torch.tensor([1.0, 0.5], dtype=torch.float64)
        ph = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
        fid = lorentzian_fid(
            t,
            a,
            torch.tensor([10.0, -20.0]).double(),
            torch.tensor([3.0, 4.0]).double(),
            ph,
        )
        expected = complex(1.0, 0.0) + complex(0.0, 0.5)
        assert complex(fid[0]) == pytest.approx(expected, abs=1e-12)

    def test_resonances_superpose_exactly(self) -> None:
        """The model is a SUM over resonances — linear in the amplitudes."""
        t = _axis(256)
        a = torch.tensor([1.0, 0.5]).double()
        f = torch.tensor([100.0, -200.0]).double()
        lw = torch.tensor([4.0, 6.0]).double()
        ph = torch.tensor([0.0, 1.0]).double()
        both = lorentzian_fid(t, a, f, lw, ph)
        separate = lorentzian_fid(t, a[:1], f[:1], lw[:1], ph[:1]) + lorentzian_fid(
            t, a[1:], f[1:], lw[1:], ph[1:]
        )
        assert float((both - separate).abs().max()) == pytest.approx(0.0, abs=1e-15)

    def test_the_peak_lands_at_the_declared_frequency(self) -> None:
        fid, freqs = _single(lw=4.0, f=100.0)
        peak = int(fid_to_spectrum(fid).abs().argmax())
        assert float(freqs[peak]) == pytest.approx(100.0, abs=0.1)

    @pytest.mark.parametrize("lw_true", [2.0, 4.0, 10.0])
    def test_absorption_mode_fwhm_equals_the_declared_linewidth(
        self, lw_true: float
    ) -> None:
        """The claim `dnu` is written to make true — for the REAL part.

        `exp(-pi*dnu*t)` FTs to `1/(pi*dnu + i*2*pi*f)`, whose real part is a true
        Lorentzian with FWHM = dnu.
        """
        fid, freqs = _single(lw=lw_true)
        assert _fwhm(fid_to_spectrum(fid).real, freqs) == pytest.approx(
            lw_true, rel=0.03
        )

    @pytest.mark.parametrize("lw_true", [2.0, 4.0, 10.0])
    def test_magnitude_mode_fwhm_is_sqrt3_times_broader(self, lw_true: float) -> None:
        """The 73% error waiting to be made — see MAGNITUDE_FWHM_FACTOR.

        A magnitude spectrum is the SQUARE ROOT of a Lorentzian, not a Lorentzian,
        so its half-max sits further out. This matters because `spectral_linewidth`
        — the metric backing this regime — documents its input as a MAGNITUDE
        spectrum, and would therefore report sqrt(3)*dnu for a model fitted with
        linewidth dnu. Grading one against the other directly is pitfall #18.
        """
        fid, freqs = _single(lw=lw_true)
        measured = _fwhm(fid_to_spectrum(fid).abs(), freqs)
        assert measured == pytest.approx(MAGNITUDE_FWHM_FACTOR * lw_true, rel=0.03)
        assert pytest.approx(math.sqrt(3.0)) == MAGNITUDE_FWHM_FACTOR

    def test_absorption_lineshape_matches_the_closed_form(self) -> None:
        lw = 4.0
        fid, freqs = _single(lw=lw)
        spec = fid_to_spectrum(fid).real
        a = math.pi * lw
        closed = a / (a**2 + (2 * math.pi * freqs) ** 2)
        sel = freqs.abs() < 40
        deviation = (spec / spec.max() - closed / closed.max())[sel].abs().max()
        assert float(deviation) < 1e-2

    def test_magnitude_lineshape_matches_the_closed_form(self) -> None:
        lw = 4.0
        fid, freqs = _single(lw=lw)
        spec = fid_to_spectrum(fid).abs()
        a = math.pi * lw
        closed = 1.0 / torch.sqrt(torch.as_tensor(a**2) + (2 * math.pi * freqs) ** 2)
        sel = freqs.abs() < 40
        deviation = (spec / spec.max() - closed / closed.max())[sel].abs().max()
        assert float(deviation) < 1e-3

    def test_a_negative_linewidth_is_clamped_not_allowed_to_explode(self) -> None:
        """exp(-pi*lw*t) with lw < 0 is a GROWING exponential -> inf.

        An untrained generator emits negatives from step 0, which is why the
        strategy softplusses this parameter. The clamp here is the last line of
        defence, so a direct caller cannot produce inf either.
        """
        t = _axis(512)
        fid = lorentzian_fid(
            t,
            torch.tensor([1.0]).double(),
            torch.tensor([0.0]).double(),
            torch.tensor([-5.0]).double(),  # negative!
            torch.tensor([0.0]).double(),
        )
        assert torch.isfinite(fid.abs()).all()

    def test_mismatched_parameter_shapes_raise(self) -> None:
        """Broadcasting them would pair the wrong amplitude with the wrong line."""
        t = _axis(64)
        with pytest.raises(ValueError, match="share a shape"):
            lorentzian_fid(
                t,
                torch.tensor([1.0, 0.5]).double(),
                torch.tensor([10.0]).double(),  # 1 frequency for 2 amplitudes
                torch.tensor([3.0, 4.0]).double(),
                torch.tensor([0.0, 0.0]).double(),
            )

    def test_a_2d_time_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            lorentzian_fid(
                torch.zeros(2, 8).double(),
                torch.tensor([1.0]).double(),
                torch.tensor([0.0]).double(),
                torch.tensor([3.0]).double(),
                torch.tensor([0.0]).double(),
            )

    def test_it_broadcasts_over_leading_voxel_dims(self) -> None:
        """[B, H, W, M] params -> [B, H, W, T]: the layout the strategy feeds."""
        t = _axis(32)
        shape = (2, 3, 3, 4)  # B, H, W, M
        fid = lorentzian_fid(
            t,
            torch.rand(shape).double(),
            torch.rand(shape).double() * 100,
            torch.rand(shape).double() + 1.0,
            torch.rand(shape).double(),
        )
        assert fid.shape == (2, 3, 3, 32)


class TestSpectrumTransform:
    def test_only_the_output_is_shifted(self) -> None:
        """An FID's origin IS index 0 — ifftshifting the input would be the bug.

        `fft2c` ifftshifts because an IMAGE's origin sits at N//2. A causal FID's
        does not. This pins the difference: a constant (DC-only) FID must produce
        all its energy at 0 Hz, which lands at the centre after the output shift.
        """
        n = 64
        dc = torch.ones(n, dtype=torch.complex128)
        spec = fid_to_spectrum(dc).abs()
        freqs = spectral_frequency_axis(n, _DWELL)
        assert float(freqs[int(spec.argmax())]) == pytest.approx(0.0, abs=1e-9)

    def test_the_frequency_axis_matches_the_spectrum_ordering(self) -> None:
        n = 1024
        freqs = spectral_frequency_axis(n, _DWELL)
        assert freqs.shape == (n,)
        assert torch.all(freqs.diff() > 0), "must be monotonically increasing"
        # Bandwidth is 1/dwell, centred on 0.
        assert float(freqs.min()) == pytest.approx(-0.5 / _DWELL, rel=1e-3)

    def test_a_real_input_is_accepted_as_complex(self) -> None:
        assert torch.is_complex(fid_to_spectrum(torch.randn(32)))

    @pytest.mark.parametrize("bad", [(0, 5e-4), (16, 0.0), (16, -1.0)])
    def test_invalid_axis_arguments_raise(self, bad: tuple) -> None:
        with pytest.raises(ValueError):
            spectral_frequency_axis(bad[0], bad[1])


class TestPPM:
    def test_naa_at_3t(self) -> None:
        """NAA sits at 2.01 ppm; at 3 T that is ~257 Hz."""
        assert float(ppm_to_hz(2.01, 3.0)) == pytest.approx(256.8, rel=1e-3)

    def test_ppm_is_field_independent_but_hz_is_not(self) -> None:
        """The whole reason ppm exists — and why field_strength_t is a real knob.

        rel=1e-5, not 1e-9: `torch.as_tensor(2.01)` on a Python float yields
        float32, whose ~1e-7 relative precision is the floor here.
        """
        assert float(ppm_to_hz(2.01, 7.0)) == pytest.approx(
            float(ppm_to_hz(2.01, 3.0)) * 7.0 / 3.0, rel=1e-5
        )

    def test_gamma_matches_the_published_value(self) -> None:
        assert pytest.approx(42.577, abs=1e-3) == GAMMA_H_MHZ_PER_T

    def test_a_non_positive_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field_strength_t"):
            ppm_to_hz(2.01, 0.0)


def test_registered_as_a_spectroscopic_signal_model() -> None:
    """It backs mri_spectroscopy's LIVE claim; a lost registration fails here."""
    spec = get_signal_model("mrs_lorentzian")
    assert spec.regime is Regime.SPECTROSCOPIC
    assert spec.parameters == (
        "amplitudes",
        "frequencies_hz",
        "linewidths_hz",
        "phases_rad",
    )
    assert spec.fn is lorentzian_fid
    assert spec.reference
