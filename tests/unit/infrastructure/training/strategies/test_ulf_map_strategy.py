"""Tests for UlfMapStrategy: Hoult-weighted physics-noise MAP (B-2.1)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.ulf_map_strategy import (
    _fft2c_dc_index,
    _lowpass_otf,
    ulf_degrade,
    ulf_map_solve,
    ulf_sigma,
)


@pytest.mark.parametrize("n", [8, 16, 32, 256])
def test_lowpass_otf_peaks_on_fft2c_dc_and_is_real_for_even(n: int) -> None:
    # Regression for the off-center OTF (#10): the Gaussian peak MUST sit on the
    # fft2c DC bin (was index h//2-1), and for even sizes filtering a real image
    # must stay real (the .real cast was silently discarding up to ~47% imag energy).
    import torch as _t

    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    dc = _fft2c_dc_index(n, n, _t.device("cpu"))
    otf = _lowpass_otf(n, n, 0.3, _t.device("cpu"), _t.float32)[0, 0]
    assert divmod(int(otf.argmax()), n) == dc  # peak on the true DC
    x = _t.zeros(1, 1, n, n)
    x[0, 0, n // 2, n // 2] = 1.0
    filt = ifft2c(otf.view(1, 1, n, n) * fft2c(x))
    assert float(filt.imag.norm() / filt.real.norm().clamp_min(1e-12)) < 1e-5


@pytest.mark.parametrize("n", [7, 9])
def test_lowpass_otf_centered_on_odd_sizes(n: int) -> None:
    # Odd-axis fft2c uses a roll that h//2 misses; the empirical DC location must
    # still place the peak on DC (residual imag is tiny vs the ~0.9 pre-fix bug).
    import torch as _t

    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    dc = _fft2c_dc_index(n, n, _t.device("cpu"))
    otf = _lowpass_otf(n, n, 0.3, _t.device("cpu"), _t.float32)[0, 0]
    assert divmod(int(otf.argmax()), n) == dc
    x = _t.zeros(1, 1, n, n)
    x[0, 0, n // 2, n // 2] = 1.0
    filt = ifft2c(otf.view(1, 1, n, n) * fft2c(x))
    assert float(filt.imag.norm() / filt.real.norm().clamp_min(1e-12)) < 0.05


def test_ulf_sigma_increases_as_field_drops() -> None:
    # Hoult: lower field -> noisier -> larger sigma.
    s_ulf = ulf_sigma(0.1, sigma_floor=0.05)
    s_mid = ulf_sigma(1.5, sigma_floor=0.05)
    s_hf = ulf_sigma(3.0, sigma_floor=0.05)
    assert s_ulf > s_mid > s_hf
    # At the 3T reference the Hoult scale is 1.0 -> sigma == floor.
    assert abs(s_hf - 0.05) < 1e-9


def test_forward_operator_is_a_nontrivial_lowpass() -> None:
    # A is a genuine field-dependent low-pass (NOT identity): it blurs and reduces
    # high-frequency energy. (Fixes the review's A=identity facade finding.)
    from spectramr.infrastructure.physics.fft_ops import fft2c

    x = torch.rand(1, 1, 16, 16)
    ax = ulf_degrade(x, blur_cutoff=0.3)
    assert not torch.allclose(ax, x, atol=1e-3)  # not identity
    # high-frequency (off-centre) spectral energy is attenuated by A.
    hf = lambda im: fft2c(im).abs()[..., :4, :4].sum()  # noqa: E731 - corner band
    assert float(hf(ax)) < float(hf(x))


def test_high_field_data_consistency() -> None:
    # Tiny sigma (high field) -> huge data weight -> the solution is data-consistent
    # under the forward operator: A(x_hat) ~= y (the correct MAP property; the output
    # itself is A^{-1}(y), the deconvolved estimate, not y).
    y = torch.rand(1, 1, 8, 8)
    out = ulf_map_solve(
        y, lambda x: x, sigma=1e-3, rho=1.0, iterations=60,
        blur_cutoff=2.0, data_weight_max=1e6,
    )
    assert torch.allclose(ulf_degrade(out, 2.0), y, atol=2e-2)


def test_data_term_constrains_passband() -> None:
    # With a real low-pass A + high data weight + identity prior, the passband (DC)
    # of the output tracks the measurement (H=1 at DC -> exact data consistency).
    from spectramr.infrastructure.physics.fft_ops import fft2c

    y = torch.rand(1, 1, 16, 16)
    out = ulf_map_solve(
        y, lambda x: x, sigma=1e-2, rho=1.0, iterations=60,
        blur_cutoff=0.5, data_weight_max=1e6,
    )
    dc_y = fft2c(y).abs()[..., 8, 8]
    dc_o = fft2c(out).abs()[..., 8, 8]
    assert (dc_o - dc_y).abs().item() < 0.05 * float(dc_y.abs().item() + 1e-6)


def test_ulf_prior_dominates_pulls_toward_denoiser() -> None:
    # Large sigma (ULF) -> tiny data weight -> output is pulled toward the denoiser
    # fixed point (a constant blob here), not the measurement.
    y = torch.rand(1, 1, 8, 8)
    blob = torch.full((1, 1, 8, 8), 0.5)
    out = ulf_map_solve(
        y, lambda x: blob, sigma=50.0, rho=1.0, iterations=50, data_weight_max=1e3
    )
    assert (out - blob).abs().mean() < (y - blob).abs().mean()


def test_output_depends_on_measurement() -> None:
    # Anti-degeneracy (pitfall #20): the output varies with the measurement, even
    # at the ULF operating point (small data weight) — the blur A still couples y.
    den = lambda x: 0.9 * x  # noqa: E731 - simple contractive denoiser
    y1 = torch.zeros(1, 1, 8, 8)
    y2 = torch.ones(1, 1, 8, 8)
    o1 = ulf_map_solve(y1, den, sigma=ulf_sigma(0.1, 0.05), rho=1.0, iterations=10)
    o2 = ulf_map_solve(y2, den, sigma=ulf_sigma(0.1, 0.05), rho=1.0, iterations=10)
    assert not torch.allclose(o1, o2)


def test_field_knob_changes_solution() -> None:
    # The physics knob (source_field via sigma) is live: different fields give
    # different reconstructions for the same measurement + denoiser.
    den = lambda x: 0.5 * x  # noqa: E731
    y = torch.rand(1, 1, 8, 8)
    o_ulf = ulf_map_solve(y, den, sigma=ulf_sigma(0.1, 0.05), rho=1.0, iterations=15)
    o_hf = ulf_map_solve(y, den, sigma=ulf_sigma(3.0, 0.05), rho=1.0, iterations=15)
    assert not torch.allclose(o_ulf, o_hf)


def test_strategy_registered_in_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "ulf_map" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def test_validation_forward_runs_map() -> None:
    # _validation_forward must run the MAP solve (not a single forward). Build a
    # bare instance with a stub denoiser and the ulf_map config knobs.
    import types

    from spectramr.infrastructure.training.strategies.ulf_map_strategy import UlfMapStrategy

    strat = object.__new__(UlfMapStrategy)
    strat._denoiser = lambda x: x  # identity prox
    strat._pnp_cfg = types.SimpleNamespace(enabled=True, rho=1.0, iterations=10)
    strat._ulf_cfg = types.SimpleNamespace(
        source_field=3.0, sigma_floor=0.05, data_weight_max=1e6, blur_cutoff=10.0
    )
    y = torch.rand(1, 1, 8, 8)
    out = strat._validation_forward(y, {"use_dc": False})
    # high field + identity denoiser + huge weight -> data-consistent: A(out) ~= y
    assert torch.allclose(ulf_degrade(out, 10.0), y, atol=2e-2)


def test_config_mounted_and_frozen() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.config.schemas.training.ulf_map import UlfMapConfig

    assert "ulf_map" in TrainingStrategyConfigSchema.model_fields
    c = UlfMapConfig()
    assert c.source_field > 0.0 and c.sigma_floor > 0.0
    assert c.blur_cutoff > 0.0
    with pytest.raises(Exception):
        c.source_field = 1.0
