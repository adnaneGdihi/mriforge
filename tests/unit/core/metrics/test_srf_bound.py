"""Oracle tests for the synthetic-to-real fidelity bound (A-8.5 SRF-Bound)."""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.core.metrics import get_metric
from mriforge.core.metrics.registry import MetricsRegistry
from mriforge.core.metrics.srf_bound import (
    SyntheticToRealFidelityBound,
    radial_band_energies,
    w1_1d,
)

# --------------------------------------------------------------------------- #
# Oracle 1: the 1-D W1 helper matches a known closed form
# --------------------------------------------------------------------------- #


def test_w1_1d_of_translated_distributions_equals_shift() -> None:
    # W1 between X and X + delta (a pure translation) is exactly delta.
    torch.manual_seed(0)
    a = torch.randn(500)
    delta = 1.7
    assert w1_1d(a, a + delta) == pytest.approx(delta, abs=0.05)


def test_w1_1d_zero_for_identical_samples() -> None:
    a = torch.randn(200)
    assert w1_1d(a, a.clone()) == pytest.approx(0.0, abs=1e-6)


def test_w1_1d_symmetric() -> None:
    torch.manual_seed(1)
    a, b = torch.randn(300), torch.randn(300) * 2 + 1
    assert w1_1d(a, b) == pytest.approx(w1_1d(b, a), rel=1e-6)


# --------------------------------------------------------------------------- #
# Oracle 2: the metric is 0 for identical sets and rises with the gap
# --------------------------------------------------------------------------- #


def _img_set(n: int, *, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return scale * torch.rand(n, 1, 32, 32)


def test_identical_sets_score_near_zero() -> None:
    s = _img_set(8, seed=3)
    val = SyntheticToRealFidelityBound()(synthetic=s, real=s.clone())
    assert val == pytest.approx(0.0, abs=1e-4)


def test_rises_with_distribution_gap() -> None:
    # Two synthetic sets: one drawn like 'real', one with very different texture
    # (a low-pass-smoothed set has a different radial spectrum -> larger SRF-Bound).
    torch.manual_seed(0)
    real = torch.rand(10, 1, 32, 32)
    near = torch.rand(10, 1, 32, 32)  # same white-noise texture as real
    # 'far' = strongly smoothed (kills high-frequency bands) -> different spectrum.
    far = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(torch.rand(10, 1, 32, 32), (3, 3, 3, 3), mode="reflect"),
        kernel_size=7, stride=1,
    )
    m = SyntheticToRealFidelityBound()
    assert m(synthetic=far, real=real) > m(synthetic=near, real=real)


def test_monotonic_in_energy_shift() -> None:
    real = _img_set(10, scale=1.0, seed=5)
    m = SyntheticToRealFidelityBound()
    small = m(synthetic=_img_set(10, scale=1.2, seed=7), real=real)
    large = m(synthetic=_img_set(10, scale=3.0, seed=7), real=real)
    assert large > small > 0.0


def test_symmetric_metric() -> None:
    a, b = _img_set(8, scale=1.0, seed=1), _img_set(8, scale=2.0, seed=2)
    m = SyntheticToRealFidelityBound()
    assert m(synthetic=a, real=b) == pytest.approx(m(synthetic=b, real=a), rel=1e-6)


# --------------------------------------------------------------------------- #
# Feature extractor + contract
# --------------------------------------------------------------------------- #


def test_radial_band_energies_shape_and_finite() -> None:
    feats = radial_band_energies(torch.rand(5, 1, 24, 24), n_bands=6)
    assert feats.shape == (5, 6)
    assert torch.isfinite(feats).all()


def test_highest_frequency_energy_captured_in_last_band() -> None:
    # Regression (A-8.5 review HIGH): a Nyquist checkerboard puts essentially ALL
    # energy at the maximum radial frequency. The last band must capture it (the
    # old strict `<` edge dropped the max-frequency pixel and the band read ~0).
    h = w = 32
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    # Zero-mean Nyquist pattern (+/-1) -> NO DC; all energy at the max radial freq.
    checker = (((yy + xx) % 2) * 2 - 1).float()[None, None]  # [1,1,H,W]
    feats = radial_band_energies(checker, n_bands=8)
    assert float(feats[0, -1]) > 0.0  # highest-frequency band is non-empty
    assert feats[0].argmax().item() == 7  # energy concentrated in the last band


def test_accepts_unequal_set_sizes() -> None:
    # Different N and M -> the quantile-grid W1 handles it without error.
    val = SyntheticToRealFidelityBound()(synthetic=_img_set(5, seed=1), real=_img_set(12, seed=2))
    assert math.isfinite(val) and val >= 0.0


def test_accepts_list_of_images() -> None:
    imgs_s = [torch.rand(1, 1, 32, 32) for _ in range(4)]
    imgs_r = [torch.rand(1, 1, 32, 32) for _ in range(4)]
    assert math.isfinite(SyntheticToRealFidelityBound()(synthetic=imgs_s, real=imgs_r))


def test_no_real_reference_returns_nan() -> None:
    # No real set -> cannot measure a synthetic-to-real gap -> nan, never a fake 0.
    assert math.isnan(SyntheticToRealFidelityBound()(synthetic=_img_set(4)))
    assert math.isnan(SyntheticToRealFidelityBound()(_img_set(4)))  # prediction only


def test_prediction_is_synthetic_with_real_reference_kwarg() -> None:
    pred = _img_set(6, seed=1)
    val = SyntheticToRealFidelityBound()(pred, real_reference=_img_set(6, seed=2))
    assert math.isfinite(val) and val >= 0.0


def test_registered_lower_is_better() -> None:
    for key in ("srf_bound", "sim2real_w1", "synthetic_real_fidelity_bound"):
        assert MetricsRegistry.is_registered(key)
    assert get_metric("srf_bound").higher_is_better is False
