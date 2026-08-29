"""Tests for mc_cm (WS-6).

Regression: ``_get_spectral_weights`` built its high-frequency ramp from
uncentered ``fftfreq`` grids (DC at index 0) while the spectra it multiplies
come from ``fft2c`` (DC at H/2, W/2) — the ramp was spatially inverted, so
the spectral loss penalized *low* frequencies most, the opposite of the
documented intent. The grid is now fftshifted to the centered layout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.generative.mc_cm import ManifoldConstrainedCM  # noqa: E402


def test_spectral_weights_are_centered_like_fft2c() -> None:
    H, W = 8, 12
    # The method reads nothing from self — call unbound to avoid building
    # the full consistency model.
    weight = ManifoldConstrainedCM._get_spectral_weights(None, H, W, torch.device("cpu"))
    assert weight.shape == (1, 1, H, W)

    w = weight[0, 0]
    center = w[H // 2, W // 2]
    # DC (weight minimum, exactly 1.0) must sit at the CENTER pixel, matching
    # the fft2c layout the weight is applied against.
    assert torch.isclose(center, torch.tensor(1.0))
    assert (w >= center).all()
    # The corners carry the highest frequencies -> maximum weight (2.0).
    assert torch.isclose(w[0, 0], torch.tensor(2.0))
    assert w.argmin() == (H // 2) * W + (W // 2)
