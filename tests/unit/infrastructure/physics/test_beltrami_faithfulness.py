"""Oracle tests for the Beltrami faithfulness threshold (A-6.2 BFT)."""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.beltrami_faithfulness import (
    BeltramiFaithfulnessThreshold,
    beltrami_faithfulness,
    beltrami_from_displacement,
    is_bijective,
    is_faithful,
)


def _coord_grid(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Pixel coordinates centred at 0 (so derivatives are well-conditioned).
    y = torch.arange(h, dtype=torch.float32) - h / 2
    x = torch.arange(w, dtype=torch.float32) - w / 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def _interior(mu: torch.Tensor) -> torch.Tensor:
    # Drop the 1-pixel replicate-padded border where the finite difference is biased.
    return mu[..., 2:-2, 2:-2]


# --------------------------------------------------------------------------- #
# Closed-form Beltrami oracles
# --------------------------------------------------------------------------- #


def test_identity_map_has_zero_beltrami() -> None:
    d = torch.zeros(1, 2, 16, 16)  # d = 0 -> identity deformation
    mu = beltrami_from_displacement(d)
    assert float(mu.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_conformal_scaling_has_zero_beltrami() -> None:
    # d = a*z (d_x = a*x, d_y = a*y) is conformal -> mu = 0.
    xx, yy = _coord_grid(20, 20)
    a = 0.1
    d = torch.stack([a * xx, a * yy], dim=0)[None]  # [1,2,H,W]
    mu = beltrami_from_displacement(d)
    assert float(_interior(mu).abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_anticonformal_map_has_beltrami_equal_constant() -> None:
    # d = c*conj(z) (d_x = c*x, d_y = -c*y) -> mu = c (the known affine oracle).
    xx, yy = _coord_grid(20, 20)
    c = 0.3
    d = torch.stack([c * xx, -c * yy], dim=0)[None]
    mu = beltrami_from_displacement(d)
    assert float(_interior(mu).abs().mean()) == pytest.approx(c, abs=1e-4)


# --------------------------------------------------------------------------- #
# Faithfulness threshold semantics
# --------------------------------------------------------------------------- #


def test_identity_is_faithful_and_bijective() -> None:
    d = torch.zeros(1, 2, 16, 16)
    assert is_faithful(d, delta=0.1)
    assert is_bijective(d)


def test_anticonformal_faithful_iff_delta_above_distortion() -> None:
    xx, yy = _coord_grid(20, 20)
    c = 0.3
    d = torch.stack([c * xx, -c * yy], dim=0)[None]
    # interior |mu| == 0.3; the border (replicate pad) is <= 0.3, so max ~ 0.3.
    assert is_faithful(d, delta=0.5)  # 0.3 <= 0.5
    assert not is_faithful(d, delta=0.2)  # 0.3 > 0.2
    assert is_bijective(d)  # 0.3 < 1


def test_folding_map_is_not_bijective() -> None:
    # A strong anticonformal warp with |mu| > 1 folds the coordinate map.
    xx, yy = _coord_grid(20, 20)
    c = 1.5  # |mu| = 1.5 > 1 -> folding
    d = torch.stack([c * xx, -c * yy], dim=0)[None]
    assert not is_bijective(d)
    assert not is_faithful(d, delta=0.9)


def test_faithful_fraction_monotonic_in_delta() -> None:
    torch.manual_seed(0)
    # A spatially varying displacement -> a spread of |mu| values.
    d = 0.05 * torch.randn(1, 2, 24, 24).cumsum(dim=-1)
    f_small = beltrami_faithfulness(d, delta=0.1)
    f_large = beltrami_faithfulness(d, delta=0.9)
    assert 0.0 <= f_small <= f_large <= 1.0


def test_delta_must_be_in_unit_interval() -> None:
    d = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="delta"):
        beltrami_faithfulness(d, delta=1.5)
    with pytest.raises(ValueError, match="delta"):
        BeltramiFaithfulnessThreshold(delta=0.0)


# --------------------------------------------------------------------------- #
# Complex displacement input + cert bundle
# --------------------------------------------------------------------------- #


def test_complex_displacement_input_supported() -> None:
    xx, yy = _coord_grid(16, 16)
    c = 0.25
    d_complex = torch.complex(c * xx, -c * yy)[None]  # [1,H,W] complex = c*conj(z)
    mu = beltrami_from_displacement(d_complex)
    assert float(_interior(mu).abs().mean()) == pytest.approx(c, abs=1e-4)


def test_cert_bundle_reports_verdicts() -> None:
    xx, yy = _coord_grid(20, 20)
    d = torch.stack([0.3 * xx, -0.3 * yy], dim=0)[None]
    out = BeltramiFaithfulnessThreshold(delta=0.5).certify(d)
    assert out["faithful"] is True and out["bijective"] is True
    assert out["max_distortion"] == pytest.approx(0.3, abs=1e-2)
    assert 0.0 <= out["faithful_fraction"] <= 1.0


def test_beltrami_finite_at_fold() -> None:
    """A fold (∂_z φ = 1 + ∂_z d → 0) must give a finite |μ|, not NaN/Inf.

    Regression: the denominator ``1 + ∂_z d`` was divided un-guarded, so a fold
    produced Inf that then poisoned ``.abs().max()``. Build d_x = -2·(x-index),
    whose per-pixel ∂_x d_x = -2 ⇒ ∂_z d = -1 ⇒ denominator = 0.
    """
    h = w = 8
    idx = torch.arange(w, dtype=torch.float32)
    d_x = (-2.0 * idx).view(1, 1, 1, w).expand(1, 1, h, w).clone()
    d_y = torch.zeros(1, 1, h, w)
    disp = torch.cat([d_x, d_y], dim=1)
    mu = beltrami_from_displacement(disp)
    assert torch.isfinite(mu).all()
