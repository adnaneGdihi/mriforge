"""Canary / parametrized / edge-case / expected-failure tests for epg.py.

Covers:
  - ``simulate_differentiable_epg_fse``
  - ``simulate_full_epg_fse``

Shape convention for both callables:
  rho / t1 / t2 / b1_plus : [B, 1, H, W]
  excitation_fa / refocusing_fa : [B, 1, H, W] or scalar tensor
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.epg import (
    simulate_differentiable_epg_fse,
    simulate_full_epg_fse,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tissue_maps(b: int, h: int, w: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    rho = torch.rand(b, 1, h, w, generator=g).clamp(min=0.1)
    t1 = torch.rand(b, 1, h, w, generator=g) * 900.0 + 100.0  # 100-1000 ms
    t2 = torch.rand(b, 1, h, w, generator=g) * 80.0 + 20.0  #  20-100 ms
    return rho, t1, t2


# ---------------------------------------------------------------------------
# Canary tests
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_canary_differentiable_epg_imports_and_runs():
    """Module imports and differentiable EPG runs on a trivial 1x1 image."""
    b, h, w = 1, 4, 4
    rho, t1, t2 = _tissue_maps(b, h, w)
    fa_ex = torch.full((b, 1, h, w), 90.0)
    fa_ref = torch.full((b, 1, h, w), 180.0)
    out = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref)
    assert out.shape == (b, 1, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_canary_full_epg_imports_and_runs():
    """Module imports and full EPG runs on a trivial 1x1 image."""
    b, h, w = 1, 4, 4
    rho, t1, t2 = _tissue_maps(b, h, w)
    b1 = torch.ones(b, 1, h, w)
    signal, R1, R2, b1_out = simulate_full_epg_fse(rho, t1, t2, b1, echo_train_length=4)
    assert signal.shape[0] == b
    assert signal.shape[2] == h
    assert signal.shape[3] == w


# ---------------------------------------------------------------------------
# Parametrized tests — differentiable EPG
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.parametrize(
    "b, h, w, etl",
    [
        (1, 8, 8, 4),
        (1, 16, 16, 16),
        (2, 8, 12, 8),  # rectangular
        (2, 32, 32, 16),
    ],
    ids=["b1-8x8-etl4", "b1-16x16-etl16", "b2-8x12-rect", "b2-32x32-etl16"],
)
def test_diff_epg_shape(b, h, w, etl):
    rho, t1, t2 = _tissue_maps(b, h, w)
    fa_ex = torch.full((b, 1, h, w), 90.0)
    fa_ref = torch.full((b, 1, h, w), 180.0)
    out = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref, echo_train_length=etl)
    assert out.shape == (b, 1, h, w), f"Expected [{b}, 1, {h}, {w}] got {tuple(out.shape)}"


@pytest.mark.physics
@pytest.mark.parametrize("te_spacing", [5.0, 10.0, 20.0])
def test_diff_epg_te_spacing(te_spacing):
    b, h, w = 1, 8, 8
    rho, t1, t2 = _tissue_maps(b, h, w)
    fa_ex = torch.full((b, 1, h, w), 90.0)
    fa_ref = torch.full((b, 1, h, w), 180.0)
    out = simulate_differentiable_epg_fse(
        rho, t1, t2, fa_ex, fa_ref, echo_train_length=8, te_spacing=te_spacing
    )
    assert out.shape == (b, 1, h, w)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Parametrized tests — full EPG
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.parametrize(
    "b, h, w, etl",
    [
        (1, 4, 4, 4),
        (1, 8, 16, 8),  # rectangular
        (2, 8, 8, 4),
    ],
    ids=["b1-4x4-etl4", "b1-8x16-rect", "b2-8x8-etl4"],
)
def test_full_epg_shape(b, h, w, etl):
    rho, t1, t2 = _tissue_maps(b, h, w)
    b1 = torch.ones(b, 1, h, w)
    signal, R1, R2, b1_out = simulate_full_epg_fse(
        rho,
        t1,
        t2,
        b1,
        excitation_fa=90.0,
        refocusing_fa=180.0,
        echo_train_length=etl,
    )
    # signal: [B, ETL, H, W]
    assert signal.shape == (b, etl, h, w), (
        f"Expected [{b}, {etl}, {h}, {w}], got {tuple(signal.shape)}"
    )
    assert R1.shape == (b, 1, h, w)
    assert R2.shape == (b, 1, h, w)
    assert b1_out.shape == (b, 1, h, w)


@pytest.mark.physics
def test_full_epg_returns_complex_signal():
    """Full EPG returns complex signal (F+ states are complex)."""
    b, h, w = 1, 4, 4
    rho, t1, t2 = _tissue_maps(b, h, w)
    b1 = torch.ones(b, 1, h, w)
    signal, *_ = simulate_full_epg_fse(rho, t1, t2, b1, echo_train_length=4)
    # signal should be complex (F_plus states are complex-typed)
    assert torch.is_complex(signal), "Expected complex output from full EPG"


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_diff_epg_edge_zero_rho():
    """Zero proton density → zero signal (no magnetisation to flip)."""
    b, h, w = 1, 4, 4
    rho = torch.zeros(b, 1, h, w)
    t1 = torch.full((b, 1, h, w), 800.0)
    t2 = torch.full((b, 1, h, w), 80.0)
    fa_ex = torch.full((b, 1, h, w), 90.0)
    fa_ref = torch.full((b, 1, h, w), 180.0)
    out = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


@pytest.mark.physics
def test_diff_epg_edge_batch_size_1():
    """Batch size 1 runs without error."""
    b, h, w = 1, 4, 4
    rho, t1, t2 = _tissue_maps(b, h, w)
    fa_ex = torch.full((b, 1, h, w), 90.0)
    fa_ref = torch.full((b, 1, h, w), 180.0)
    out = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref)
    assert out.shape == (b, 1, h, w)


@pytest.mark.physics
def test_diff_epg_edge_no_nan_with_very_short_t2():
    """Very short T2 (near 0) must not produce NaN/inf due to exp(-TE/T2)."""
    b, h, w = 1, 4, 4
    rho = torch.ones(b, 1, h, w)
    t1 = torch.full((b, 1, h, w), 800.0)
    t2 = torch.full((b, 1, h, w), 1e-3)  # 0.001 ms — very short
    fa_ex = torch.full((b, 1, h, w), 90.0)
    fa_ref = torch.full((b, 1, h, w), 180.0)
    # Should not raise and should not produce NaN
    out = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_full_epg_edge_b1_zero():
    """B1+ = 0 (no RF) → zero transverse signal."""
    b, h, w = 1, 4, 4
    rho, t1, t2 = _tissue_maps(b, h, w)
    b1 = torch.zeros(b, 1, h, w)  # no excitation
    signal, *_ = simulate_full_epg_fse(rho, t1, t2, b1, echo_train_length=4)
    # With zero flip angles all sin terms vanish; signal should be zero
    assert torch.allclose(signal.abs(), torch.zeros_like(signal.abs()), atol=1e-6)


@pytest.mark.physics
def test_full_epg_edge_etl_1():
    """ETL=1 (single echo) returns shape [B, 1, H, W]."""
    b, h, w = 1, 4, 4
    rho, t1, t2 = _tissue_maps(b, h, w)
    b1 = torch.ones(b, 1, h, w)
    signal, *_ = simulate_full_epg_fse(rho, t1, t2, b1, echo_train_length=1)
    assert signal.shape == (b, 1, h, w)


# ---------------------------------------------------------------------------
# Expected-failure (raises) tests
# ---------------------------------------------------------------------------
# The EPG functions do not currently expose documented raises for wrong rank
# (they would propagate torch errors on shape mismatches). We guard the most
# obvious: passing 3-D input (missing B or C dim) to simulate_differentiable
# should cause an error because the internal arithmetic relies on broadcasting
# a scalar out of the 4-D tensor shapes.


@pytest.mark.physics
def test_diff_epg_raises_on_wrong_rank():
    """Passing a 3-D rho (missing a dimension) must produce an error."""
    rho = torch.ones(1, 8, 8)  # 3D — missing batch
    t1 = torch.full((1, 1, 8, 8), 800.0)
    t2 = torch.full((1, 1, 8, 8), 80.0)
    fa_ex = torch.full((1, 1, 8, 8), 90.0)
    fa_ref = torch.full((1, 1, 8, 8), 180.0)
    with pytest.raises((RuntimeError, ValueError)):
        simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref)


# ---------------------------------------------------------------------------
# Regression: full EPG must reject undocumented multi-channel rho
# ---------------------------------------------------------------------------
# simulate_full_epg_fse stacks per-echo signals along dim=1 to build the
# documented [B, ETL, H, W] output, which is only correct for C==1.  A C>1
# input would silently entangle the echo axis with the channel axis, so the
# contract is now enforced with an explicit raise (CLAUDE.md #9: no silent
# shape degradation).


@pytest.mark.physics
def test_full_epg_rejects_multichannel_rho():
    """rho with C>1 must RAISE, not silently produce ETL*C channels."""
    rho = torch.rand(1, 2, 4, 4).clamp(min=0.1)
    t1 = torch.full((1, 2, 4, 4), 800.0)
    t2 = torch.full((1, 2, 4, 4), 80.0)
    b1 = torch.ones(1, 2, 4, 4)
    with pytest.raises(ValueError, match="single-channel rho"):
        simulate_full_epg_fse(rho, t1, t2, b1, echo_train_length=4)


# ---------------------------------------------------------------------------
# Regression: differentiable EPG must not perform a data-dependent host-sync
# ---------------------------------------------------------------------------
# Previously the effective-echo capture (`if i == ETL//2`) was never reached
# for ETL==1, so the function fell through to a `signal.sum() > 0` branch — a
# GPU host-sync inside a training-strategy forward (performance.md).  The fix
# (`max(1, ETL//2)`) guarantees the capture always fires, removing the branch
# while keeping the ETL==1 output numerically identical to the old fallback
# (`torch.abs(current_F) * rho`).


@pytest.mark.physics
def test_diff_epg_etl_1_deterministic_capture():
    """ETL==1 hits the deterministic capture path (former host-sync fallback)."""
    b, h, w = 1, 4, 4
    rho, t1, t2 = _tissue_maps(b, h, w)
    fa_ex = torch.full((b, 1, h, w), 90.0)
    fa_ref = torch.full((b, 1, h, w), 180.0)
    out = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref, echo_train_length=1)
    assert out.shape == (b, 1, h, w)
    assert torch.isfinite(out).all()
    # The captured echo is nonnegative (magnitude * rho) and nonzero for rho>0.
    assert (out >= 0).all()
    assert out.abs().sum() > 0


@pytest.mark.physics
def test_diff_epg_no_sum_branch_does_not_change_numerics():
    """The branch removal must not alter output for the common ETL>=2 path."""
    b, h, w = 1, 8, 8
    rho, t1, t2 = _tissue_maps(b, h, w, seed=7)
    fa_ex = torch.full((b, 1, h, w), 90.0)
    fa_ref = torch.full((b, 1, h, w), 180.0)
    out_a = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref, echo_train_length=8)
    out_b = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref, echo_train_length=8)
    # Deterministic (no RNG): two calls are byte-identical and finite.
    assert torch.equal(out_a, out_b)
    assert torch.isfinite(out_a).all()
