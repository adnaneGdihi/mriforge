"""Behavioral coverage for core/metrics/perfusion_metrics.py.

Covers the four public perfusion-MRI metric classes:
- WashSlope        (DCE wash-in / wash-out slope)
- ToftsModel       (Ktrans / Ve / Kep pharmacokinetics)
- BAT              (bolus arrival time)
- NegativeVoxelCount (% negative voxels in perfusion maps)

Each test hand-constructs a deterministic input and asserts the output is
finite and sign/scale-sensible, plus checks degenerate / shape-mismatch
behavior matches the code's intent. CPU-only, seeded.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.perfusion_metrics import (
    BAT,
    NegativeVoxelCount,
    ToftsModel,
    WashSlope,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# WashSlope
# ---------------------------------------------------------------------------


def _triangular_dce(peak: int = 10, length: int = 20) -> torch.Tensor:
    """[1, 1, T] curve that rises linearly to `peak` then decays."""
    curve = torch.zeros(1, 1, length)
    for i in range(length):
        if i <= peak:
            curve[0, 0, i] = float(i)
        else:
            curve[0, 0, i] = peak - (i - peak) * 0.5
    return curve


def test_wash_slope_signs_and_finiteness():
    """Rise-then-decay curve: wash-in > 0, wash-out < 0, all finite."""
    metric = WashSlope(temporal_resolution=1.0)
    out = metric(_triangular_dce(peak=10, length=20), baseline_frames=5)

    for key in ("wash_in_slope", "wash_out_slope", "peak_enhancement"):
        assert key in out
        assert torch.isfinite(out[key]), f"{key} not finite"

    # Contrast uptake is positive, elimination is negative.
    assert out["wash_in_slope"].item() > 0.0
    assert out["wash_out_slope"].item() < 0.0
    # Peak enhancement above baseline must be strictly positive.
    assert out["peak_enhancement"].item() > 0.0


def test_wash_slope_temporal_resolution_scales_inversely():
    """Doubling seconds-per-frame halves the per-second slope magnitude."""
    curve = _triangular_dce(peak=10, length=20)
    fast = WashSlope(temporal_resolution=1.0)(curve, baseline_frames=5)
    slow = WashSlope(temporal_resolution=2.0)(curve, baseline_frames=5)

    # slope = enhancement / (dt_frames * temporal_resolution)
    assert slow["wash_in_slope"].item() == pytest.approx(
        fast["wash_in_slope"].item() / 2.0, rel=1e-5
    )
    # peak_enhancement is independent of the time axis.
    assert slow["peak_enhancement"].item() == pytest.approx(
        fast["peak_enhancement"].item(), rel=1e-6
    )


def test_wash_slope_peak_in_baseline_window_gives_zero_wash_in():
    """When max enhancement occurs at/inside the baseline window, the documented
    branch sets wash-in to exactly 0.0 (no spurious slope)."""
    metric = WashSlope(temporal_resolution=1.0)
    # Monotonically decreasing curve -> peak enhancement at frame 0.
    curve = torch.zeros(1, 1, 20)
    curve[0, 0] = torch.arange(20).float().flip(0)  # 19, 18, ..., 0
    out = metric(curve, baseline_frames=5)

    assert out["wash_in_slope"].item() == 0.0
    assert torch.isfinite(out["wash_out_slope"])


def test_wash_slope_wrong_ndim_raises():
    """A 2-D [B, T] tensor (missing the voxel axis) must raise, not proceed."""
    metric = WashSlope()
    with pytest.raises((ValueError, RuntimeError)):
        metric(torch.zeros(2, 20))


# ---------------------------------------------------------------------------
# ToftsModel
# ---------------------------------------------------------------------------


def test_tofts_model_params_in_grid_range_and_finite():
    """Fitted Ktrans/Ve land inside the documented grid; Kep == Ktrans/Ve;
    fit_error is finite and non-negative."""
    torch.manual_seed(0)
    metric = ToftsModel(temporal_resolution=1.0)
    T = 30
    t = torch.arange(T).float()
    # A plausible tissue-concentration curve (gamma-variate-like).
    tissue = (0.3 * t * torch.exp(-0.1 * t)).unsqueeze(0)  # [1, T]
    out = metric(tissue)

    for key in ("Ktrans", "Ve", "Kep", "fit_error"):
        assert key in out
        assert torch.isfinite(out[key]), f"{key} not finite"

    # Grid bounds from the implementation: Ktrans in [0.01, 1.0], Ve in [0.05, 0.8].
    # linspace endpoints are float32 (e.g. 0.8 -> 0.80000001); allow a tiny tol.
    tol = 1e-5
    assert 0.01 - tol <= out["Ktrans"].item() <= 1.0 + tol
    assert 0.05 - tol <= out["Ve"].item() <= 0.8 + tol
    # Kep is defined as Ktrans / Ve.
    assert out["Kep"].item() == pytest.approx(out["Ktrans"].item() / out["Ve"].item(), rel=1e-5)
    # Mean-squared fit error cannot be negative.
    assert out["fit_error"].item() >= 0.0


def test_tofts_model_custom_aif_does_not_crash():
    """An explicit AIF of matching length yields finite parameters."""
    torch.manual_seed(0)
    metric = ToftsModel(temporal_resolution=1.0)
    T = 24
    t = torch.arange(T).float()
    tissue = (0.2 * t * torch.exp(-0.12 * t)).unsqueeze(0)
    aif = torch.exp(-0.1 * t)  # simple decaying AIF, shape [T]
    out = metric(tissue, aif=aif)
    assert torch.isfinite(out["Ktrans"])
    assert torch.isfinite(out["fit_error"])


# ---------------------------------------------------------------------------
# BAT
# ---------------------------------------------------------------------------


def test_bat_detects_arrival_and_scales_with_resolution():
    """A clean step at frame 10 is detected at 10 * temporal_resolution; a curve
    that never crosses returns the T * resolution no-arrival sentinel."""
    metric = BAT(threshold_factor=2.0)
    B, T = 2, 30
    curve = torch.zeros(B, T)
    curve[0, :10] = 0.1
    curve[0, 10:] = 5.0  # arrival at frame 10
    curve[1, :] = 0.1  # never arrives

    res = metric(curve, baseline_frames=5, temporal_resolution=2.0)
    assert res.shape == (B,)
    assert torch.isfinite(res).all()

    # Arrival frame 10 * resolution 2.0 = 20.0 s.
    assert res[0].item() == pytest.approx(20.0)
    # No arrival -> T * resolution = 30 * 2.0 = 60.0 s sentinel.
    assert res[1].item() == pytest.approx(60.0)


def test_bat_higher_threshold_delays_arrival():
    """A larger threshold_factor never detects arrival earlier than a smaller one."""
    curve = torch.zeros(1, 40)
    curve[0, :5] = torch.tensor([1.0, 1.05, 0.95, 1.0, 1.02])  # noisy baseline
    curve[0, 5:] = torch.linspace(1.1, 4.0, 35)  # gradual rise

    low = BAT(threshold_factor=2.0)(curve, baseline_frames=5)
    high = BAT(threshold_factor=8.0)(curve, baseline_frames=5)
    assert high.item() >= low.item()


def test_bat_flat_curve_returns_no_arrival_sentinel():
    """A perfectly flat curve (zero baseline std) never exceeds its threshold and
    falls back to the T * resolution sentinel rather than crashing."""
    metric = BAT(threshold_factor=2.0)
    T = 30
    flat = torch.full((1, T), 3.0)
    res = metric(flat, baseline_frames=5, temporal_resolution=1.0)
    assert res.item() == pytest.approx(float(T))


# ---------------------------------------------------------------------------
# NegativeVoxelCount
# ---------------------------------------------------------------------------


def test_negative_voxel_count_percentage():
    """2 of 16 voxels negative -> 12.5 %; output is a finite scalar tensor."""
    metric = NegativeVoxelCount()
    m = torch.ones(1, 1, 4, 4)
    m[0, 0, 0, 0] = -1.0
    m[0, 0, 1, 1] = -2.0
    pct = metric(m)
    assert pct.ndim == 0
    assert torch.isfinite(pct)
    assert pct.item() == pytest.approx(12.5)


def test_negative_voxel_count_all_positive_and_all_negative_bounds():
    """All-positive -> 0 %, all-negative -> 100 % (bounded percentage)."""
    metric = NegativeVoxelCount()
    assert metric(torch.ones(1, 1, 2, 2)).item() == pytest.approx(0.0)
    assert metric(-torch.ones(1, 1, 3, 3)).item() == pytest.approx(100.0)
