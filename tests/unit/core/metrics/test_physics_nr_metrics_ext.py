"""Tests for the ``physics_nr_metrics`` no-reference battery extensions.

Covers the five metrics appended to
``mriforge.core.metrics.physics_nr_metrics`` by the NR-metrics spec (§3):

* ``psdc`` — Power-Spectrum Decay Conformity  (lower is better)
* ``ksk``  — k-Space Spike Kurtosis           (lower is better)
* ``ciea`` — Contrast-Invariant Edge Acutance (higher is better)
* ``hfrw`` — High-Frequency Residual Whiteness (higher is better)
* ``qvcr`` — Quantitative Variance-to-CRLB Ratio (target ≥ 1, higher better)

Each metric gets a **monotonicity** check under a graded degradation and an
**analytic anchor** (ciea intensity-scale invariance, hfrw whiteness ordering,
ksk spike spike-up, psdc slope-shift, qvcr graceful-nan + Var≈CRLB at the
bound). CPU-only, deterministic, fast. This file is *separate* from the
pre-existing ``test_physics_nr_metrics.py`` and touches none of it.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from mriforge.core.metrics.context import MetricContext  # noqa: E402
from mriforge.core.metrics.registry import get_metric  # noqa: E402
from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, no scipy dependency."""

    def _ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def _gaussian_kernel(k: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(k).float() - k // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    k = max(3, int(2 * round(3 * sigma) + 1))
    return F.conv2d(img, _gaussian_kernel(k, sigma), padding=k // 2)


def _disk(n: int = 64, r2: float = 0.5) -> torch.Tensor:
    lin = torch.linspace(-1, 1, n)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return ((xx**2 + yy**2) < r2).float()[None, None]


def _structured(n: int = 96, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = _disk(n, 0.5) * 0.8
    base = base + 0.15 * _disk(n, 0.15)
    return base + 0.03 * torch.rand(1, 1, n, n, generator=g)


# ---------------------------------------------------------------------------
# psdc — Power-Spectrum Decay Conformity (lower is better)
# ---------------------------------------------------------------------------
def test_psdc_returns_finite_float() -> None:
    v = get_metric("psdc")(_structured())
    assert isinstance(v, float)
    assert math.isfinite(v)


def test_psdc_monotone_under_oversmoothing() -> None:
    """Over-smoothing steepens the PSD slope → PSDC departs from beta_ref.

    The sweep stays in the moderate-blur regime (σ ≤ 1.5): under heavy blur the
    high-κ radial bins are driven to the numerical floor and drop out of the
    log-log regression, which *flattens* the apparent slope again — a
    degenerate regime, not a monotonicity failure.
    """
    metric = get_metric("psdc")
    base = _structured(n=128)
    sigmas = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
    # Pin beta_ref + drop the noise-floor term so degradation can only raise
    # PSDC via the slope-shift it is meant to capture.
    ref = MetricContext(nr_params={"psdc_beta_ref": 0.0, "psdc_lambda": 0.0})
    vals = [metric(_blur(base, s), context=ref) for s in sigmas]
    assert all(math.isfinite(v) for v in vals)
    rho = _spearman(sigmas, vals)
    assert rho >= 0.6, f"PSDC should rise with over-smoothing, rho={rho}"


def test_psdc_uses_configured_beta_ref() -> None:
    """The fitted slope is compared against the configured reference."""
    metric = get_metric("psdc")
    img = _structured()
    far = MetricContext(nr_params={"psdc_beta_ref": -50.0, "psdc_lambda": 0.0})
    near = MetricContext(nr_params={"psdc_beta_ref": 0.0, "psdc_lambda": 0.0})
    assert metric(img, context=far) > metric(img, context=near)


# ---------------------------------------------------------------------------
# ksk — k-Space Spike Kurtosis (lower is better)
# ---------------------------------------------------------------------------
def test_ksk_spike_anchor() -> None:
    """≈0-ish for spike-free; explodes with an injected k-space spike."""
    metric = get_metric("ksk")
    img = _structured(seed=3)
    clean = metric(img)
    ks = fft2c(img)
    ks_spike = ks.clone()
    ks_spike[0, 0, 12, 50] = ks_spike[0, 0, 12, 50] + 80.0 * ks.abs().max()
    spiked = metric(ifft2c(ks_spike).abs())
    assert math.isfinite(clean) and math.isfinite(spiked)
    assert spiked > clean + 5.0, f"spike should raise KSK: {clean} -> {spiked}"


def test_ksk_monotone_with_spike_amplitude() -> None:
    """Graded amplitude of a near-DC spike raises KSK monotonically.

    The spike is placed where the radial PSD envelope is large so the
    de-trended outlier grows *smoothly* with amplitude. (A spike out in the
    high-κ tail instead saturates KSK immediately — one dominant outlier
    maximises kurtosis regardless of how much larger it gets — which is exactly
    the binary clean-vs-spiked behaviour the analytic anchor above checks.)
    """
    metric = get_metric("ksk")
    img = _structured(seed=4, n=128)
    ks = fft2c(img)
    cy, cx = img.shape[-2] // 2, img.shape[-1] // 2
    local = ks.abs()[0, 0, cy + 3, cx + 3]
    amps = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    vals = []
    for a in amps:
        ks2 = ks.clone()
        ks2[0, 0, cy + 3, cx + 3] = ks2[0, 0, cy + 3, cx + 3] + a * local
        vals.append(metric(ifft2c(ks2).abs()))
    assert all(math.isfinite(v) for v in vals)
    assert _spearman(amps, vals) >= 0.6


# ---------------------------------------------------------------------------
# ciea — Contrast-Invariant Edge Acutance (higher is better)
# ---------------------------------------------------------------------------
def test_ciea_intensity_scale_invariance() -> None:
    """CIEA is unchanged under a global intensity scaling x -> s x."""
    metric = get_metric("ciea")
    img = _structured(seed=5)
    base = metric(img)
    for s in (0.25, 3.0, 17.0):
        scaled = metric(img * s)
        assert math.isfinite(scaled)
        assert abs(scaled - base) <= 1e-3 * max(1.0, abs(base)), (
            f"CIEA not scale-invariant at s={s}: {base} vs {scaled}"
        )


def test_ciea_drops_under_blur() -> None:
    metric = get_metric("ciea")
    base = _structured(seed=6)
    sigmas = [0.0, 0.5, 1.0, 2.0, 3.0]
    vals = [metric(_blur(base, s)) for s in sigmas]
    assert all(math.isfinite(v) for v in vals)
    # higher_is_better → acutance falls as blur rises (negative correlation).
    assert _spearman(sigmas, vals) <= -0.6


# ---------------------------------------------------------------------------
# hfrw — High-Frequency Residual Whiteness (higher is better)
# ---------------------------------------------------------------------------
def test_hfrw_white_higher_than_smoothed() -> None:
    """White-noise residual is far whiter than an over-smoothed one."""
    metric = get_metric("hfrw")
    g = torch.Generator().manual_seed(7)
    white = torch.randn(1, 1, 128, 128, generator=g)
    sfm_white = metric(white)
    sfm_smooth = metric(_blur(white, 4.0))
    assert 0.0 < sfm_white <= 1.0
    assert 0.0 <= sfm_smooth <= 1.0
    # AM-GM: a flat (white) residual maximises SFM; over-smoothing collapses it.
    assert sfm_white > sfm_smooth + 0.2
    assert sfm_white > 0.3


def test_hfrw_monotone_decreasing_with_blur() -> None:
    metric = get_metric("hfrw")
    g = torch.Generator().manual_seed(8)
    base = _structured(seed=8) + 0.2 * torch.rand(1, 1, 96, 96, generator=g)
    sigmas = [0.0, 0.5, 1.0, 2.0, 3.0]
    vals = [metric(_blur(base, s)) for s in sigmas]
    assert all(math.isfinite(v) for v in vals)
    assert _spearman(sigmas, vals) <= -0.6


# ---------------------------------------------------------------------------
# qvcr — Quantitative Variance-to-CRLB Ratio (target ≥ 1)
# ---------------------------------------------------------------------------
def test_qvcr_gates_to_nan_without_context() -> None:
    """No acq_params / qmaps ⇒ graceful nan (never raises)."""
    v = get_metric("qvcr")(_structured())
    assert isinstance(v, float)
    assert math.isnan(v)


def test_qvcr_registers_and_self_declares_direction() -> None:
    m = get_metric("qvcr")
    assert m.name == "qvcr"
    assert m.higher_is_better is True


def test_qvcr_unbiased_fit_attains_bound() -> None:
    """A tiny Bloch round-trip: with Var supplied == CRLB, QVCR ≈ 1."""
    metric = get_metric("qvcr")
    img = _structured(seed=9)
    ctx = MetricContext(
        acq_params={"tr": 15.0, "te": 5.0, "flip_angle_rad": 0.3},
        noise_cov=torch.tensor(1e-4),
        nr_params={"qvcr_qmaps": {"t1": 1000.0, "t2": 80.0, "pd": 1.0}},
    )
    v = metric(img, context=ctx)
    assert math.isfinite(v)
    # var defaulted to the CRLB → ratio exactly at the information bound.
    assert abs(v - 1.0) < 1e-6


def test_qvcr_overregularised_below_bound() -> None:
    """An estimator with Var < CRLB (over-regularised) certifies QVCR < 1."""
    metric = get_metric("qvcr")
    img = _structured(seed=10)
    base_ctx = {
        "acq_params": {"tr": 15.0, "te": 5.0, "flip_angle_rad": 0.3},
        "noise_cov": torch.tensor(1e-4),
    }
    # First read the CRLB-at-bound value, then halve the reported variance.
    at_bound = metric(
        img,
        context=MetricContext(
            **base_ctx,
            nr_params={"qvcr_qmaps": {"t1": 1000.0, "t2": 80.0}},
        ),
    )
    assert math.isfinite(at_bound)
    # Supply explicit per-parameter variances well below CRLB by using a
    # tiny absolute variance; the ratio must drop under the bound.
    low = metric(
        img,
        context=MetricContext(
            **base_ctx,
            nr_params={
                "qvcr_qmaps": {"t1": 1000.0, "t2": 80.0},
                "qvcr_var": [1e-30, 1e-30],
            },
        ),
    )
    assert math.isfinite(low)
    assert low < at_bound
