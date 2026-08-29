"""Tests for the ``no_reference_metrics`` battery extensions.

Covers the two metrics appended to
``mriforge.core.metrics.no_reference_metrics`` by the NR-metrics spec (§3):

* ``ecfd`` — Empirical-Characteristic-Function Rician Deviation (lower better)
* ``bnac`` — Background-Noise Autocorrelation Colour            (lower better)

Each gets a **monotonicity** check and an **analytic anchor**
(``bnac ≈ 0`` on a white-noise background, ``ecfd ≈ 0`` on a
Rician-distributed magnitude). CPU-only, deterministic. Separate from the
pre-existing ``test_no_reference_metrics.py``.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from mriforge.core.metrics.context import MetricContext  # noqa: E402
from mriforge.core.metrics.registry import get_metric  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spearman(xs: list[float], ys: list[float]) -> float:
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


def _rician_magnitude(
    n: int, nu: float, sigma: float, seed: int = 0
) -> torch.Tensor:
    """A Rician-distributed magnitude image |nu + sigma*N(0,1) + i sigma*N(0,1)|."""
    g = torch.Generator().manual_seed(seed)
    re = nu + sigma * torch.randn(n, n, generator=g)
    im = sigma * torch.randn(n, n, generator=g)
    return torch.sqrt(re**2 + im**2)[None, None]


def _full_fg(n: int) -> MetricContext:
    return MetricContext(foreground_mask=torch.ones(n, n, dtype=torch.bool))


# ---------------------------------------------------------------------------
# ecfd — Empirical-CF Rician Deviation (lower is better)
# ---------------------------------------------------------------------------
def test_ecfd_zero_on_rician_magnitude() -> None:
    """A genuinely Rician magnitude has ≈0 deviation from the fitted model."""
    metric = get_metric("ecfd")
    rician = _rician_magnitude(160, nu=0.0, sigma=0.1, seed=1)  # Rayleigh
    v = metric(rician, context=_full_fg(160))
    assert math.isfinite(v)
    assert v < 1e-2, f"ECFD should ~vanish on a Rician magnitude, got {v}"


def test_ecfd_rises_under_non_rician_corruption() -> None:
    """Mixing in a heavy non-Rician component raises ECFD monotonically."""
    metric = get_metric("ecfd")
    g = torch.Generator().manual_seed(2)
    base = _rician_magnitude(160, nu=0.0, sigma=0.1, seed=2)
    fg = _full_fg(160)
    # Inject an increasing uniform-bright (non-Rician) contamination.
    fracs = [0.0, 0.1, 0.2, 0.4, 0.6]
    vals = []
    contam = 1.0 + torch.rand(1, 1, 160, 160, generator=g)
    for f in fracs:
        mix = (1 - f) * base + f * contam
        vals.append(metric(mix, context=fg))
    assert all(math.isfinite(v) for v in vals)
    assert _spearman(fracs, vals) >= 0.6


def test_ecfd_gates_to_nan_on_empty_foreground() -> None:
    metric = get_metric("ecfd")
    img = _rician_magnitude(64, 0.0, 0.1, seed=3)
    ctx = MetricContext(foreground_mask=torch.zeros(64, 64, dtype=torch.bool))
    v = metric(img, context=ctx)
    assert isinstance(v, float)
    assert math.isnan(v)


# ---------------------------------------------------------------------------
# bnac — Background-Noise Autocorrelation Colour (lower is better)
# ---------------------------------------------------------------------------
def _bg_image(noise: torch.Tensor, n: int = 128) -> torch.Tensor:
    """Foreground square in the centre, the supplied noise as background."""
    fg = torch.zeros(n, n)
    fg[n // 3 : 2 * n // 3, n // 3 : 2 * n // 3] = 1.0
    return (fg + noise)[None, None], (fg > 0.5)


def test_bnac_near_zero_on_white_background() -> None:
    """White background noise has ≈0 off-zero autocorrelation."""
    metric = get_metric("bnac")
    g = torch.Generator().manual_seed(11)
    noise = 0.02 * torch.randn(128, 128, generator=g)
    img, fgm = _bg_image(noise)
    v = metric(img, context=MetricContext(foreground_mask=fgm))
    assert math.isfinite(v)
    assert v < 0.5, f"BNAC should be ~0 on white background, got {v}"


def test_bnac_rises_with_colouring() -> None:
    """Smoothing (g-factor/regularisation colour) raises the autocorrelation."""
    metric = get_metric("bnac")
    g = torch.Generator().manual_seed(12)
    base_noise = 0.02 * torch.randn(128, 128, generator=g)
    sigmas = [0.0, 0.5, 1.0, 2.0, 3.0]
    vals = []
    for s in sigmas:
        if s == 0.0:
            noise = base_noise
        else:
            k = max(3, int(2 * round(3 * s) + 1))
            noise = F.conv2d(
                base_noise[None, None], _gaussian_kernel(k, s), padding=k // 2
            )[0, 0]
        img, fgm = _bg_image(noise)
        vals.append(metric(img, context=MetricContext(foreground_mask=fgm)))
    assert all(math.isfinite(v) for v in vals)
    assert _spearman(sigmas, vals) >= 0.6


def test_bnac_self_declares_direction_and_gates() -> None:
    metric = get_metric("bnac")
    assert metric.name == "bnac"
    assert metric.higher_is_better is False
    # No foreground and a uniform image (no background variance) → nan.
    v = metric(torch.ones(1, 1, 64, 64))
    assert isinstance(v, float)
    assert math.isnan(v)
