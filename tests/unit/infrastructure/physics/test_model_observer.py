"""Tests for the Channelized Hotelling Observer (CHO) — spec §8.3.

CPU-only, fast, deterministic (RNG via explicit ``torch.Generator``).

Anchors / monotonicity covered:
- ``auc -> 0.5`` and ``d'^2 ~ 0`` when present == absent (no signal).
- ``d'^2`` rises monotonically with inserted signal amplitude; ``auc -> 1``.
- channels are finite and centred (zero-mean, unit-norm columns).
- determinism: identical Generator seed -> identical d'^2 / AUC.
- analytic single-channel anchor: d'^2 ≈ (μ/σ)^2 for a 1-D Gaussian shift.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.model_observer import (
    ChannelizedHotellingObserver,
    ddog_channel_bank,
    gabor_channel_bank,
)


def _spearman_sign(x: list[float], y: list[float]) -> float:
    """Spearman rho via Pearson on ranks (small-n, ties unlikely here)."""
    tx = torch.tensor(x, dtype=torch.float64)
    ty = torch.tensor(y, dtype=torch.float64)
    rx = tx.argsort().argsort().to(torch.float64)
    ry = ty.argsort().argsort().to(torch.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if denom == 0:
        return 0.0
    return float((rx @ ry) / denom)


# --------------------------------------------------------------------- banks
def test_gabor_bank_shape_and_finite() -> None:
    u = gabor_channel_bank((24, 24), n_channels=10)
    assert u.shape == (24 * 24, 10)
    assert torch.isfinite(u).all()


def test_ddog_bank_shape_and_finite() -> None:
    u = ddog_channel_bank((24, 24), n_channels=7)
    assert u.shape == (24 * 24, 7)
    assert torch.isfinite(u).all()


@pytest.mark.parametrize("bank_fn", [gabor_channel_bank, ddog_channel_bank])
def test_channels_are_centred_unit_norm(bank_fn) -> None:
    """Each channel column is zero-mean (DC removed) and unit L2 norm."""
    u = bank_fn((20, 20), n_channels=6).to(torch.float64)
    col_mean = u.mean(dim=0)
    col_norm = torch.linalg.norm(u, dim=0)
    assert torch.allclose(col_mean, torch.zeros_like(col_mean), atol=1e-5)
    assert torch.allclose(col_norm, torch.ones_like(col_norm), atol=1e-4)


def test_bank_invalid_n_channels_raises() -> None:
    with pytest.raises(ValueError):
        gabor_channel_bank((16, 16), n_channels=0)
    with pytest.raises(ValueError):
        ddog_channel_bank((16, 16), n_channels=0)


# ------------------------------------------------------------- no-signal anchor
def test_auc_half_and_dsq_zero_with_no_signal() -> None:
    """present == absent -> d'^2 ~ 0 and AUC ~ 0.5."""
    u = gabor_channel_bank((16, 16), n_channels=8)
    g = torch.Generator().manual_seed(7)
    ens = torch.randn(16 * 16, 300, generator=g)
    cho = ChannelizedHotellingObserver(u)
    assert cho.detectability(ens, ens.clone()) == pytest.approx(0.0, abs=1e-6)
    assert cho.auc(ens, ens.clone()) == pytest.approx(0.5, abs=1e-6)


# -------------------------------------------------------- monotonicity in amp
@pytest.mark.parametrize("bank_fn", [gabor_channel_bank, ddog_channel_bank])
def test_dsq_and_auc_rise_with_signal_amplitude(bank_fn) -> None:
    """d'^2 and AUC increase monotonically with the inserted signal amplitude."""
    shape = (20, 20)
    u = bank_fn(shape, n_channels=8)
    cho = ChannelizedHotellingObserver(u)
    g = torch.Generator().manual_seed(123)
    n_pix = shape[0] * shape[1]
    n_samp = 400
    absent = torch.randn(n_pix, n_samp, generator=g)
    # fixed signal template = a mid-band channel (so it is detectable by U)
    signal = u[:, u.shape[1] // 2].unsqueeze(1)

    amps = [0.0, 0.5, 1.0, 2.0, 3.0]
    dsqs: list[float] = []
    aucs: list[float] = []
    for a in amps:
        gp = torch.Generator().manual_seed(999)  # same present noise across amps
        present = torch.randn(n_pix, n_samp, generator=gp) + a * signal
        dsqs.append(cho.detectability(present, absent))
        aucs.append(cho.auc(present, absent))

    # All finite, strictly-positive direction.
    assert all(math.isfinite(v) for v in dsqs + aucs)
    assert _spearman_sign(amps, dsqs) >= 0.6
    assert _spearman_sign(amps, aucs) >= 0.6
    # Strong-signal end: d'^2 clearly above the no-signal end; AUC well above 0.5.
    assert dsqs[-1] > dsqs[0] + 0.5
    assert aucs[-1] > 0.7
    # Weak / no signal end stays near chance.
    assert aucs[0] < 0.6


def test_strong_signal_auc_approaches_one() -> None:
    """High signal-to-noise ratio drives AUC -> 1.

    DDoG channels are well-separated radial bands, so a signal aligned with one
    channel lives in a low-noise channel direction; with a small noise level
    the detectability is large.
    """
    u = ddog_channel_bank((20, 20), n_channels=8)
    cho = ChannelizedHotellingObserver(u)
    g = torch.Generator().manual_seed(5)
    n_pix = 20 * 20
    noise = 0.02
    absent = noise * torch.randn(n_pix, 400, generator=g)
    signal = u[:, 4].unsqueeze(1)
    gp = torch.Generator().manual_seed(6)
    present = noise * torch.randn(n_pix, 400, generator=gp) + 2.0 * signal
    assert cho.auc(present, absent) > 0.99


# ------------------------------------------------------------- analytic anchor
def test_single_channel_dsq_matches_snr_squared() -> None:
    """1-channel observer on a pure shifted Gaussian -> d'^2 ≈ (μ/σ)^2.

    With one channel and aligned signal, the channel response is g = c + shift,
    so d'^2 collapses to the univariate (mean-diff)^2 / variance = SNR^2.
    """
    n_pix = 64
    # single unit-norm, zero-mean channel
    raw = torch.randn(n_pix, 1, generator=torch.Generator().manual_seed(1))
    raw = raw - raw.mean()
    u = raw / torch.linalg.norm(raw)
    cho = ChannelizedHotellingObserver(u)

    sigma = 0.5
    mu = 1.2  # channel-domain mean shift we will inject
    n_samp = 4000
    g0 = torch.Generator().manual_seed(10)
    g1 = torch.Generator().manual_seed(11)
    # Build ensembles whose channel response has variance sigma^2 and the
    # present mean shifted by mu *in channel space*. Inject along u so that
    # u^T x reproduces the intended 1-D statistics.
    base0 = (sigma * torch.randn(1, n_samp, generator=g0)) * u
    base1 = (sigma * torch.randn(1, n_samp, generator=g1) + mu) * u
    dsq = cho.detectability(base1, base0)
    expected = (mu / sigma) ** 2  # = 5.76
    # generous tolerance for finite-sample covariance estimate
    assert dsq == pytest.approx(expected, rel=0.15)
    auc = cho.auc(base1, base0)
    assert auc == pytest.approx(0.5 * (1 + math.erf((mu / sigma) / 2.0)), rel=0.05)


# ----------------------------------------------------------------- determinism
def test_deterministic_given_generator() -> None:
    u = gabor_channel_bank((16, 16), n_channels=8)
    cho = ChannelizedHotellingObserver(u)

    def build() -> tuple[float, float]:
        g0 = torch.Generator().manual_seed(42)
        g1 = torch.Generator().manual_seed(43)
        absent = torch.randn(256, 250, generator=g0)
        present = torch.randn(256, 250, generator=g1) + 1.5 * u[:, 4].unsqueeze(1)
        return cho.detectability(present, absent), cho.auc(present, absent)

    d1, a1 = build()
    d2, a2 = build()
    assert d1 == d2
    assert a1 == a2


# ------------------------------------------------------------------- w_cho api
def test_w_cho_realises_dsq() -> None:
    """d'^2 == Δḡ · w_cho (the template that achieves the detectability)."""
    u = gabor_channel_bank((16, 16), n_channels=8)
    cho = ChannelizedHotellingObserver(u)
    g0 = torch.Generator().manual_seed(2)
    g1 = torch.Generator().manual_seed(3)
    absent = torch.randn(256, 300, generator=g0)
    present = torch.randn(256, 300, generator=g1) + 2.0 * u[:, 2].unsqueeze(1)
    dsq = cho.detectability(present, absent)
    w = cho.w_cho
    assert w.shape == (u.shape[1],)
    # Recompute Δḡ in channel space and verify d'^2 = Δḡ·w.
    uu = u.to(torch.float64)
    g1c = uu.t() @ present.to(torch.float64)
    g0c = uu.t() @ absent.to(torch.float64)
    d = g1c.mean(1) - g0c.mean(1)
    assert float(torch.dot(d, w.to(torch.float64))) == pytest.approx(dsq, rel=1e-4)


def test_w_cho_before_call_raises() -> None:
    u = gabor_channel_bank((16, 16), n_channels=4)
    cho = ChannelizedHotellingObserver(u)
    with pytest.raises(RuntimeError):
        _ = cho.w_cho


# --------------------------------------------------------------- degenerate / guards
def test_bad_channels_shape_raises() -> None:
    with pytest.raises(ValueError):
        ChannelizedHotellingObserver(torch.randn(10))


def test_mismatched_pixels_returns_nan() -> None:
    u = gabor_channel_bank((16, 16), n_channels=4)
    cho = ChannelizedHotellingObserver(u)
    # present has wrong n_pixels -> projection raises internally -> nan
    present = torch.randn(100, 50)
    absent = torch.randn(100, 50)
    assert math.isnan(cho.detectability(present, absent))
    assert math.isnan(cho.auc(present, absent))


def test_single_sample_returns_nan() -> None:
    """< 2 total samples cannot form a covariance -> graceful nan, not raise."""
    u = gabor_channel_bank((16, 16), n_channels=4)
    cho = ChannelizedHotellingObserver(u)
    # 1 present + 0 absent would be < 2; build 1+0 by zero-width absent.
    present = torch.randn(256, 1)
    absent = torch.randn(256, 1)
    # total = 2 here is fine; force the < 2 path with empty ensembles instead.
    empty = torch.randn(256, 0)
    assert math.isnan(cho.detectability(present[:, :0], empty))


def test_ridge_lands_on_the_input_device() -> None:
    """The ridge is built with ``torch.eye(...)`` — it MUST carry ``device=``.

    Without it the identity lands on CPU while ``k`` is on the accelerator,
    ``torch.linalg.solve`` raises a device-mismatch ``RuntimeError``, and
    :meth:`detectability`'s ``except (RuntimeError, ValueError) -> nan`` converts a
    device bug into a silent NaN. That is exactly how ``task_based_detectability``
    came back ``never_computed`` on all 600 (axis, severity) cells of the 2026-07-27
    GPU brain run while computing fine in every CPU test — the graceful-NaN contract
    that keeps a metric sweep alive also hides the crash that caused it.

    Trivially true on CPU; it is the GPU twin below that has teeth. Kept CPU-side so
    the contract is stated where CI can read it.
    """
    u = gabor_channel_bank((16, 16), n_channels=4)
    cho = ChannelizedHotellingObserver(u)
    g = torch.randn(256, 32, generator=torch.Generator().manual_seed(0))
    _d, k = cho._channel_stats(g + 0.1, g)
    assert k.device == g.device


@pytest.mark.gpu
def test_detectability_is_finite_on_a_non_cpu_device() -> None:
    """The regression: a cross-device ridge NaN'd the whole metric."""
    if not torch.cuda.is_available():  # pragma: no cover - marker also gates this
        pytest.skip("needs CUDA")
    dev = torch.device("cuda")
    u = gabor_channel_bank((16, 16), n_channels=4).to(dev)
    cho = ChannelizedHotellingObserver(u)
    g = torch.randn(256, 64, generator=torch.Generator().manual_seed(0)).to(dev)
    d_sq = cho.detectability(g + 0.2, g)
    assert math.isfinite(d_sq), "cross-device ridge turned into a silent NaN"
    assert d_sq > 0.0
