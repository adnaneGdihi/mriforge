"""Unit tests for the DPSS / Slepian prolate helper.

Covers, per the SSEL spec deliverable:
    (a) eigenvalues are descending and in [0, 1];
    (b) the leading taper concentration eigenvalue is high (> 0.9) for
        reasonable (W, K);
    (c) a band-limited (low-pass) signal has slepian_concentration near 1
        while a high-pass / white signal has lower concentration;
    (d) orthonormality of the returned tapers (Gram ~= I);
plus input validation on n and bandwidth.

CPU-only, deterministic (fixed torch.Generator), fast.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.prolate import (
    prolate_basis,
    prolate_eigis,
    prolate_kernel,
    slepian_concentration,
)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
class TestValidation:
    def test_n_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be > 1"):
            prolate_basis(1, 0.1, 1)
        with pytest.raises(ValueError, match="n must be > 1"):
            prolate_eigis(0, 0.1, 1)

    @pytest.mark.parametrize("bad_w", [0.0, -0.1, 0.5, 0.7, 1.0])
    def test_bandwidth_out_of_range_raises(self, bad_w: float) -> None:
        with pytest.raises(ValueError, match="bandwidth"):
            prolate_basis(64, bad_w, 2)

    def test_n_tapers_exceeds_n_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            prolate_eigis(8, 0.1, 9)

    def test_n_tapers_nonpositive_raises(self) -> None:
        with pytest.raises(ValueError, match="n_tapers must be >= 1"):
            prolate_eigis(8, 0.1, 0)

    def test_kernel_validation(self) -> None:
        with pytest.raises(ValueError):
            prolate_kernel(1, 0.1)
        with pytest.raises(ValueError):
            prolate_kernel(16, 0.6)


# ----------------------------------------------------------------------------
# Kernel structure
# ----------------------------------------------------------------------------
class TestKernel:
    def test_symmetric_and_diagonal(self) -> None:
        n, w = 32, 0.15
        B = prolate_kernel(n, w)
        assert B.shape == (n, n)
        # Symmetric.
        assert torch.allclose(B, B.T, atol=1e-12)
        # Diagonal entries equal 2W.
        diag = torch.diagonal(B)
        assert torch.allclose(diag, torch.full_like(diag, 2.0 * w), atol=1e-10)

    def test_offdiagonal_sinc(self) -> None:
        n, w = 16, 0.2
        B = prolate_kernel(n, w)
        # B[0, 1] should equal sin(2*pi*W*1)/(pi*1).
        expected = math.sin(2.0 * math.pi * w * 1) / (math.pi * 1)
        assert abs(float(B[0, 1]) - expected) < 1e-10


# ----------------------------------------------------------------------------
# (a) eigenvalues descending and in [0, 1]
# ----------------------------------------------------------------------------
class TestEigenvalues:
    def test_descending_and_bounded(self) -> None:
        n, w, k = 128, 0.1, 8
        _, evals = prolate_eigis(n, w, k)
        assert evals.shape == (k,)
        # Descending.
        diffs = evals[1:] - evals[:-1]
        assert torch.all(diffs <= 1e-9), f"not descending: {evals}"
        # In [0, 1].
        assert float(evals.min()) >= -1e-9
        assert float(evals.max()) <= 1.0 + 1e-9

    def test_shannon_number_count(self) -> None:
        """Count of eigenvalues > 1 - eps is ~ 2NW (Shannon number)."""
        n, w = 256, 0.1
        k = n  # all tapers
        _, evals = prolate_eigis(n, w, k)
        shannon = 2 * n * w  # ~51.2
        # Number near 1 should bracket the Shannon number generously.
        n_high = int((evals > 0.99).sum())
        assert shannon * 0.7 < n_high < shannon * 1.3, (
            f"n_high={n_high}, shannon={shannon}"
        )


# ----------------------------------------------------------------------------
# (b) leading taper concentration high
# ----------------------------------------------------------------------------
class TestLeadingConcentration:
    @pytest.mark.parametrize(
        ("n", "w", "k"),
        [(128, 0.1, 4), (256, 0.05, 4), (64, 0.2, 3)],
    )
    def test_leading_eigenvalue_high(self, n: int, w: float, k: int) -> None:
        _, evals = prolate_eigis(n, w, k)
        assert float(evals[0]) > 0.9, f"leading lambda={float(evals[0])}"


# ----------------------------------------------------------------------------
# (c) band-limited vs high-pass concentration
# ----------------------------------------------------------------------------
class TestSlepianConcentration:
    def _rng(self) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(1234)
        return g

    def test_lowpass_near_one(self) -> None:
        """A band-limited (low-frequency) signal concentrates near 1."""
        n, w = 256, 0.1
        k = int(round(2 * n * w))  # Shannon number ~= subspace dim
        t = torch.arange(n, dtype=torch.float64)
        g = self._rng()
        # Sum of a few low-frequency sinusoids inside the band [-W, W].
        sig = torch.zeros(n, dtype=torch.float64)
        for f in (0.01, 0.03, 0.06):  # all < W = 0.1 cycles/sample
            phase = 2 * math.pi * f * t + float(
                torch.rand(1, generator=g).item()
            ) * 2 * math.pi
            sig = sig + torch.cos(phase)
        conc = slepian_concentration(sig, k, w)
        assert float(conc) > 0.9, f"lowpass concentration={float(conc)}"

    def test_highpass_lower(self) -> None:
        """A high-frequency / broadband signal leaks out of the subspace."""
        n, w = 256, 0.1
        k = int(round(2 * n * w))
        t = torch.arange(n, dtype=torch.float64)
        # High-frequency tone near Nyquist, well outside [-W, W].
        highpass = torch.cos(2 * math.pi * 0.45 * t)
        conc_hp = slepian_concentration(highpass, k, w)
        # Compare to the low-pass case from above.
        lowpass = torch.cos(2 * math.pi * 0.04 * t)
        conc_lp = slepian_concentration(lowpass, k, w)
        assert float(conc_hp) < float(conc_lp)
        assert float(conc_hp) < 0.5, f"highpass concentration={float(conc_hp)}"

    def test_white_noise_lower_than_lowpass(self) -> None:
        n, w = 256, 0.08
        k = int(round(2 * n * w))
        g = self._rng()
        white = torch.randn(n, generator=g, dtype=torch.float64)
        lowpass = torch.cos(2 * math.pi * 0.02 * torch.arange(n, dtype=torch.float64))
        conc_white = float(slepian_concentration(white, k, w))
        conc_low = float(slepian_concentration(lowpass, k, w))
        # White energy spreads roughly k/n into the subspace (small);
        # the low-pass tone concentrates strongly.
        assert conc_white < conc_low
        assert conc_white < 0.6

    def test_monotone_sweep_over_frequency(self) -> None:
        """Concentration decreases monotonically as the tone moves out of band."""
        n, w = 256, 0.1
        k = int(round(2 * n * w))
        t = torch.arange(n, dtype=torch.float64)
        freqs = [0.02, 0.08, 0.15, 0.30, 0.45]  # crossing W = 0.1
        concs = [
            float(slepian_concentration(torch.cos(2 * math.pi * f * t), k, w))
            for f in freqs
        ]
        # Spearman sign: rank-correlation of (freq, conc) should be strongly
        # negative. Compute a simple Spearman without scipy.
        rho = _spearman(freqs, concs)
        assert rho <= -0.6, f"rho={rho}, concs={concs}"

    def test_zero_signal_returns_nan(self) -> None:
        n, w, k = 64, 0.1, 4
        zero = torch.zeros(n)
        out = slepian_concentration(zero, k, w)
        assert math.isnan(float(out))

    def test_complex_signal_supported(self) -> None:
        n, w = 128, 0.1
        k = int(round(2 * n * w))
        t = torch.arange(n, dtype=torch.float64)
        # Complex low-frequency exponential inside the band.
        sig = torch.exp(1j * 2 * math.pi * 0.03 * t).to(torch.complex128)
        conc = slepian_concentration(sig, k, w)
        assert 0.0 <= float(conc) <= 1.0
        assert float(conc) > 0.8

    def test_concentration_bounded(self) -> None:
        n, w, k = 128, 0.12, 6
        g = self._rng()
        sig = torch.randn(n, generator=g, dtype=torch.float64)
        conc = float(slepian_concentration(sig, k, w))
        assert 0.0 <= conc <= 1.0

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            slepian_concentration(torch.randn(4, 4), 2, 0.1)


# ----------------------------------------------------------------------------
# (d) orthonormality of returned tapers (Gram ~= I)
# ----------------------------------------------------------------------------
class TestOrthonormality:
    @pytest.mark.parametrize(
        ("n", "w", "k"),
        [(64, 0.1, 6), (128, 0.15, 10), (200, 0.2, 8)],
    )
    def test_gram_is_identity(self, n: int, w: float, k: int) -> None:
        basis = prolate_basis(n, w, k)
        assert basis.shape == (k, n)
        gram = basis @ basis.T  # [k, k]
        eye = torch.eye(k, dtype=gram.dtype)
        assert torch.allclose(gram, eye, atol=1e-8), f"max off = {(gram - eye).abs().max()}"

    def test_unit_norm_rows(self) -> None:
        basis = prolate_basis(96, 0.1, 5)
        norms = torch.linalg.norm(basis, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-8)


# ----------------------------------------------------------------------------
# dtype / device plumbing
# ----------------------------------------------------------------------------
class TestPlumbing:
    def test_float32_request(self) -> None:
        basis = prolate_basis(64, 0.1, 4, dtype=torch.float32)
        assert basis.dtype == torch.float32
        # Still orthonormal at float32 tolerance.
        gram = basis @ basis.T
        assert torch.allclose(gram, torch.eye(4), atol=1e-4)

    def test_deterministic_across_calls(self) -> None:
        a = prolate_basis(80, 0.13, 5)
        b = prolate_basis(80, 0.13, 5)
        assert torch.allclose(a, b, atol=1e-12)


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation, scipy-free."""
    xr = _rank(x)
    yr = _rank(y)
    xr_t = torch.tensor(xr, dtype=torch.float64)
    yr_t = torch.tensor(yr, dtype=torch.float64)
    xr_t = xr_t - xr_t.mean()
    yr_t = yr_t - yr_t.mean()
    denom = (xr_t.norm() * yr_t.norm()).clamp_min(1e-12)
    return float((xr_t @ yr_t) / denom)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks
