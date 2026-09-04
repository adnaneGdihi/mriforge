"""Tests for the no-reference *spectral* metric battery (``nr_spectral``).

CPU-only, fast, deterministic. Each metric gets:

* a **monotonicity** test — a graded degradation sweep whose Spearman rank
  correlation with the metric must carry the spec'd sign, and
* an **analytic anchor** — e.g. ``csr ≈ 0`` on a real centered phantom,
  ``bpci ≈ 0`` on Gaussian noise (odd cumulants vanish), ``bpci ≈ 1`` on a
  fully phase-coupled chirp, SHER rank inflation under noise.

``ssel`` depends on the concurrently-authored ``physics.prolate`` module;
we (a) test the graceful ``nan`` path when it is absent and (b) inject a
DPSS-like **stub** module to exercise the full ringing-leakage wiring.
"""

from __future__ import annotations

import math
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.context import MetricContext  # noqa: E402
from spectramr.core.metrics.registry import get_metric  # noqa: E402

PROLATE_MOD = "spectramr.infrastructure.physics.prolate"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""

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


def _disk(h: int = 48, w: int = 48) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    return ((xx**2 + yy**2) < 0.5**2).float(), yy, xx


def _install_stub_prolate() -> None:
    """Inject a DPSS-like stub at ``physics.prolate`` for SSEL wiring tests."""
    mod = types.ModuleType(PROLATE_MOD)

    def prolate_basis(n, bandwidth, n_tapers, *, device=None, dtype=None):
        t = torch.arange(n, dtype=torch.float64)
        rows = []
        for k in range(int(n_tapers)):
            v = torch.cos(math.pi * (k + 0.5) * (t + 0.5) / n)
            rows.append(v / v.norm())
        out = torch.stack(rows)
        return out.to(dtype=dtype or torch.float64)

    def prolate_eigis(n, bandwidth, n_tapers, *, device=None, dtype=None):
        vecs = prolate_basis(n, bandwidth, n_tapers)
        vals = torch.linspace(0.999, 0.5, int(n_tapers), dtype=torch.float64)
        return vecs, vals

    def slepian_concentration(signal_1d, n_tapers, bandwidth):
        sig = signal_1d.to(torch.float64).flatten()
        n = sig.numel()
        basis = prolate_basis(n, bandwidth, min(int(n_tapers), n))
        retained = (basis @ sig).pow(2).sum()
        total = sig.pow(2).sum().clamp_min(1e-12)
        return float((retained / total).clamp(0.0, 1.0).item())

    mod.prolate_basis = prolate_basis
    mod.prolate_eigis = prolate_eigis
    mod.slepian_concentration = slepian_concentration
    sys.modules[PROLATE_MOD] = mod


# ===========================================================================
# csr — Conjugate-Symmetry Residual (↓)
# ===========================================================================
class TestCSR:
    def test_anchor_real_phantom_near_zero(self) -> None:
        """A purely real centered phantom has Hermitian k-space ⇒ CSR ≈ 0."""
        disk, _, _ = _disk()
        img = disk.unsqueeze(0).unsqueeze(0).to(torch.complex64)
        csr = get_metric("csr")
        val = csr(img)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_monotone_phase_corruption(self) -> None:
        """Non-smooth phase corruption raises CSR (↓ metric, ↑ degradation)."""
        torch.manual_seed(0)
        disk, yy, xx = _disk()
        ramp = 0.5 * (xx + yy)  # smooth baseline phase
        csr = get_metric("csr")
        sevs = [0.0, 0.25, 0.5, 1.0, 2.0]
        vals = []
        for s in sevs:
            ph = ramp + s * torch.randn(*disk.shape)
            img = (disk * torch.exp(1j * ph)).unsqueeze(0).unsqueeze(0)
            vals.append(csr(img.to(torch.complex64)))
        rho = _spearman(sevs, vals)
        assert rho >= 0.6, (sevs, vals, rho)
        assert vals[-1] > vals[0]

    def test_returns_float_and_handles_real_input(self) -> None:
        csr = get_metric("csr")
        out = csr(torch.rand(2, 1, 32, 32))  # real [B,1,H,W]
        assert isinstance(out, float)


# ===========================================================================
# sher — Structured Block-Hankel Effective Rank (↓ deviation)
# ===========================================================================
class TestSHER:
    def test_anchor_finite_support_rank_is_bounded(self) -> None:
        """Effective rank of a finite-support disk is a small, finite r*."""
        disk, _, _ = _disk()
        img = disk.unsqueeze(0).unsqueeze(0)
        sher = get_metric("sher")  # raw effective rank
        r = sher(img)
        assert math.isfinite(r)
        # kernel 5x5 ⇒ rank in (1, 25]; a band-limited disk sits well below 25.
        assert 1.0 < r < 25.0

    def test_monotone_noise_inflates_rank(self) -> None:
        """Additive noise fattens the singular spectrum ⇒ SHER rises."""
        torch.manual_seed(0)
        disk, _, _ = _disk()
        base = disk.unsqueeze(0).unsqueeze(0)
        sher = get_metric("sher")
        sigmas = [0.0, 0.05, 0.1, 0.2, 0.4]
        vals = [sher(base + s * torch.randn_like(base)) for s in sigmas]
        rho = _spearman(sigmas, vals)
        assert rho >= 0.6, (sigmas, vals, rho)
        assert vals[-1] > vals[0]

    def test_deviation_mode_penalises_both_directions(self) -> None:
        """With r_star set, |r_eff - r*| is a deviation; perfect match ≈ 0."""
        torch.manual_seed(0)
        disk, _, _ = _disk()
        base = disk.unsqueeze(0).unsqueeze(0)
        raw = get_metric("sher")(base)
        dev = get_metric("sher", r_star=raw)
        assert dev(base) == pytest.approx(0.0, abs=1e-3)
        # A noisier image deviates above zero.
        noisy = base + 0.3 * torch.randn_like(base)
        assert dev(noisy) > 1e-2


# ===========================================================================
# bpci — Bicoherence Phase-Coupling Index (↓)
# ===========================================================================
class TestBPCI:
    def test_anchor_gaussian_noise_near_zero(self) -> None:
        """Gaussian noise has vanishing odd cumulants ⇒ BPCI → 0 with batch."""
        torch.manual_seed(0)
        noise = torch.randn(128, 1, 48, 48)
        bpci = get_metric("bpci")
        assert bpci(noise) < 0.05

    def test_anchor_chirp_fully_coupled(self) -> None:
        """A quadratic-phase chirp is fully phase-coupled ⇒ BPCI ≈ 1."""
        _, yy, xx = _disk()
        h, w = yy.shape
        ph = 0.02 * ((xx * w / 2) ** 2 + (yy * h / 2) ** 2)
        chirp = torch.cos(ph).unsqueeze(0).unsqueeze(0).expand(8, 1, h, w).contiguous()
        bpci = get_metric("bpci")
        assert bpci(chirp) > 0.9

    def test_monotone_noise_decouples_chirp(self) -> None:
        """Adding white noise to a coherent chirp decouples it ⇒ BPCI drops."""
        _, yy, xx = _disk()
        h, w = yy.shape
        ph = 0.02 * ((xx * w / 2) ** 2 + (yy * h / 2) ** 2)
        chirp = torch.cos(ph).unsqueeze(0).unsqueeze(0).expand(32, 1, h, w).contiguous()
        bpci = get_metric("bpci")
        nls = [0.0, 0.5, 1.0, 2.0, 4.0]
        vals = []
        for nl in nls:
            torch.manual_seed(0)
            vals.append(bpci(chirp + nl * torch.randn(32, 1, h, w)))
        # noise level ↑ ⇒ coherence ↓: negative correlation.
        rho = _spearman(nls, vals)
        assert rho <= -0.6, (nls, vals, rho)
        assert vals[0] > vals[-1]


class TestBPCISingleImage:
    """B=1 must not collapse to the constant 1.0 (the sim2rank calling shape).

    Bicoherence normalises ``|E[Y1 Y2 Y3*]|^2`` by ``E|Y1 Y2|^2 E|Y3|^2``. Those
    expectations must run over an *ensemble*. The tests above supply one via the
    batch (B = 8/32/128), which is the estimator's intended form. sim2rank scores
    a single image, and with B=1 every expectation is over one sample, so the
    numerator becomes ``|Y1|^2|Y2|^2|Y3|^2`` — exactly the denominator. b^2 is
    then identically 1 for every ``(k1, k2)`` and every input (Cauchy-Schwarz at
    equality), and averaging 1s over the grid still gives 1.

    So the metric returned a constant ``1.0`` on all 30 sweep axes and was
    recorded "dead" — not because it could not see the degradation, but because
    it could not see the *image*. B=1 now builds the ensemble by tiling.
    """

    @staticmethod
    def _img(seed: int, h: int = 128, w: int = 128):
        torch.manual_seed(seed)
        return torch.randn(1, 1, h, w)

    def test_single_image_is_not_pinned_at_one(self) -> None:
        v = get_metric("bpci")(self._img(0))

        assert v < 0.99, f"B=1 collapsed to the degenerate constant: {v}"

    def test_distinct_single_images_score_differently(self) -> None:
        """The tell for the old bug: every input gave the identical value."""
        bpci = get_metric("bpci")
        vals = [bpci(self._img(s)) for s in range(4)]

        assert len(set(vals)) > 1, f"all inputs scored identically: {vals}"

    def test_single_image_noise_floor_matches_the_ensemble_size(self) -> None:
        """White noise sits near 1/S for S segments — the documented anchor."""
        bpci = get_metric("bpci")
        v = bpci(self._img(0))  # 128x128, 4x4 tiling -> S = 16

        assert 0.0 <= v <= 3.0 / 16, f"noise should sit near the 1/16 floor, got {v}"

    def test_single_image_responds_to_structure(self) -> None:
        """A structured image must be distinguishable from white noise."""
        bpci = get_metric("bpci")
        _, yy, xx = _disk()
        h, w = yy.shape
        ph = 0.02 * ((xx * w / 2) ** 2 + (yy * h / 2) ** 2)
        chirp = torch.cos(ph)[None, None]

        assert bpci(chirp) != pytest.approx(bpci(self._img(0)), abs=1e-3)

    def test_batched_path_is_unchanged(self) -> None:
        """The B>=2 estimator is the intended one and must not be re-routed."""
        torch.manual_seed(0)
        batch = torch.randn(64, 1, 48, 48)
        # 48x48 cannot be tiled at band=4 (needs >=17 bins/axis at n>=2 -> 24 ok,
        # but the batch path must win regardless and stay finite and small).
        v = get_metric("bpci")(batch)

        assert math.isfinite(v) and v < 0.05

    def test_too_small_to_estimate_returns_nan_not_one(self) -> None:
        """Unestimable must read as NaN; 1.0 would be a fabricated 'fully coupled'."""
        v = get_metric("bpci")(torch.randn(1, 1, 16, 16))

        assert math.isnan(v), f"expected NaN for an unestimable input, got {v}"

    def test_degenerate_segment_count_is_rejected(self) -> None:
        from spectramr.core.metrics.nr_spectral import BicoherencePhaseCouplingIndex

        with pytest.raises(ValueError, match="n_segments must be >= 2"):
            BicoherencePhaseCouplingIndex(n_segments=1)


# ===========================================================================
# ssel — Slepian Subspace Energy Leakage (↓)  [needs prolate]
# ===========================================================================
class TestSSEL:
    def test_needs_and_direction(self) -> None:
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.needs("ssel") == ("foreground_mask", "mask")
        assert get_metric("ssel").higher_is_better is False
        assert MetricsRegistry.requires_reference("ssel") is False

    def test_graceful_nan_when_prolate_absent(self, monkeypatch) -> None:
        """If physics.prolate is not importable, SSEL returns nan (not raise)."""
        # Force the import to fail even if a real/stub module is installed.
        monkeypatch.setitem(sys.modules, PROLATE_MOD, None)
        ssel = get_metric("ssel")
        ctx = MetricContext(
            foreground_mask=torch.ones(1, 1, 32, 32), mask=torch.ones(32, 32)
        )
        val = ssel(torch.rand(1, 1, 32, 32), context=ctx)
        assert math.isnan(val)

    def test_smooth_disk_low_leakage_with_stub(self, monkeypatch) -> None:
        _install_stub_prolate()
        monkeypatch.setitem(sys.modules, PROLATE_MOD, sys.modules[PROLATE_MOD])
        disk, _, _ = _disk()
        base = disk.unsqueeze(0).unsqueeze(0)
        ctx = MetricContext(
            foreground_mask=torch.ones_like(base), mask=torch.ones(48, 48)
        )
        val = get_metric("ssel", n_tapers=16)(base, context=ctx)
        assert math.isfinite(val)
        assert 0.0 <= val < 0.2  # well-concentrated ⇒ small leakage

    def test_monotone_ringing_increases_leakage_with_stub(self, monkeypatch) -> None:
        _install_stub_prolate()
        monkeypatch.setitem(sys.modules, PROLATE_MOD, sys.modules[PROLATE_MOD])
        disk, yy, xx = _disk()
        ssel = get_metric("ssel", n_tapers=16)
        ctx = MetricContext(
            foreground_mask=torch.ones(1, 1, 48, 48), mask=torch.ones(48, 48)
        )
        hf = torch.cos(2 * math.pi * 18 * xx) * torch.cos(2 * math.pi * 18 * yy)
        rings = [0.0, 0.1, 0.3, 0.6, 1.0]
        vals = []
        for r in rings:
            sig = (disk + r * hf * disk).unsqueeze(0).unsqueeze(0)
            vals.append(ssel(sig, context=ctx))
        rho = _spearman(rings, vals)
        assert rho >= 0.6, (rings, vals, rho)
        assert vals[-1] > vals[0]

    def test_falls_back_to_magnitude_foreground_with_stub(self, monkeypatch) -> None:
        """No foreground_mask in context ⇒ |x|-threshold fallback, still finite."""
        _install_stub_prolate()
        monkeypatch.setitem(sys.modules, PROLATE_MOD, sys.modules[PROLATE_MOD])
        disk, _, _ = _disk()
        base = disk.unsqueeze(0).unsqueeze(0)
        val = get_metric("ssel", n_tapers=8)(base)  # no context at all
        assert math.isfinite(val)

    def test_real_prolate_not_nan_regression(self, monkeypatch) -> None:
        """Regression: SSEL must be FINITE with the REAL prolate (not the stub).

        Every other SSEL test stubs ``prolate``; the stub's
        ``slepian_concentration`` ignores the bandwidth range, which masked a
        unit bug — SSEL passed the time-half-bandwidth PRODUCT (NW=4.0) where
        the real prolate requires the normalised half-bandwidth W in (0, 0.5),
        so the real call raised ``ValueError`` on every row and SSEL was always
        NaN. The fix passes ``W = NW / n``. Force the real module here so the
        unit contract is actually exercised.
        """
        # Drop any stub a sibling test left in sys.modules → re-import the real one.
        monkeypatch.delitem(sys.modules, PROLATE_MOD, raising=False)
        real = pytest.importorskip(PROLATE_MOD)
        # The real module exposes prolate_kernel; the test stub does not.
        assert hasattr(real, "prolate_kernel"), "expected the real prolate, got a stub"
        disk, _, _ = _disk()
        base = disk.unsqueeze(0).unsqueeze(0)
        # mask=None is the fully-sampled cohort path (uses the NW default branch).
        ctx = MetricContext(foreground_mask=torch.ones_like(base), mask=None)
        val = get_metric("ssel", n_tapers=8)(base, context=ctx)
        assert math.isfinite(
            val
        ), "SSEL is NaN with the real prolate (NW vs W unit bug)"
        assert 0.0 <= val <= 1.0


# ===========================================================================
# registration / contract
# ===========================================================================
class TestRegistration:
    @pytest.mark.parametrize("key", ["csr", "sher", "bpci", "ssel"])
    def test_registered_no_reference_lower_is_better(self, key: str) -> None:
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.is_registered(key)
        inst = get_metric(key)
        assert inst.higher_is_better is False
        assert inst.name == key
        assert MetricsRegistry.requires_reference(key) is False

    def test_module_imports_cleanly(self) -> None:
        import spectramr.core.metrics.nr_spectral as mod

        assert hasattr(mod, "ConjugateSymmetryResidual")
        assert hasattr(mod, "SlepianSubspaceEnergyLeakage")
