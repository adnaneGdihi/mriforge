"""Tests for the no-reference *artifact* metric battery (``nr_artifact.py``).

CPU-only, deterministic, fast. For each of the five metrics we assert:

* a **monotonicity** sweep — a graded degradation moves the metric in the
  spec'd direction (Spearman sign correct, ``|rho| >= 0.6`` where feasible);
* every **analytic anchor** the spec gives —
  GRI ``≈ 0.09`` on a truncated-k-space step edge; CPGE strongly negative on
  smooth-phase; PEGC / ARE ``≈ 0`` on a clean / ``R=1`` image.

Synthetic phantoms only; no GPU, no real data, fixed RNG.
"""

from __future__ import annotations

import math
import struct

import pytest
import torch

from mriforge.core.metrics.context import MetricContext
from mriforge.core.metrics.nr_artifact import (
    AliasingReplicaEnergy,
    ComplexPhaseGradientEntropy,
    GibbsRingingIndex,
    PhaseEncodeGhostCoherence,
    ResolutionAnisotropyRatio,
)
from mriforge.core.metrics.registry import MetricsRegistry

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spearman_sign(values: list[float]) -> float:
    """Spearman rho between the index 0..n-1 and ``values`` (rank correlation)."""
    n = len(values)
    x = torch.arange(n, dtype=torch.float64)
    y = torch.tensor(values, dtype=torch.float64)
    # rank-transform y
    yr = y.argsort().argsort().to(torch.float64)
    xr = x  # already monotone integer ranks
    xm, ym = xr.mean(), yr.mean()
    cov = ((xr - xm) * (yr - ym)).sum()
    denom = torch.sqrt(((xr - xm) ** 2).sum() * ((yr - ym) ** 2).sum()) + 1e-12
    return float(cov / denom)


def _disk(n: int = 64, radius: float = 0.4, cx: float = 0.0, cy: float = 0.0):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    return ((xx - cx) ** 2 + (yy - cy) ** 2 < radius**2).float()


def _truncated_step_edge(n: int = 128, keep_frac: float = 0.20) -> torch.Tensor:
    """Vertical step edge after symmetric k-space truncation (Gibbs ringing)."""
    step = torch.zeros(n, n)
    step[:, n // 2 :] = 1.0
    k = torch.fft.fftshift(torch.fft.fft2(step))
    keep = torch.zeros_like(k)
    c = n // 2
    bw = int(keep_frac * n)
    keep[:, c - bw : c + bw] = 1.0
    return torch.fft.ifft2(torch.fft.ifftshift(k * keep)).abs()


# ---------------------------------------------------------------------------
# registration / contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,cls,hib,needs",
    [
        ("pegc", PhaseEncodeGhostCoherence, False, ("foreground_mask",)),
        ("are", AliasingReplicaEnergy, False, ("acceleration", "foreground_mask")),
        ("gri", GibbsRingingIndex, False, ()),
        ("cpge", ComplexPhaseGradientEntropy, False, ("foreground_mask",)),
        ("rar", ResolutionAnisotropyRatio, False, ()),
    ],
)
def test_registration_and_contract(key, cls, hib, needs):
    inst = MetricsRegistry.get(key)
    assert isinstance(inst, cls)
    assert inst.name == key
    assert isinstance(inst.higher_is_better, bool)
    assert inst.higher_is_better is hib
    assert MetricsRegistry.requires_reference(key) is False
    assert MetricsRegistry.needs(key) == needs


def test_all_metrics_return_python_float():
    img = _disk()[None, None]
    ctx = MetricContext(acceleration=4.0)
    for inst in (
        PhaseEncodeGhostCoherence(),
        AliasingReplicaEnergy(),
        GibbsRingingIndex(),
        ResolutionAnisotropyRatio(),
    ):
        v = inst(img, context=ctx)
        assert isinstance(v, float)
    cplx = (_disk() * torch.exp(1j * torch.zeros(64, 64)))[None, None]
    assert isinstance(ComplexPhaseGradientEntropy()(cplx, context=ctx), float)


def test_no_requires_grad_scalar_warning_on_grad_input():
    """The SFA autograd probe calls every metric on a ``requires_grad`` leaf.

    These NR metrics reduce to a Python ``float`` (non-differentiable), so the
    magnitude/complex helpers must ``detach`` at the boundary — otherwise
    ``float(tensor)`` on a grad-carrying tensor trips a UserWarning on every
    call (the noise the user reported). Regression for that detach.
    """
    import warnings

    from mriforge.core.metrics.nr_artifact import _as_bchw_magnitude, _as_bhw_complex

    grad_img = _disk()[None, None].clone().requires_grad_(True)
    assert not _as_bchw_magnitude(grad_img).requires_grad

    grad_cplx = (
        (_disk() * torch.exp(1j * torch.zeros(64, 64)))[None, None].clone().requires_grad_(True)
    )
    assert not _as_bhw_complex(grad_cplx).requires_grad

    ctx = MetricContext(acceleration=4.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning -> test failure
        for inst in (
            PhaseEncodeGhostCoherence(),
            AliasingReplicaEnergy(),
            GibbsRingingIndex(),
            ResolutionAnisotropyRatio(),
        ):
            assert isinstance(inst(grad_img, context=ctx), float)
        assert isinstance(ComplexPhaseGradientEntropy()(grad_cplx, context=ctx), float)


# ---------------------------------------------------------------------------
# pegc
# ---------------------------------------------------------------------------


def test_pegc_clean_image_near_zero():
    """A clean (ghost-free) image has ≈0 phase-encode ghost coherence."""
    obj = _disk(radius=0.22, cx=0.3)[None, None]
    val = PhaseEncodeGhostCoherence(num_shifts=24, min_shift_frac=0.05)(
        obj, context=MetricContext()
    )
    assert val == pytest.approx(0.0, abs=1e-3)


def test_pegc_monotonic_in_ghost_amplitude():
    """PEGC rises as a faint shifted ghost replica is added (sub-threshold)."""
    obj = _disk(n=64, radius=0.18, cx=0.3)[None, None]
    pegc = PhaseEncodeGhostCoherence(num_shifts=24, min_shift_frac=0.05)
    vals = []
    for amp in (0.0, 0.01, 0.02, 0.03, 0.04):
        ghost = torch.roll(obj, shifts=-26, dims=-1) * amp  # into background
        vals.append(pegc(obj + ghost, context=MetricContext()))
    # higher_is_better=False ⇒ degradation INCREASES the metric ⇒ rho > 0
    assert _spearman_sign(vals) >= 0.6
    assert vals[-1] > vals[0]


def test_pegc_honours_pe_axis_param():
    """pe_axis nr_param picks the axis the ghost is shifted along."""
    obj = _disk(n=64, radius=0.18, cy=0.3)[None, None]
    ghost = torch.roll(obj, shifts=-26, dims=-2) * 0.03  # shift along ROWS (-2)
    img = obj + ghost
    pegc = PhaseEncodeGhostCoherence(num_shifts=24, min_shift_frac=0.05)
    along_rows = pegc(img, context=MetricContext(nr_params={"pe_axis": -2}))
    along_cols = pegc(img, context=MetricContext(nr_params={"pe_axis": -1}))
    assert along_rows > along_cols


def test_pegc_nan_on_nonfinite():
    bad = torch.full((1, 1, 16, 16), float("nan"))
    assert math.isnan(PhaseEncodeGhostCoherence()(bad, context=MetricContext()))


# ---------------------------------------------------------------------------
# are
# ---------------------------------------------------------------------------


def test_are_nan_without_acceleration():
    """ARE needs the acceleration; absent context ⇒ nan (not a crash)."""
    img = _disk()[None, None]
    assert math.isnan(AliasingReplicaEnergy()(img, context=MetricContext()))


def test_are_zero_at_unit_acceleration():
    """Anchor: Δ_R = N at R=1 ⇒ cyclic shift wraps the object onto itself ⇒ 0."""
    img = _disk(radius=0.45)[None, None]
    val = AliasingReplicaEnergy()(img, context=MetricContext(acceleration=1.0))
    assert val == pytest.approx(0.0, abs=1e-3)


def test_are_monotonic_in_residual_replica_amplitude():
    """At fixed R, ARE rises with the amplitude of a residual aliasing replica."""
    obj = _disk(n=64, radius=0.16, cx=0.25)[None, None]
    are = AliasingReplicaEnergy()
    r = 4.0
    delta_r = round(64 / r)
    vals = []
    for amp in (0.0, 0.1, 0.2, 0.3, 0.4):
        # NB: keep amp small enough vs the object so the threshold mask does
        # NOT absorb the replica into the foreground at amp=0 we already
        # measure the object's own geometric replica; the *trend* is what we
        # assert, and it is monotone increasing in amp.
        repl = torch.roll(obj, shifts=delta_r, dims=-1) * amp
        vals.append(are(obj + repl, context=MetricContext(acceleration=r)))
    # strip the saturated amp=0 baseline (object self-replica) and test the
    # genuine residual-aliasing regime.
    trend = vals[1:]
    assert _spearman_sign(trend) >= 0.6
    assert trend[-1] > trend[0]


# ---------------------------------------------------------------------------
# gri
# ---------------------------------------------------------------------------


def test_gri_gibbs_constant_anchor():
    """Anchor: GRI ≈ 0.09 (Gibbs constant) on a truncated-k-space step edge."""
    edge = _truncated_step_edge(n=128, keep_frac=0.20)
    gri = GibbsRingingIndex()(edge[None, None])
    # Gibbs first-overshoot fraction is ~0.0895; a discrete profile read lands
    # in [0.06, 0.13]. Centre is the textbook 0.09.
    assert 0.06 <= gri <= 0.13


def test_gri_collapses_under_oversmoothing():
    """Over-smoothing (windowing/blur) suppresses the overshoot toward 0."""
    edge = _truncated_step_edge(n=128, keep_frac=0.20)
    import torch.nn.functional as f

    sig = 2.0
    ax = torch.arange(-5, 6).float()
    g = torch.exp(-(ax**2) / (2 * sig**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None]
    smoothed = f.conv2d(edge[None, None], k, padding=5)
    gri_sharp = GibbsRingingIndex()(edge[None, None])
    gri_smooth = GibbsRingingIndex()(smoothed)
    assert gri_smooth < gri_sharp
    assert gri_smooth < 0.05


def test_gri_monotonic_with_smoothing_strength():
    """GRI decreases monotonically with increasing Gaussian smoothing sigma."""
    import torch.nn.functional as f

    edge = _truncated_step_edge(n=128, keep_frac=0.20)
    gri = GibbsRingingIndex()
    vals = []
    for sig in (0.3, 0.8, 1.5, 2.5):
        n = max(3, int(6 * sig) | 1)
        ax = torch.arange(n).float() - n // 2
        g = torch.exp(-(ax**2) / (2 * sig**2))
        g = g / g.sum()
        k = (g[:, None] * g[None, :])[None, None]
        sm = f.conv2d(edge[None, None], k, padding=n // 2)
        vals.append(gri(sm))
    # ringing decreases as smoothing grows ⇒ rho < 0
    assert _spearman_sign(vals) <= -0.6


def test_gri_nan_on_flat_image():
    """No edges ⇒ no overshoot ratios ⇒ nan (graceful, no crash)."""
    flat = torch.ones(1, 1, 32, 32)
    assert math.isnan(GibbsRingingIndex()(flat))


# ---------------------------------------------------------------------------
# cpge
# ---------------------------------------------------------------------------


def _complex_phantom(phase_noise: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = 64
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    mag = (xx**2 + yy**2 < 0.4).float()
    smooth = 2.0 * xx + 1.5 * yy
    ph = smooth + phase_noise * torch.randn(n, n, generator=g)
    obj = mag * torch.exp(1j * ph)
    bg = 0.02 * (torch.randn(n, n, generator=g) + 1j * torch.randn(n, n, generator=g))
    return (obj + (1 - mag) * bg)[None, None]


def test_cpge_strongly_negative_on_smooth_phase():
    """Anchor: smooth foreground phase + noisy background ⇒ strongly negative."""
    val = ComplexPhaseGradientEntropy()(_complex_phantom(0.0), context=MetricContext())
    assert val < -1.0


def test_cpge_monotonic_under_phase_corruption():
    """CPGE rises toward 0 as the foreground phase is corrupted."""
    cpge = ComplexPhaseGradientEntropy()
    vals = [cpge(_complex_phantom(p), context=MetricContext()) for p in (0.0, 0.3, 0.6, 1.0, 2.0)]
    # higher_is_better=False ⇒ degradation increases the metric ⇒ rho > 0
    assert _spearman_sign(vals) >= 0.6
    assert vals[-1] > vals[0]


def test_cpge_nan_on_real_only_input():
    """No phase information (real-only single channel) ⇒ nan."""
    real = _disk()[None, None]
    assert math.isnan(ComplexPhaseGradientEntropy()(real, context=MetricContext()))


def test_cpge_accepts_interleaved_real_imag():
    """A [B,2,H,W] interleaved real/imag layout yields a finite CPGE."""
    cplx = _complex_phantom(0.0)[0, 0]  # [H,W] complex
    inter = torch.stack([cplx.real, cplx.imag], dim=0)[None]  # [1,2,H,W]
    val = ComplexPhaseGradientEntropy()(inter, context=MetricContext())
    assert math.isfinite(val)
    assert val < -1.0


# ---------------------------------------------------------------------------
# rar
# ---------------------------------------------------------------------------


def test_rar_near_zero_deviation_isotropic():
    """Anchor: isotropic image ⇒ RAR ≈ 1 ⇒ reported deviation ≈ 0."""
    obj = _disk(radius=0.4)[None, None]
    dev = ResolutionAnisotropyRatio()(obj, context=MetricContext())
    assert dev < 0.1


def test_rar_monotonic_with_pe_blur():
    """RAR deviation rises as directional (PE-axis) blur increases."""
    import torch.nn.functional as f

    obj = _disk(n=64, radius=0.4)[None, None]
    rar = ResolutionAnisotropyRatio()
    vals = []
    for sig in (0.01, 1.0, 2.0, 3.5):
        n = max(3, int(6 * sig) | 1)
        ax = torch.arange(n).float() - n // 2
        g = torch.exp(-(ax**2) / (2 * sig**2))
        g = g / g.sum()
        k = g.view(1, 1, 1, -1)  # blur along last axis (PE)
        blurred = f.conv2d(obj, k, padding=(0, n // 2))
        vals.append(rar(blurred, context=MetricContext()))
    # higher_is_better=False ⇒ anisotropy increases the deviation ⇒ rho > 0
    assert _spearman_sign(vals) >= 0.6
    assert vals[-1] > vals[0]


def test_rar_nan_on_nonfinite():
    bad = torch.full((1, 1, 16, 16), float("inf"))
    assert math.isnan(ResolutionAnisotropyRatio()(bad, context=MetricContext()))


# ---------------------------------------------------------------------------
# loose-kwargs context path (engine back-compat)
# ---------------------------------------------------------------------------


def test_metrics_accept_loose_kwargs_context():
    """resolve_context maps loose kwargs (e.g. acceleration=) to a context."""
    img = _disk(radius=0.45)[None, None]
    val = AliasingReplicaEnergy()(img, acceleration=1.0)
    assert val == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# rar — docstring regression (WS-2): score is |RAR - 1|, deviation from the
# isotropic target 1.0, higher_is_better=False.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rar_deviation_isotropic_small_anisotropic_large_and_direction():
    """Doc regression: ``rar`` reports |RAR-1| (deviation from isotropic 1.0).

    Ties the metric output to the (docstring-only) WS-2 fix: an approximately
    isotropic image scores ≈ 0 (small deviation from the target 1.0), while an
    image blurred along *one* axis only scores a LARGER strictly-positive
    deviation. Also asserts the self-declared direction ``higher_is_better is
    False`` (closer to isotropic ⇒ smaller score ⇒ ranked higher).
    """
    import torch.nn.functional as f

    torch.manual_seed(0)
    rar = ResolutionAnisotropyRatio()

    # direction contract: the metric advertises a *deviation* score, so smaller
    # is better — this is the load-bearing claim the docstring now documents.
    assert rar.higher_is_better is False

    # isotropic phantom: a disk has edges in every direction equally, so the
    # readout/PE acutance ratio ≈ 1 ⇒ reported deviation |RAR-1| ≈ 0.
    iso = _disk(n=96, radius=0.4)[None, None]
    dev_iso = rar(iso, context=MetricContext())
    assert math.isfinite(dev_iso)
    assert dev_iso == pytest.approx(0.0, abs=0.15)

    # anisotropic phantom: blur the SAME image along the last (PE) axis only.
    # This lowers the PE-axis acutance while leaving the readout axis intact, so
    # RAR departs from 1 and the reported deviation grows.
    sig = 3.0
    n = int(6 * sig) | 1
    ax = torch.arange(n).float() - n // 2
    g = torch.exp(-(ax**2) / (2 * sig**2))
    g = g / g.sum()
    k = g.view(1, 1, 1, -1)  # 1-D kernel along the width (PE) axis only
    aniso = f.conv2d(iso, k, padding=(0, n // 2))
    dev_aniso = rar(aniso, context=MetricContext())

    # the anisotropically-blurred image deviates MORE from isotropic, and the
    # deviation is a strictly-positive number (it is an absolute value).
    assert math.isfinite(dev_aniso)
    assert dev_aniso > 0.0
    assert dev_aniso > dev_iso
    # a one-axis blur this strong is unambiguously anisotropic, well clear of
    # the near-zero isotropic baseline.
    assert dev_aniso > dev_iso + 0.3


class TestGriRoutesItsThresholdThroughTheQuantileOwner:
    """Non-negotiable 16: a capability is not delivered until the production
    path calls it. A fast path added to ``core.quantile`` reaches GRI only if
    GRI actually resolves it, so both halves are observed here -- that the call
    happens, and that the fast path engages rather than silently delegating.

    The spies patch the name *as bound in* ``nr_artifact``, not the definition
    in ``core.quantile``: patching the helper's home would stay green even if
    this module went back to calling ``torch.quantile`` directly.
    """

    @staticmethod
    def _edge_image(size: int = 128) -> torch.Tensor:
        """A truncated step edge -- the signal GRI is built to measure."""
        img = torch.zeros(size, size)
        img[:, size // 2 :] = 1.0
        k = torch.fft.fftshift(torch.fft.fft2(img))
        keep = size // 4
        mask = torch.zeros_like(k, dtype=torch.bool)
        c = size // 2
        mask[c - keep : c + keep, c - keep : c + keep] = True
        return torch.fft.ifft2(torch.fft.ifftshift(k * mask)).abs()[None, None]

    def test_the_threshold_is_computed_by_robust_quantile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mriforge.core.metrics import nr_artifact

        calls: list[int] = []
        real = nr_artifact.robust_quantile

        def spy(x, q, *args, **kwargs):
            calls.append(x.numel())
            return real(x, q, *args, **kwargs)

        monkeypatch.setattr(nr_artifact, "robust_quantile", spy)
        GibbsRingingIndex()(self._edge_image())

        assert calls, "GRI computed its edge threshold without robust_quantile"
        assert calls[0] == 128 * 128

    def test_the_fast_path_actually_engages_at_this_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Observed, not assumed. GRI's gradient magnitudes are
        ``sqrt(gy**2 + gx**2 + 1e-12)`` -- strictly positive and finite, so no
        guard should decline; if one does, the fast path is dead code here."""
        from mriforge.core import quantile as quantile_mod

        engaged: list[bool] = []
        real = quantile_mod._fast_select_quantile

        def spy(flat, q):
            out = real(flat, q)
            engaged.append(out is not None)
            return out

        monkeypatch.setattr(quantile_mod, "_fast_select_quantile", spy)
        GibbsRingingIndex()(self._edge_image())

        assert engaged, "the fast path was never consulted"
        assert any(engaged), "every call declined -- the fast path is unreachable here"

    def test_the_metric_value_is_unchanged_by_the_routing(self) -> None:
        """The whole contract in one assertion: bit-identical to the value the
        direct ``torch.quantile`` spelling produced."""
        from mriforge.core.metrics import nr_artifact

        img = self._edge_image()
        via_owner = GibbsRingingIndex()(img)

        real = nr_artifact.robust_quantile
        try:
            nr_artifact.robust_quantile = lambda x, q, *a, **k: torch.quantile(x, q)
            via_torch = GibbsRingingIndex()(img)
        finally:
            nr_artifact.robust_quantile = real

        assert struct.pack("<d", via_owner) == struct.pack("<d", via_torch)
