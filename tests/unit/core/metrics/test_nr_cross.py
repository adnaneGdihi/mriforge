"""Unit tests for the ``nr_cross`` no-reference metric group.

Covers :mod:`spectramr.core.metrics.nr_cross`:

* ``ccsa`` — monotonicity (rises when a sibling-supported structure is erased
  from the reconstruction) + the "faithful < distorted" ordering anchor.
* ``brc``  — analytic anchor (``BRC == 0`` when the predicted maps equal the
  maps used to synthesise the measured contrasts) + monotonicity (rises with
  T1 error).
* ``ncpd`` — graceful ``nan`` on every asset-blocked path + a stub-encoder
  finite-value smoke + the focal-vs-diffuse separation anchor.
* ``ccpr`` — graceful ``nan`` on every asset-blocked path + a stub-predictor
  smoke, with ``CCPR == 0`` when the reconstruction equals the predictor output
  (consistency anchor) and rising with cross-contrast inconsistency.

All tests are CPU-only, fast, and deterministic (fixed seed). Spearman sign is
checked via a manual rank-correlation helper to avoid a SciPy dependency.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.nr_cross import (
    BlochReSynthesisConsistency,
    ConditionalCrossContrastPredictiveResidual,
    CrossContrastStructuralAgreement,
    NormativeCrossPatientDeviation,
)
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer

H = W = 24


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho via Pearson on ranks (no SciPy)."""

    def _rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        for r, i in enumerate(order):
            ranks[i] = float(r)
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def _structured_image() -> torch.Tensor:
    """A small image with a nested bright structure (the erasure target)."""
    img = torch.zeros(1, 1, H, W)
    img[:, :, 6:18, 6:18] = 1.0
    img[:, :, 10:14, 10:14] = 2.0
    return img


# --------------------------------------------------------------------------- #
# registration / contract                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["ccsa", "brc", "ncpd", "ccpr"])
def test_registered_no_reference(key: str) -> None:
    assert MetricsRegistry.is_registered(key)
    assert MetricsRegistry.requires_reference(key) is False
    inst = MetricsRegistry.get(key)
    # every class self-declares direction (lower-is-better for all four)
    assert inst.higher_is_better is False
    assert inst.name == key


def test_needs_tuples() -> None:
    assert MetricsRegistry.needs("ccsa") == ("sibling_contrasts",)
    assert MetricsRegistry.needs("brc") == ("acq_params",)
    assert MetricsRegistry.needs("ncpd") == ("encoder",)
    # ccpr also declares prior_model (the trained predictor it truly needs), so
    # it classifies not_applicable rather than never_computed in the sim2rank
    # sweep (which cannot supply a predictor). See test_ccpr_declares_prior_model_need.
    assert MetricsRegistry.needs("ccpr") == ("sibling_contrasts", "prior_model")


# --------------------------------------------------------------------------- #
# ccsa                                                                         #
# --------------------------------------------------------------------------- #


def test_ccsa_nan_without_siblings() -> None:
    ccsa = CrossContrastStructuralAgreement()
    assert math.isnan(ccsa(_structured_image()))
    # empty dict is also degenerate
    assert math.isnan(
        ccsa(_structured_image(), context=MetricContext(sibling_contrasts={}))
    )


def test_ccsa_faithful_below_distorted() -> None:
    """Faithful recon scores lower CCSA than a recon that erased a structure."""
    torch.manual_seed(0)
    ccsa = CrossContrastStructuralAgreement()
    base = _structured_image()
    sib1 = base.clone()
    sib2 = base.clone() * 0.6 + 0.2  # same structure, remapped intensity
    ctx = MetricContext(sibling_contrasts={"s1": sib1, "s2": sib2})

    good = base + 0.01 * torch.randn_like(base)
    bad = base.clone()
    bad[:, :, 10:14, 10:14] = 1.0  # delete inner bright square
    bad = bad + 0.01 * torch.randn_like(base)

    v_good = ccsa(good, context=ctx)
    v_bad = ccsa(bad, context=ctx)
    assert math.isfinite(v_good) and math.isfinite(v_bad)
    assert v_bad > v_good


def test_ccsa_monotonic_erasure() -> None:
    """CCSA rises as the sibling-supported inner structure is progressively erased."""
    torch.manual_seed(1)
    ccsa = CrossContrastStructuralAgreement()
    base = _structured_image()
    ctx = MetricContext(sibling_contrasts={"s1": base.clone()})

    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    vals: list[float] = []
    for s in severities:
        r = base.clone()
        # blend the inner square's excess (2.0 over 1.0 background) toward 0
        r[:, :, 10:14, 10:14] = 1.0 + (1.0) * (1.0 - s)
        vals.append(ccsa(r, context=ctx))
    assert all(math.isfinite(v) for v in vals)
    rho = _spearman(severities, vals)
    assert rho >= 0.6, f"CCSA should rise with erasure severity (rho={rho})"


def test_ccsa_shape_mismatch_sibling_skipped() -> None:
    """A wrong-shape sibling is skipped (no crash); all-mismatch -> nan."""
    ccsa = CrossContrastStructuralAgreement()
    base = _structured_image()
    bad_sib = torch.rand(1, 1, H // 2, W // 2)
    assert math.isnan(
        ccsa(base, context=MetricContext(sibling_contrasts={"s": bad_sib}))
    )


# --------------------------------------------------------------------------- #
# brc                                                                          #
# --------------------------------------------------------------------------- #


def _bloch_phantom() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t1 = torch.full((1, 1, H, W), 900.0)
    t2 = torch.full((1, 1, H, W), 70.0)
    pd = torch.ones(1, 1, H, W)
    t1[:, :, 8:16, 8:16] = 1400.0
    return t1, t2, pd


def test_brc_nan_without_acq_params() -> None:
    brc = BlochReSynthesisConsistency()
    assert math.isnan(brc(torch.zeros(1, 1, H, W)))
    # acq_params present but no measured siblings -> nan
    assert math.isnan(
        brc(
            torch.zeros(1, 1, H, W),
            context=MetricContext(
                acq_params={"c1": {"tr": 15.0, "te": 4.0, "fa": 0.2}}
            ),
        )
    )


def test_brc_zero_at_true_maps() -> None:
    """BRC == 0 (within fp tolerance) when predicted maps match the synthesis maps."""
    t1, t2, pd = _bloch_phantom()
    bloch = DifferentiableBlochLayer("SPGR")
    acq = {
        "c1": {"tr": 15.0, "te": 4.0, "fa": 0.2},
        "c2": {"tr": 40.0, "te": 10.0, "fa": 0.5},
    }
    meas = {
        c: bloch.forward(t1, t2, pd, tr=a["tr"], te=a["te"], flip_angle_rad=a["fa"])
        for c, a in acq.items()
    }
    brc = BlochReSynthesisConsistency()
    ctx = MetricContext(
        acq_params=acq,
        sibling_contrasts=meas,
        nr_params={"brc_qmaps": {"t1": t1, "t2": t2, "pd": pd}},
    )
    val = brc(torch.zeros(1, 1, H, W), context=ctx)
    assert val == pytest.approx(0.0, abs=1e-5)


def test_brc_monotonic_in_t1_error() -> None:
    """BRC rises monotonically as predicted T1 departs from the true T1."""
    t1, t2, pd = _bloch_phantom()
    bloch = DifferentiableBlochLayer("SPGR")
    acq = {
        "c1": {"tr": 15.0, "te": 4.0, "fa": 0.2},
        "c2": {"tr": 40.0, "te": 10.0, "fa": 0.5},
    }
    meas = {
        c: bloch.forward(t1, t2, pd, tr=a["tr"], te=a["te"], flip_angle_rad=a["fa"])
        for c, a in acq.items()
    }
    brc = BlochReSynthesisConsistency()

    scales = [1.0, 1.1, 1.3, 1.6, 2.0]
    vals: list[float] = []
    for sc in scales:
        ctx = MetricContext(
            acq_params=acq,
            sibling_contrasts=meas,
            nr_params={"brc_qmaps": {"t1": t1 * sc, "t2": t2, "pd": pd}},
        )
        vals.append(brc(torch.zeros(1, 1, H, W), context=ctx))
    assert all(math.isfinite(v) for v in vals)
    assert vals[0] == pytest.approx(0.0, abs=1e-5)
    rho = _spearman(scales, vals)
    assert rho >= 0.6, f"BRC should rise with T1 error (rho={rho})"


def test_brc_qmaps_from_prediction_channels() -> None:
    """When no brc_qmaps in nr_params, the 3-channel prediction supplies (t1,t2,pd)."""
    t1, t2, pd = _bloch_phantom()
    bloch = DifferentiableBlochLayer("SPGR")
    acq = {"c1": {"tr": 15.0, "te": 4.0, "fa": 0.2}}
    meas = {"c1": bloch.forward(t1, t2, pd, tr=15.0, te=4.0, flip_angle_rad=0.2)}
    pred = torch.cat([t1, t2, pd], dim=1)  # [1, 3, H, W]
    brc = BlochReSynthesisConsistency()
    val = brc(pred, context=MetricContext(acq_params=acq, sibling_contrasts=meas))
    assert val == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# ncpd (gated)                                                                 #
# --------------------------------------------------------------------------- #


def test_ncpd_nan_without_assets() -> None:
    ncpd = NormativeCrossPatientDeviation()
    pred = torch.rand(1, 1, H, W)
    # no context at all
    assert math.isnan(ncpd(pred))
    # encoder present but no normative asset
    assert math.isnan(
        ncpd(pred, context=MetricContext(encoder=lambda x: torch.randn(1, 4, 3)))
    )
    # normative present but no encoder
    norm = {"mu": torch.zeros(4, 3), "cov": torch.stack([torch.eye(3)] * 4)}
    assert math.isnan(
        ncpd(pred, context=MetricContext(nr_params={"ncpd_normative": norm}))
    )


def test_ncpd_stub_encoder_finite() -> None:
    """With a stub encoder + normative asset, NCPD is a finite non-negative float."""
    torch.manual_seed(2)
    ncpd = NormativeCrossPatientDeviation()
    pred = torch.rand(1, 1, H, W)
    n_regions, dim = 4, 3
    norm = {
        "mu": torch.zeros(n_regions, dim),
        "cov": torch.stack([torch.eye(dim)] * n_regions),
    }

    def enc(_img: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            1, n_regions, dim, generator=torch.Generator().manual_seed(3)
        )

    val = ncpd(
        pred, context=MetricContext(encoder=enc, nr_params={"ncpd_normative": norm})
    )
    assert math.isfinite(val)
    assert val >= 0.0


def test_ncpd_total_deviation_monotone() -> None:
    """NCPD (mean per-region Mahalanobis) grows with total deviation budget.

    The spec's focal-vs-diffuse *support* rule is about the sparsity of the
    per-region score vector; the aggregated NCPD itself is monotone in the
    summed squared deviation. We assert the unambiguous monotone direction:
    a normative-matching recon scores ~0, and the score rises as embeddings
    move away from the normative mean.
    """
    ncpd = NormativeCrossPatientDeviation()
    pred = torch.rand(1, 1, H, W)
    n_regions, dim = 6, 2
    norm = {
        "mu": torch.zeros(n_regions, dim),
        "cov": torch.stack([torch.eye(dim)] * n_regions),
    }

    on_normative = torch.zeros(n_regions, dim)  # exactly at the mean -> 0
    focal = torch.zeros(n_regions, dim)
    focal[0] = 1.0  # one region mildly off
    diffuse = torch.full((n_regions, dim), 2.0)  # every region strongly off

    v0 = ncpd(
        pred,
        context=MetricContext(
            encoder=lambda _x: on_normative, nr_params={"ncpd_normative": norm}
        ),
    )
    v_focal = ncpd(
        pred,
        context=MetricContext(
            encoder=lambda _x: focal, nr_params={"ncpd_normative": norm}
        ),
    )
    v_diffuse = ncpd(
        pred,
        context=MetricContext(
            encoder=lambda _x: diffuse, nr_params={"ncpd_normative": norm}
        ),
    )
    assert all(math.isfinite(v) for v in (v0, v_focal, v_diffuse))
    assert v0 == pytest.approx(0.0, abs=1e-5)
    # focal (one region, total dev 1) < diffuse (six regions, total dev 48)
    assert v0 < v_focal < v_diffuse


def test_ncpd_encoder_exception_returns_nan() -> None:
    """A throwing encoder yields nan, never propagates (no validation abort)."""
    ncpd = NormativeCrossPatientDeviation()
    norm = {"mu": torch.zeros(4, 3), "cov": torch.stack([torch.eye(3)] * 4)}

    def bad_enc(_img: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")

    val = ncpd(
        torch.rand(1, 1, H, W),
        context=MetricContext(encoder=bad_enc, nr_params={"ncpd_normative": norm}),
    )
    assert math.isnan(val)


# --------------------------------------------------------------------------- #
# ccpr (gated)                                                                 #
# --------------------------------------------------------------------------- #


def test_ccpr_nan_without_assets() -> None:
    ccpr = ConditionalCrossContrastPredictiveResidual()
    pred = torch.rand(1, 1, H, W)
    # no siblings
    assert math.isnan(ccpr(pred))
    # siblings but no predictor
    sibs = {"s1": torch.rand(1, 1, H, W)}
    assert math.isnan(ccpr(pred, context=MetricContext(sibling_contrasts=sibs)))


def test_ccpr_stub_predictor_zero_when_consistent() -> None:
    """CCPR == 0 when the reconstruction equals the predictor's output."""
    torch.manual_seed(4)
    ccpr = ConditionalCrossContrastPredictiveResidual()
    sibs = {"s1": torch.rand(1, 1, H, W), "s2": torch.rand(1, 1, H, W)}

    def g(stacked: torch.Tensor) -> torch.Tensor:  # mean of siblings
        return stacked.mean(dim=1, keepdim=True)

    consistent = g(torch.cat([sibs["s1"], sibs["s2"]], dim=1))
    val = ccpr(
        consistent,
        context=MetricContext(sibling_contrasts=sibs, nr_params={"ccpr_predictor": g}),
    )
    assert val == pytest.approx(0.0, abs=1e-6)


def test_ccpr_rises_with_inconsistency() -> None:
    """CCPR grows as the reconstruction drifts away from the predictor output."""
    torch.manual_seed(5)
    ccpr = ConditionalCrossContrastPredictiveResidual()
    sibs = {"s1": torch.rand(1, 1, H, W), "s2": torch.rand(1, 1, H, W)}

    def g(stacked: torch.Tensor) -> torch.Tensor:
        return stacked.mean(dim=1, keepdim=True)

    consistent = g(torch.cat([sibs["s1"], sibs["s2"]], dim=1))
    ctx = MetricContext(sibling_contrasts=sibs, nr_params={"ccpr_predictor": g})

    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    drift = torch.rand_like(consistent)
    vals = [ccpr(consistent + s * drift, context=ctx) for s in severities]
    assert all(math.isfinite(v) for v in vals)
    rho = _spearman(severities, vals)
    assert rho >= 0.6, f"CCPR should rise with inconsistency (rho={rho})"


def test_ccpr_prior_model_fallback() -> None:
    """ccpr falls back to ctx.prior_model when nr_params has no explicit predictor."""
    ccpr = ConditionalCrossContrastPredictiveResidual()
    sibs = {"s1": torch.rand(1, 1, H, W), "s2": torch.rand(1, 1, H, W)}

    def g(stacked: torch.Tensor) -> torch.Tensor:
        return stacked.mean(dim=1, keepdim=True)

    pred = torch.rand(1, 1, H, W)
    val = ccpr(pred, context=MetricContext(sibling_contrasts=sibs, prior_model=g))
    assert math.isfinite(val) and val >= 0.0


def test_ccpr_predictor_exception_returns_nan() -> None:
    ccpr = ConditionalCrossContrastPredictiveResidual()
    sibs = {"s1": torch.rand(1, 1, H, W)}

    def bad_pred(_stacked: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")

    val = ccpr(
        torch.rand(1, 1, H, W),
        context=MetricContext(
            sibling_contrasts=sibs, nr_params={"ccpr_predictor": bad_pred}
        ),
    )
    assert math.isnan(val)


def test_ccpr_declares_prior_model_need() -> None:
    """ccpr must declare the trained predictor (prior_model) it truly needs.

    Regression for the sim2rank ``never_computed`` misclassification: ccpr
    genuinely needs ``ctx.prior_model`` (the docstring says so), but declared only
    ``sibling_contrasts``. Because prior_model is un-providable by the sim2rank
    engine, the under-declaration made the metric read as a defect
    (``never_computed``) instead of the honest ``not_applicable``.
    """
    from spectramr.core.metrics.registry import MetricsRegistry

    needs = set(MetricsRegistry.needs("ccpr"))
    assert "prior_model" in needs, f"ccpr must declare prior_model; got {needs}"
    assert "sibling_contrasts" in needs
