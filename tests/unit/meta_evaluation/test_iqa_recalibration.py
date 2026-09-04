r"""Tests for in-domain IQA recalibration (SOTA plan T5).

T5 replaces the REJECTed zero-shot foundation IQA (CLIP-IQA / Q-Align, no
radiologist-anchored MRI validation) with a *monotone* recalibration map
(isotonic / Platt) from foundation / NR-battery scores to MRIQC- or
radiologist-anchored labels, accepted only if held-out Kendall-τ clears a
preregistered threshold. Corner case: the identity map recovers the failing
zero-shot metric, so a non-trivial fit is what justifies deployment.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation.iqa_recalibration import (
    RecalibrationResult,
    accept_recalibration,
    fit_recalibration,
    kendall_tau,
    recalibrate,
)


def test_kendall_tau_perfect_and_anti():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau(x, x) == pytest.approx(1.0)
    assert kendall_tau(x, -x) == pytest.approx(-1.0)


def test_isotonic_recovers_monotone_relationship():
    # labels = monotone (squared) function of scores + noise → isotonic should
    # recover a high-tau calibrated mapping.
    rng = np.random.default_rng(0)
    scores = np.linspace(0, 1, 60)
    labels = scores**2 + 0.02 * rng.standard_normal(60)
    fit = fit_recalibration(scores, labels, method="isotonic")
    cal = recalibrate(fit, scores)
    assert kendall_tau(cal, labels) > 0.9
    # monotone non-decreasing
    assert np.all(np.diff(cal) >= -1e-6)


def test_platt_linear_map():
    scores = np.linspace(0, 1, 40)
    labels = 3.0 * scores + 0.5
    fit = fit_recalibration(scores, labels, method="platt")
    cal = recalibrate(fit, scores)
    assert np.allclose(cal, labels, atol=1e-2)


def test_identity_corner_case_recovers_raw_metric():
    """Identity recalibration (fit scores→scores) leaves the metric unchanged —
    i.e. the failing zero-shot metric; a real fit must beat it."""
    scores = np.linspace(0, 1, 30)
    fit = fit_recalibration(scores, scores, method="isotonic")
    cal = recalibrate(fit, scores)
    assert np.allclose(cal, scores, atol=1e-6)


def test_acceptance_gate_on_heldout_tau():
    rng = np.random.default_rng(1)
    scores = np.linspace(0, 1, 80)
    labels = scores + 0.01 * rng.standard_normal(80)  # strongly correlated
    # interleaved split so the held-out scores lie within the calibration range
    # (a disjoint range would make isotonic extrapolate flat).
    cal, hld = scores[0::2], scores[1::2]
    cal_y, hld_y = labels[0::2], labels[1::2]
    fit = fit_recalibration(cal, cal_y, method="isotonic")
    res = accept_recalibration(fit, hld, hld_y, tau_threshold=0.5)
    assert isinstance(res, RecalibrationResult)
    assert res.accepted and res.heldout_tau > 0.5
    # uncorrelated held-out labels → reject
    res2 = accept_recalibration(fit, hld, rng.standard_normal(hld.size), tau_threshold=0.5)
    assert not res2.accepted


def test_rejects_bad_method_and_length_mismatch():
    with pytest.raises(ValueError):
        fit_recalibration(np.array([1.0, 2.0]), np.array([1.0, 2.0]), method="magic")
    with pytest.raises(ValueError):
        fit_recalibration(np.array([1.0, 2.0]), np.array([1.0]), method="platt")
