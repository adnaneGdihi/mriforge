"""Tests for the L4 sim-to-real ranking-transfer certificate.

These pin the executable counterparts of the Lean theorems in
``docs/lean/Sim2Rank/Sim2Rank/Transfer.lean`` (total variation, the bounded
functional ``4·C·TV`` bound, and the ``TV < Δ/8`` ranking-transfer certificate
including its ``TV=0`` consistency reduction).
"""

from __future__ import annotations

import math

import pytest

from spectramr.core.metrics.meta_evaluation.transfer import (
    pmean_diff_bound,
    tv_distance,
    tv_distance_samples,
    tv_transfer_certificate,
)


def test_tv_distance_identical_and_disjoint() -> None:
    assert tv_distance([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert tv_distance([1.0, 0.0], [0.0, 1.0]) == 1.0  # disjoint support -> 1


def test_tv_samples_disjoint_is_one() -> None:
    # Two samples with non-overlapping ranges -> TV ~ 1.
    tv = tv_distance_samples([0.0, 0.1, 0.2], [10.0, 10.1, 10.2], n_bins=16)
    assert tv == pytest.approx(1.0, abs=1e-9)


def test_tv_samples_identical_is_zero() -> None:
    xs = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert tv_distance_samples(xs, list(xs), n_bins=16) == pytest.approx(0.0, abs=1e-9)


def test_pmean_diff_bound_is_four_c_tv() -> None:
    # Lean pmean_diff_le_tv: |tau(P) - tau(Q)| <= 4*C*TV.
    assert pmean_diff_bound(0.1, kernel_bound=1.0) == pytest.approx(0.4)
    assert pmean_diff_bound(0.1, kernel_bound=2.0) == pytest.approx(0.8)


def test_certificate_fires_within_budget() -> None:
    # gap 0.8 -> budget 0.1; TV 0.05 < 0.1 -> transfers.
    certs = tv_transfer_certificate({"a": 0.9, "b": 0.1}, tv=0.05)
    assert len(certs) == 1
    c = certs[0]
    assert c.better == "a" and c.worse == "b"
    assert c.sim_gap == pytest.approx(0.8)
    assert c.budget == pytest.approx(0.1)
    assert c.transfers


def test_certificate_withholds_outside_budget() -> None:
    # TV 0.2 > budget 0.1 -> does not transfer.
    certs = tv_transfer_certificate({"a": 0.9, "b": 0.1}, tv=0.2)
    assert not certs[0].transfers


def test_certificate_zero_tv_recovers_sim_ranking() -> None:
    # Lean ranking_transfer_recovers_sim: at TV=0 every strictly-ordered pair transfers.
    scores = {"a": 0.9, "b": 0.5, "c": 0.1}
    certs = tv_transfer_certificate(scores, tv=0.0)
    assert len(certs) == 3  # C(3, 2)
    assert all(c.transfers for c in certs)
    # direction always points from higher to lower simulated score.
    for c in certs:
        assert scores[c.better] >= scores[c.worse]


def test_certificate_direction_is_orientation_invariant() -> None:
    # Whichever order the dict lists the pair, the better metric is the higher score.
    certs = tv_transfer_certificate({"low": 0.1, "high": 0.9}, tv=0.0)
    assert certs[0].better == "high"


def test_negative_tv_rejected() -> None:
    with pytest.raises(ValueError):
        tv_transfer_certificate({"a": 1.0, "b": 0.0}, tv=-0.1)
    with pytest.raises(ValueError):
        pmean_diff_bound(-0.1)


# --- crash->NaN contract: non-finite metric values must be dropped, not crash ---
# simulator.precompute_metric_values / _build_real_metric_values record a metric
# that crashes on a sample as NaN; every downstream consumer must mask non-finite
# values before binning (int(NaN) raises "cannot convert float NaN to integer").


def test_tv_samples_filters_nan_in_one_sample() -> None:
    # A NaN in the simulated sample must be dropped, leaving the finite TV intact.
    clean = tv_distance_samples([0.0, 0.1, 0.2], [0.0, 0.1, 0.2], n_bins=16)
    with_nan = tv_distance_samples(
        [0.0, 0.1, 0.2, float("nan")], [0.0, 0.1, 0.2], n_bins=16
    )
    assert with_nan == pytest.approx(clean, abs=1e-12)
    assert with_nan == pytest.approx(0.0, abs=1e-9)


def test_tv_samples_filters_nan_and_inf_in_both_samples() -> None:
    # NaN/±inf in either sample are dropped; the disjoint finite supports give TV ~ 1.
    tv = tv_distance_samples(
        [0.0, 0.1, float("nan"), float("-inf")],
        [10.0, 10.1, float("inf"), float("nan")],
        n_bins=16,
    )
    assert tv == pytest.approx(1.0, abs=1e-9)


def test_tv_samples_all_nonfinite_raises_clearly() -> None:
    # A metric that crashed on the WHOLE cohort has no estimable TV -> raise.
    with pytest.raises(ValueError, match="finite"):
        tv_distance_samples([float("nan"), float("nan")], [0.0, 0.1], n_bins=8)
    with pytest.raises(ValueError, match="finite"):
        tv_distance_samples([0.0, 0.1], [float("inf"), float("-inf")], n_bins=8)


def test_tv_samples_empty_still_rejected() -> None:
    # The pre-existing empty-input contract is unchanged.
    with pytest.raises(ValueError, match="non-empty"):
        tv_distance_samples([], [0.0, 0.1], n_bins=8)


def test_tv_samples_no_nan_in_histogram_path() -> None:
    # Regression guard: with a NaN present, lo/hi must be computed on finite values
    # only (raw min/max with NaN is order-dependent and would poison the bin width).
    tv = tv_distance_samples(
        [float("nan"), 5.0, 5.0, 5.0], [5.0, 5.0, 5.0], n_bins=8
    )
    assert math.isfinite(tv)
    assert tv == pytest.approx(0.0, abs=1e-9)


# --- A5: finite-sample bias of the plug-in histogram TV ---


def test_tv_debias_removes_same_law_upward_bias() -> None:
    # Two INDEPENDENT samples from the SAME law have a spurious raw TV > 0 (the
    # plug-in histogram bias). Debiasing drives it ~0 (critique A5).
    import random

    rng = random.Random(0)
    a = [rng.random() for _ in range(200)]
    b = [rng.random() for _ in range(200)]
    raw = tv_distance_samples(a, b, n_bins=32)
    debiased = tv_distance_samples(a, b, n_bins=32, debias=True)
    assert raw > 0.1  # the upward bias is real
    assert debiased < raw
    assert debiased == pytest.approx(0.0, abs=0.05)


def test_tv_adaptive_binning_bias_shrinks_with_n() -> None:
    # With adaptive (n-aware) binning the same-law bias shrinks as n grows.
    import random

    rng = random.Random(1)
    small = tv_distance_samples(
        [rng.random() for _ in range(40)], [rng.random() for _ in range(40)],
        n_bins=None,
    )
    big = tv_distance_samples(
        [rng.random() for _ in range(1000)], [rng.random() for _ in range(1000)],
        n_bins=None,
    )
    assert big < small


def test_tv_debias_keeps_disjoint_laws_separated() -> None:
    # Debiasing must NOT collapse genuinely different laws: disjoint supports stay
    # high (the pooled-null bias is small when the samples truly differ).
    a = [0.0, 0.1, 0.2] * 30
    b = [10.0, 10.1, 10.2] * 30
    assert tv_distance_samples(a, b, n_bins=None, debias=True) > 0.8


# --- joint-vs-marginal TV gap: the joint estimator catches dependence shifts ---


def test_joint_tv_catches_dependence_shift_marginals_miss() -> None:
    import numpy as np

    from spectramr.core.metrics.meta_evaluation.transfer import (
        tv_distance_joint_samples,
    )

    rng = np.random.default_rng(0)
    n = 300
    # IDENTICAL marginals, OPPOSITE dependence: sim's two metrics are correlated,
    # real's are anti-correlated. Every marginal is ~N(0,1) in both cohorts, so the
    # marginal TV is ~0 — but the JOINT laws differ, which the joint estimator sees.
    a = rng.standard_normal(n)
    sim = np.stack([a, a + 0.1 * rng.standard_normal(n)], axis=1)
    b = rng.standard_normal(n)
    real = np.stack([b, -b + 0.1 * rng.standard_normal(n)], axis=1)
    m0 = tv_distance_samples(sim[:, 0].tolist(), real[:, 0].tolist(), debias=True)
    m1 = tv_distance_samples(sim[:, 1].tolist(), real[:, 1].tolist(), debias=True)
    joint = tv_distance_joint_samples(sim.tolist(), real.tolist())
    assert m0 < 0.2 and m1 < 0.2  # marginals miss it
    assert joint > 0.6  # the joint catches the dependence-structure shift


def test_joint_tv_zero_for_same_law_high_for_shift() -> None:
    import numpy as np

    from spectramr.core.metrics.meta_evaluation.transfer import (
        tv_distance_joint_samples,
    )

    rng = np.random.default_rng(1)
    sim = rng.standard_normal((300, 3))
    same = rng.standard_normal((300, 3))
    shifted = rng.standard_normal((300, 3)) + 3.0
    assert tv_distance_joint_samples(sim.tolist(), same.tolist()) < 0.2
    assert tv_distance_joint_samples(sim.tolist(), shifted.tolist()) > 0.8


def test_joint_tv_degenerate_inputs_return_zero() -> None:
    from spectramr.core.metrics.meta_evaluation.transfer import (
        tv_distance_joint_samples,
    )

    assert tv_distance_joint_samples([[1.0]], [[2.0]]) == 0.0  # too few samples
    assert tv_distance_joint_samples([], []) == 0.0
