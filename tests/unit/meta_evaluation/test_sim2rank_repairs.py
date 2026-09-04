"""Sim2Rank ranker-internal repairs: ADR strict-rank (§9) + robust normaliser (§18)."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.rankers.sim2rank import (
    _midranks,
    _minmax_normalise,
    _normalise,
    _rank_normalise,
    _spearman_midrank,
    compute_adr,
    compute_cdrs,
    compute_scvr,
)


def test_midranks_share_ties() -> None:
    r = _midranks(torch.tensor([5.0, 5.0, 1.0]))
    assert r.tolist() == [1.5, 1.5, 0.0]  # the two 5s share the average of ranks 1,2


def test_eps_nonsat_is_reparam_fragile() -> None:
    v = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    cubed = v**3  # strictly-increasing nonlinear reparameterisation
    a = compute_adr(v, epsilon=0.05, nonsat="epsilon")
    b = compute_adr(cubed, epsilon=0.05, nonsat="epsilon")
    # Linear trajectory saturates nothing (ADR=1); the cubed one trips the eps gate.
    assert a == pytest.approx(1.0)
    assert b < a


def test_strict_rank_is_reparam_invariant() -> None:
    v = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    cubed = v**3
    a = compute_adr(v, nonsat="strict_rank")
    b = compute_adr(cubed, nonsat="strict_rank")
    # Strict-rank non-saturation depends only on order/equality -> invariant.
    assert a == pytest.approx(b)
    assert a == pytest.approx(1.0)


def test_strict_rank_is_degenerate_which_is_why_it_is_not_the_default() -> None:
    """The L1/§9 repair is invariant but CANNOT discriminate (#241).

    It only detects *exact* ties, which real-valued data essentially never has, so a
    hard-saturating metric and a perfectly linear one both score 1.0. Measured on a
    realistic 40-metric pool it produced ONE distinct score. The default is the
    increment perplexity instead.
    """
    # Both decay with severity, so both are higher-is-better metrics (#237: the flag is
    # load-bearing now — omit it and a decaying trajectory is the WRONG direction for
    # the lower-is-better default, and correctly scores 0).
    linear = torch.linspace(1.0, 0.0, 20).double()
    saturating = torch.exp(-8.0 * torch.linspace(0.05, 1.0, 20)).double()
    assert compute_adr(
        linear, higher_is_better=True, nonsat="strict_rank"
    ) == pytest.approx(
        compute_adr(saturating, higher_is_better=True, nonsat="strict_rank")
    )
    # …whereas the default (perplexity) separates them.
    assert compute_adr(linear, higher_is_better=True) > compute_adr(
        saturating, higher_is_better=True
    )


def test_strict_rank_penalises_genuine_plateau() -> None:
    # A true flat step (tie) is the only thing the strict-rank factor penalises.
    flat = torch.tensor([0.0, 1.0, 1.0, 2.0, 3.0])
    score = compute_adr(flat, nonsat="strict_rank")
    assert score < 1.0  # one of four steps is flat -> non-sat 3/4


def test_rank_normaliser_is_outlier_robust() -> None:
    clean = _rank_normalise([1.0, 2.0, 3.0])
    outlier = _rank_normalise([1.0, 2.0, 1000.0])
    # The 1-vs-2 gap is preserved at 1/(K-1)=0.5 regardless of the outlier.
    assert clean[1] - clean[0] == pytest.approx(0.5)
    assert outlier[1] - outlier[0] == pytest.approx(0.5)


def test_minmax_collapses_under_outlier() -> None:
    from spectramr.core.metrics.meta_evaluation.rankers.sim2rank import _minmax_normalise

    g = _minmax_normalise([1.0, 2.0, 1000.0])
    # Min-max collapses the 1-vs-2 gap toward zero (range dominated by outlier).
    assert g[1] - g[0] < 0.01


def test_normalise_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        _normalise([1.0, 2.0], "bogus")


# --- H1: tie-corrected Spearman in ADR ---


def test_spearman_midrank_penalises_saturation() -> None:
    # A monotone-then-flat trajectory: the tie-blind argsort(argsort) returns 1.0;
    # the tie-corrected mid-rank must be strictly below 1 (critique H1).
    traj = torch.tensor([0.1, 0.5, 1.0, 1.0, 1.0, 1.0])
    t = torch.arange(6, dtype=torch.float64)
    assert abs(_spearman_midrank(traj, t)) < 0.99


def test_adr_below_one_on_saturated_trajectory() -> None:
    # ADR = directed-rho * non-sat; with the tie-corrected rho a saturating
    # metric can no longer reach 1.0 on the tracking factor.
    traj = torch.tensor([0.1, 0.4, 0.8, 1.0, 1.0, 1.0])
    assert compute_adr(traj, nonsat="strict_rank") < 1.0


# --- H4 / B2: +inf is the BEST sub-score (maps to the top, not the bottom) ---


def test_minmax_maps_plus_inf_to_top() -> None:
    g = _minmax_normalise([0.0, 1.0, 2.0, float("inf")])
    assert g[3] == pytest.approx(1.0)  # +inf -> top, not 0.0
    assert g[0] == pytest.approx(0.0)


def test_minmax_maps_minus_inf_and_nan_to_bottom() -> None:
    g = _minmax_normalise([0.0, 1.0, 2.0, float("-inf"), float("nan")])
    assert g[3] == pytest.approx(0.0)  # -inf -> bottom
    assert g[4] == pytest.approx(0.0)  # NaN -> bottom


def test_rank_normalise_maps_plus_inf_to_top() -> None:
    g = _rank_normalise([0.0, 1.0, 2.0, float("inf")])
    assert g[3] == pytest.approx(1.0)  # +inf ranks above every finite value


# --- H5 / B3: family variance is NOT folded into the content nuisance ---


def test_scvr_excludes_family_variance_from_nuisance() -> None:
    # Identical content (no anatomy variance) but a family- and severity-dependent
    # response. With family wrongly in the nuisance, ss_content > 0 -> finite ratio;
    # correctly, content nuisance is 0 -> +inf (the best case). (S=2, C=2, T=4)
    base = torch.tensor([0.0, 1.0, 2.0, 3.0])
    fam_a = base
    fam_b = base * 2.0  # different family response, but SAME for both contents
    tensor = torch.stack(
        [
            torch.stack([fam_a, fam_b]),
            torch.stack([fam_a, fam_b]),
        ]  # 2 identical contents
    )
    assert tensor.shape == (2, 2, 4)
    assert compute_scvr(tensor) == float("inf")


# --- M1: CDRS clean-stability is robust for a zero-centred metric ---


def test_cdrs_cm_robust_for_zero_centred_metric() -> None:
    # A signed metric centred at ~0 with small absolute dispersion must not be
    # scored maximally unstable (c_m -> 0) the way std/|mean| does (critique M1).
    _, c_m, _ = compute_cdrs([-0.01, 0.01, -0.02, 0.015, -0.005], [])
    assert c_m > 0.3  # robust scale floors the relative dispersion at ~exp(-1)
