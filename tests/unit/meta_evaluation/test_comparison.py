"""Unit tests for the real-vs-betting meta-evaluation comparison core.

Covers :func:`spearman_rank_correlation`, :func:`compare_summaries`, and
:func:`render_comparison_markdown` — in particular the two contracts the
``submit_sim2rank_matrix.sbatch`` dispatcher relies on:

* betting is *additive*, so the real and betting arms of one ``eval_mode`` must
  have identical rankings (Spearman == 1) — the comparison flags any divergence;
* a *missing* arm (the fail-loud 3-D guard wrote no summary) is reported under
  ``missing_arms`` rather than crashing the comparison.
"""

from __future__ import annotations

from mriforge.core.metrics.meta_evaluation import (
    compare_summaries,
    render_comparison_markdown,
    spearman_rank_correlation,
)


def _summary(
    *,
    eval_mode: str = "2d",
    scores: dict[str, float] | None = None,
    defensible: set[str] | None = None,
    powered: list[list[str]] | None = None,
    betting: list[list[str]] | None = None,
) -> dict:
    scores = scores or {"psnr": 1.0, "ssim": 0.0, "rmse": -1.0}
    ranks = sorted(scores, key=lambda m: scores[m], reverse=True)
    defensible = defensible or set()
    out: dict = {
        "eval_mode": eval_mode,
        "final_ranks": ranks,
        "final_scores": scores,
        "defensibility_flags": {m: (m in defensible) for m in scores},
        "n_samples": 100,
        "powered": {"cdscr": {"certified": powered or []}},
        "betting": ({"aggregate": {"certified": betting}} if betting else {}),
    }
    return out


# ---------------------------------------------------------------------------
# spearman_rank_correlation
# ---------------------------------------------------------------------------


def test_spearman_identical_is_one() -> None:
    s = {"a": 3.0, "b": 1.0, "c": 2.0}
    assert spearman_rank_correlation(s, s) == 1.0


def test_spearman_reversed_is_minus_one() -> None:
    a = {"x": 1.0, "y": 2.0, "z": 3.0}
    b = {"x": 3.0, "y": 2.0, "z": 1.0}
    assert spearman_rank_correlation(a, b) == -1.0


def test_spearman_fewer_than_two_shared_is_none() -> None:
    assert spearman_rank_correlation({"a": 1.0}, {"a": 1.0}) is None
    assert spearman_rank_correlation({"a": 1.0}, {"b": 1.0}) is None


def test_spearman_constant_vectors_degenerate() -> None:
    flat = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert spearman_rank_correlation(flat, flat) == 1.0


def test_compare_rank_identical_handles_none_rho_without_crash() -> None:
    # When two arms share < 2 metrics, rho is None. ``rank_identical`` must guard
    # the None before ``math.isclose`` (critique L2) — neither crash nor True.
    cmp = compare_summaries(
        [
            ("a", _summary(scores={"psnr": 1.0})),
            ("b", _summary(scores={"ssim": 1.0})),
        ]
    )
    pw = cmp["pairwise"][0]
    assert pw["spearman_final_scores"] is None
    assert pw["rank_identical"] is False


def test_compare_rank_identical_true_for_same_ranking() -> None:
    # Identical rankings -> rho == 1 -> rank_identical True (isclose, not ==).
    s = _summary(scores={"psnr": 1.0, "ssim": 0.0, "rmse": -1.0})
    cmp = compare_summaries([("real", s), ("betting", dict(s))])
    assert cmp["pairwise"][0]["rank_identical"] is True


def test_spearman_drops_nonfinite_scores() -> None:
    # crash→NaN contract: a metric with a non-finite score on either side must be
    # dropped before ranking. Otherwise it poisons the rank ordering and the
    # function returns a spurious ρ=1.0 ("rankings agree perfectly") that would
    # mask a real divergence in the betting-is-additive check.
    a = {"bad": float("nan"), "x": 1.0, "y": 2.0, "z": 3.0}
    b = {"bad": 0.0, "x": 3.0, "y": 2.0, "z": 1.0}  # reversed on the finite metrics
    assert spearman_rank_correlation(a, b) == -1.0


def test_spearman_nonfinite_leaves_fewer_than_two_is_none() -> None:
    # Dropping non-finite scores can leave < 2 comparable metrics -> ρ undefined.
    a = {"p": float("nan"), "q": 1.0}
    b = {"p": 0.0, "q": 1.0}
    assert spearman_rank_correlation(a, b) is None


# ---------------------------------------------------------------------------
# compare_summaries — the real vs betting pair (one eval_mode)
# ---------------------------------------------------------------------------


def test_real_vs_betting_identical_ranking() -> None:
    real = _summary(defensible={"psnr"}, powered=[["psnr", "ssim"]])
    betting = _summary(
        defensible={"psnr"},
        powered=[["psnr", "ssim"]],
        betting=[["psnr", "ssim"], ["psnr", "rmse"]],
    )
    cmp = compare_summaries([("real", real), ("betting", betting)])
    assert cmp["consensus_eval_mode"] == "2d"
    assert cmp["n_present"] == 2 and cmp["missing_arms"] == []
    pw = cmp["pairwise"][0]
    assert pw["rank_identical"] is True
    assert pw["spearman_final_scores"] == 1.0
    assert pw["defensible_jaccard"] == 1.0
    # betting-vs-powered delta is taken from the betting arm.
    bvp = cmp["betting_vs_powered"]
    assert bvp["arm"] == "betting"
    assert bvp["n_betting"] == 2 and bvp["n_powered"] == 1
    assert bvp["betting_only"] == [["psnr", "rmse"]]


def test_divergent_ranking_is_flagged() -> None:
    real = _summary(scores={"a": 2.0, "b": 1.0, "c": 0.0})
    betting = _summary(scores={"a": 0.0, "b": 1.0, "c": 2.0})  # reversed
    cmp = compare_summaries([("real", real), ("betting", betting)])
    pw = cmp["pairwise"][0]
    assert pw["rank_identical"] is False
    assert pw["spearman_final_scores"] == -1.0


def test_missing_3d_arm_reported_not_raised() -> None:
    """Both 3-D arms failed loud → None summaries → reported, no crash."""
    cmp = compare_summaries([("real", None), ("betting", None)])
    assert cmp["n_present"] == 0
    assert cmp["missing_arms"] == ["real", "betting"]
    assert cmp["pairwise"] == []
    assert cmp["betting_vs_powered"] is None


def test_one_missing_one_present() -> None:
    cmp = compare_summaries([("real", _summary()), ("betting", None)])
    assert cmp["n_present"] == 1
    assert cmp["missing_arms"] == ["betting"]
    assert cmp["pairwise"] == []  # need two present arms to compare


def test_eval_mode_mismatch_has_no_consensus() -> None:
    cmp = compare_summaries(
        [("a", _summary(eval_mode="2d")), ("b", _summary(eval_mode="3d"))]
    )
    assert cmp["consensus_eval_mode"] is None
    assert cmp["eval_modes"] == ["2d", "3d"]


# ---------------------------------------------------------------------------
# render_comparison_markdown
# ---------------------------------------------------------------------------


def test_markdown_mentions_eval_mode_and_verdict() -> None:
    cmp = compare_summaries(
        [("real", _summary()), ("betting", _summary(betting=[["psnr", "ssim"]]))]
    )
    md = render_comparison_markdown(cmp)
    assert "eval_mode=2d" in md
    assert "betting is additive" in md


def test_markdown_reports_failed_3d_guard() -> None:
    md = render_comparison_markdown(
        compare_summaries([("real", None), ("betting", None)])
    )
    assert "fail-loud guard" in md
    assert "no summary" in md
