"""Sim2Rank composite ranker tests."""

from __future__ import annotations

import torch

from mriforge.core.metrics.meta_evaluation.rankers.sim2rank import (
    Sim2RankConfig,
    Sim2RankRanker,
    compute_adr,
    compute_cdrs,
    compute_scvr,
)


def test_adr_perfect_monotone() -> None:
    # Linearly DECREASING trajectory: that is a higher-is-better metric decaying with
    # severity, so higher_is_better=True is the correct declaration. (This test used to
    # pass higher_is_better=False and still score ~1.0 — only because the flag was a
    # dead parameter under |rho|. It is now load-bearing, #237.)
    traj = torch.linspace(1.0, 0.0, steps=10).double()
    assert compute_adr(traj, epsilon=0.01, higher_is_better=True) > 0.95
    # …and declaring the wrong direction now correctly scores zero.
    assert compute_adr(traj, epsilon=0.01, higher_is_better=False) == 0.0


def test_adr_constant_zero() -> None:
    traj = torch.full((10,), 0.5, dtype=torch.float64)
    assert compute_adr(traj) == 0.0


def test_adr_saturated() -> None:
    # Sharp drop then flat — saturation should dominate the score.
    traj = torch.tensor([1.0, 0.5] + [0.5] * 8, dtype=torch.float64)
    score = compute_adr(traj, epsilon=0.01)
    assert score < 0.3


def test_adr_default_nonsat_is_perplexity() -> None:
    # The production default is the increment-perplexity estimator, NOT "strict_rank"
    # (which is degenerate on real-valued data: every strictly-monotone metric scores
    # 1.0). A silent flip back to strict_rank would reintroduce the all-ties collapse
    # #241 fixed, so pin the default in both the function and the config.
    import inspect

    assert inspect.signature(compute_adr).parameters["nonsat"].default == "perplexity"
    assert Sim2RankConfig().nonsat == "perplexity"


def test_scvr_signal_dominated() -> None:
    # Build a (S, C, T) tensor where degradation explains most variance.
    S, C, T = 3, 2, 8
    severity = torch.linspace(0.0, 1.0, T).double()
    base = severity.view(1, 1, T).expand(S, C, T).clone()
    # Tiny content jitter.
    base += 0.01 * torch.randn(S, C, T, generator=torch.Generator().manual_seed(0)).double()
    f = compute_scvr(base)
    assert f > 5.0  # signal far stronger than content


def test_scvr_content_dominated() -> None:
    S, C, T = 3, 2, 8
    content = torch.randn(S, C, 1, generator=torch.Generator().manual_seed(1)).double() * 5
    base = content.expand(S, C, T).clone()
    base += 0.01 * torch.randn(S, C, T, generator=torch.Generator().manual_seed(2)).double()
    f = compute_scvr(base)
    assert f < 1.0


def test_cdrs_clean_baseline_stable() -> None:
    clean = [0.5, 0.51, 0.49, 0.50]
    # Decaying family trajectories => a higher-is-better metric (#237: the direction is
    # now load-bearing in CDRS's monotonicity term too).
    fam_traj = [torch.linspace(0.5, 0.0, 5).double() for _ in range(3)]
    cdrs, c_m, m_m = compute_cdrs(clean, fam_traj, higher_is_better=True)
    assert c_m > 0.9
    assert m_m > 0.5
    assert cdrs > 0.4


def test_cdrs_monotonicity_is_directed() -> None:
    """A metric that tracks severity BACKWARDS earns no monotonicity credit."""
    clean = [0.5, 0.51, 0.49, 0.50]
    rising = [torch.linspace(0.0, 0.5, 5).double() for _ in range(3)]
    _cdrs, _c_m, m_m = compute_cdrs(clean, rising, higher_is_better=True)
    assert m_m == 0.0


def test_sim2rank_real_metric_above_constant(small_dataset, metric_set) -> None:
    result = Sim2RankRanker().rank(metric_set, small_dataset)
    assert result.method == "Sim2Rank"
    assert "mse" in result.scores and "constant" in result.scores
    assert result.scores["mse"] > result.scores["constant"]
    assert result.scores["mse"] >= result.scores["noise"]
    assert "adr" in result.diagnostics
    assert "scvr" in result.diagnostics
    assert "cdrs" in result.diagnostics


def test_sim2rank_config_weights_respected(small_dataset, metric_set) -> None:
    custom = Sim2RankConfig(weights=(1.0, 0.0, 0.0))
    result = Sim2RankRanker(custom).rank(metric_set, small_dataset)
    diag = result.diagnostics
    assert diag["weights"]["adr"] == 1.0
    assert diag["weights"]["scvr"] == 0.0


def test_sim2rank_non_measuring_excluded_from_composite(small_dataset, metric_set) -> None:
    """A non-measuring metric (ADR ≡ 0) must score 0 and rank below every live one.

    Ranker-side of #412. ``_normalise`` (both minmax and rank) hands a tie block of
    zero-ADR metrics a *positive* normalised value, so the ``constant`` metric would
    otherwise carry a positive composite and could outrank a live metric on the SCVR
    axis. The measuring mask holds it out of the pool → composite 0, below every
    metric that actually measures. Keeps this ranking surface consistent with the
    main leaderboard.
    """
    result = Sim2RankRanker().rank(metric_set, small_dataset)
    adr = result.diagnostics["adr"]

    non_measuring = [n for n, a in adr.items() if a <= 1e-9]
    measuring = [n for n, a in adr.items() if a > 1e-9]
    assert "constant" in non_measuring, "the constant metric must be ADR≡0 here"
    assert measuring, "fixture must contain at least one live metric"

    for n in non_measuring:
        assert result.scores[n] == 0.0, f"{n} is non-measuring; composite must be 0"
    worst_live = min(result.scores[n] for n in measuring)
    best_non_measuring = max(result.scores[n] for n in non_measuring)
    assert best_non_measuring <= worst_live, (
        "a non-measuring metric outranks a live one — the #412 leak is open here"
    )


def test_masked_normalise_sinks_non_measuring() -> None:
    """Unit: non-measuring entries → 0; measuring entries ranked among themselves."""
    from mriforge.core.metrics.meta_evaluation.rankers.sim2rank import _masked_normalise

    values = [0.3, 0.2, 0.0, 0.0]
    mask = [True, True, False, False]
    for mode in ("rank", "minmax"):
        out = _masked_normalise(values, mode, mask)
        assert out[2] == 0.0 and out[3] == 0.0, f"{mode}: non-measuring not sunk"
        assert out[0] > out[1] >= 0.0, f"{mode}: live order not preserved"
        assert min(out[0], out[1]) >= max(out[2], out[3])
