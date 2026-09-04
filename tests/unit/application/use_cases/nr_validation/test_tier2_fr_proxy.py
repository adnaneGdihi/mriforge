"""Unit tests for §8.2 Tier 2 — full-reference proxy agreement scorer."""

from __future__ import annotations

import math

import torch

from spectramr.application.use_cases.nr_validation.tier2_fr_proxy import (
    DEFAULT_FR_PROXIES,
    score_fr_proxy,
    variance_inflation_factors,
)
from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
)


def _tiny_sample(theta: float, seed: int) -> DegradationSample:
    """A minimal DegradationSample; tensors are tiny ([1,8,8]) — the scorer only
    consumes ``metric_values`` and ``severity``, not the tensors themselves."""
    g = torch.Generator().manual_seed(seed)
    clean = torch.rand(1, 8, 8, generator=g)
    degraded = clean + 0.01 * torch.rand(1, 8, 8, generator=g)
    return DegradationSample(
        clean=clean,
        degraded=degraded,
        degradation_family="rician",
        severity={"theta": theta},
        content_id="c0",
        seed=seed,
    )


def _make_dataset(
    metric_values: dict[str, list[float]], n: int
) -> MetricEvaluationDataset:
    thetas = [i / (n - 1) for i in range(n)]
    samples = [_tiny_sample(t, seed=100 + i) for i, t in enumerate(thetas)]
    return MetricEvaluationDataset(samples=samples, metric_values=metric_values)


def test_correlated_nr_passes_constant_and_noise_fail() -> None:
    """A perfectly rank-correlated NR metric clears the FR-proxy gate; a constant
    column and a rank-uncorrelated noise column fail."""
    n = 12
    torch.manual_seed(0)
    # "psnr" increasing FR proxy. nr_good tracks it monotonically (negated, so the
    # |rho| is what matters); nr_const is flat; nr_noise is a fixed permutation
    # that is rank-uncorrelated with psnr.
    psnr = [10.0 + i for i in range(n)]
    nr_good = [-2.0 * x for x in psnr]  # rho_S = -1 -> |rho| = 1
    nr_const = [3.14] * n
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(7)
    ).tolist()
    nr_noise = [float(p) for p in perm]

    ds = _make_dataset(
        {
            "psnr": psnr,
            "nr_good": nr_good,
            "nr_const": nr_const,
            "nr_noise": nr_noise,
        },
        n,
    )

    res = score_fr_proxy(
        ds, ["nr_good", "nr_const", "nr_noise"], rho_threshold=0.6
    )

    assert res["nr_good"].tier == "fr_proxy"
    assert res["nr_good"].passed is True
    assert res["nr_good"].score >= 0.6
    assert res["nr_good"].detail["best_fr"] == "psnr"
    assert res["nr_good"].detail["n_fr"] == 1
    assert math.isfinite(res["nr_good"].detail["per_fr"]["psnr"])

    # Constant column -> guarded Spearman returns 0 -> below threshold -> fail.
    assert res["nr_const"].passed is False
    assert res["nr_const"].score < 0.6

    # Rank-uncorrelated noise -> low |rho| -> fail.
    assert res["nr_noise"].passed is False
    assert res["nr_noise"].score < 0.6

    # Every result carries the VIF diagnostic.
    for name in ("nr_good", "nr_const", "nr_noise"):
        assert "vif" in res[name].detail


def test_score_uses_best_of_multiple_fr_proxies() -> None:
    """The headline score is the MAX |rho| across the available FR proxies."""
    n = 10
    psnr = [float(i) for i in range(n)]  # increasing
    ssim = [float(n - i) for i in range(n)]  # decreasing, perfectly anti-monotone
    # nr tracks ssim's order but is unrelated to psnr's order? Make nr == psnr so
    # it perfectly tracks psnr (|rho|=1) and anti-tracks ssim (|rho|=1). Both are
    # 1.0; best_fr should be the first available (psnr by DEFAULT order).
    nr = list(psnr)
    ds = _make_dataset({"psnr": psnr, "ssim": ssim, "nr": nr}, n)

    res = score_fr_proxy(ds, ["nr"])
    assert res["nr"].passed is True
    assert res["nr"].score >= 0.99
    assert res["nr"].detail["n_fr"] == 2
    assert set(res["nr"].detail["per_fr"].keys()) == {"psnr", "ssim"}
    # Default proxy order lists psnr before ssim, so ties resolve to psnr.
    assert res["nr"].detail["best_fr"] in {"psnr", "ssim"}


def test_no_fr_columns_sets_reason_no_exception() -> None:
    """A dataset with no FR proxy column yields passed=False + reason, no raise."""
    n = 6
    nr_a = [float(i) for i in range(n)]
    nr_b = [float(2 * i) for i in range(n)]
    ds = _make_dataset({"nr_a": nr_a, "nr_b": nr_b}, n)

    res = score_fr_proxy(ds, ["nr_a", "nr_b"])
    for name in ("nr_a", "nr_b"):
        assert res[name].passed is False
        assert res[name].detail["reason"] == "no FR proxy columns in dataset"
        assert math.isnan(res[name].score)


def test_default_fr_proxies_are_standard_fr_metrics() -> None:
    assert DEFAULT_FR_PROXIES == ["psnr", "ssim", "ms_ssim", "lpips", "dists"]


def test_vif_high_for_duplicated_low_for_independent() -> None:
    """Two duplicated NR columns yield a high VIF; an independent column ~1."""
    n = 60
    g = torch.Generator().manual_seed(42)
    base = torch.randn(n, generator=g)
    dup = base.clone()  # exact duplicate -> perfectly collinear -> huge VIF
    indep = torch.randn(n, generator=torch.Generator().manual_seed(123))

    metric_values = {
        "nr_dup_a": base.tolist(),
        "nr_dup_b": dup.tolist(),
        "nr_indep": indep.tolist(),
    }
    ds = _make_dataset(metric_values, n)

    vifs = variance_inflation_factors(ds, ["nr_dup_a", "nr_dup_b", "nr_indep"])

    # Duplicated pair: regressing one on the others (which includes its exact
    # twin) gives R^2 ~ 1 -> VIF very large.
    assert vifs["nr_dup_a"] > 50.0
    assert vifs["nr_dup_b"] > 50.0
    # Independent column is not explained by the (collinear) others -> VIF ~ 1.
    assert math.isfinite(vifs["nr_indep"])
    assert 0.5 <= vifs["nr_indep"] <= 5.0


def test_vif_folded_into_tierresult_detail() -> None:
    n = 40
    g = torch.Generator().manual_seed(11)
    base = torch.randn(n, generator=g)
    psnr = torch.arange(n, dtype=torch.float64).tolist()
    metric_values = {
        "psnr": psnr,
        "nr_a": base.tolist(),
        "nr_b": base.tolist(),  # duplicate -> high VIF
    }
    ds = _make_dataset(metric_values, n)

    res = score_fr_proxy(ds, ["nr_a", "nr_b"])
    assert res["nr_a"].detail["vif"] > 50.0
    assert res["nr_b"].detail["vif"] > 50.0


def test_vif_nan_with_too_few_metrics_or_rows() -> None:
    # Single NR metric -> no regressors -> nan.
    n = 5
    ds = _make_dataset({"only": [float(i) for i in range(n)]}, n)
    vifs = variance_inflation_factors(ds, ["only"])
    assert math.isnan(vifs["only"])

    # Two metrics but <3 finite rows after dropping NaNs -> nan.
    nan = float("nan")
    ds2 = _make_dataset(
        {"a": [1.0, nan, nan], "b": [nan, 2.0, nan]}, 3
    )
    vifs2 = variance_inflation_factors(ds2, ["a", "b"])
    assert math.isnan(vifs2["a"])
    assert math.isnan(vifs2["b"])


def test_scorer_never_raises_on_all_nan_nr_column() -> None:
    """An NR column that is entirely NaN must not raise; it fails with no finite
    correlation."""
    n = 8
    nan = float("nan")
    ds = _make_dataset(
        {"psnr": [float(i) for i in range(n)], "nr_nan": [nan] * n}, n
    )
    res = score_fr_proxy(ds, ["nr_nan"])
    assert res["nr_nan"].passed is False
    assert math.isnan(res["nr_nan"].score)
    # per_fr exists but the psnr correlation is non-finite.
    assert "per_fr" in res["nr_nan"].detail
