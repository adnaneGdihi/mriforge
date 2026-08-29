"""Unit tests for §8.4 Tier 4 construct-validity scorers (CPU, deterministic)."""

from __future__ import annotations

import math

import torch

from mriforge.application.use_cases.nr_validation.cards import TierResult
from mriforge.application.use_cases.nr_validation.tier4_construct import (
    icc_2_1,
    redundancy_gate,
    score_construct_validity,
    score_invariance,
    score_test_retest,
)
from mriforge.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
)


def _make_dataset(
    metric_values: dict[str, list[float]], n: int
) -> MetricEvaluationDataset:
    """Build a minimal aligned dataset with ``n`` placeholder samples."""
    g = torch.Generator().manual_seed(0)
    samples = [
        DegradationSample(
            clean=torch.zeros(1, 4, 4),
            degraded=torch.zeros(1, 4, 4),
            degradation_family="rician",
            severity={"theta": float(i) / max(n - 1, 1)},
            content_id=f"c{i}",
            seed=int(torch.randint(0, 1_000_000, (1,), generator=g).item()),
        )
        for i in range(n)
    ]
    return MetricEvaluationDataset(samples=samples, metric_values=metric_values)


# ---------------------------------------------------------------------------
# icc_2_1 primitive
# ---------------------------------------------------------------------------
def test_icc_high_when_replicates_agree() -> None:
    torch.manual_seed(0)
    n = 30
    a = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    b = a + 0.001 * torch.randn(n, generator=torch.Generator().manual_seed(1)).double()
    matrix = torch.stack([a, b], dim=1)
    icc = icc_2_1(matrix)
    assert icc == icc  # not NaN
    assert icc > 0.9


def test_icc_low_when_replicates_independent() -> None:
    g = torch.Generator().manual_seed(7)
    n = 30
    a = torch.randn(n, generator=g).double()
    b = torch.randn(n, generator=g).double()
    matrix = torch.stack([a, b], dim=1)
    icc = icc_2_1(matrix)
    assert icc == icc
    assert icc < 0.75


def test_icc_tiny_n_returns_nan() -> None:
    matrix = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    assert math.isnan(icc_2_1(matrix))


def test_icc_drops_nonfinite_rows() -> None:
    a = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    b = a.clone()
    b[0] = float("nan")  # this row is dropped; remaining 9 still agree
    matrix = torch.stack([a, b], dim=1)
    icc = icc_2_1(matrix)
    assert icc == icc
    assert icc > 0.9


def test_icc_never_raises_on_bad_shape() -> None:
    assert math.isnan(icc_2_1(torch.zeros(5)))  # 1-D → NaN, no crash


# ---------------------------------------------------------------------------
# score_test_retest
# ---------------------------------------------------------------------------
def test_score_test_retest_pass_and_fail() -> None:
    n = 40
    base = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    noise = 0.001 * torch.randn(n, generator=torch.Generator().manual_seed(3)).double()
    reliable_a = base.tolist()
    reliable_b = (base + noise).tolist()

    g = torch.Generator().manual_seed(4)
    unreliable_a = torch.randn(n, generator=g).double().tolist()
    unreliable_b = torch.randn(n, generator=g).double().tolist()

    ds_a = _make_dataset(
        {"reliable": reliable_a, "unreliable": unreliable_a}, n
    )
    ds_b = _make_dataset(
        {"reliable": reliable_b, "unreliable": unreliable_b}, n
    )
    res = score_test_retest(ds_a, ds_b, ["reliable", "unreliable"])

    assert isinstance(res["reliable"], TierResult)
    assert res["reliable"].tier == "test_retest"
    assert res["reliable"].passed is True
    assert res["reliable"].detail["n"] == n
    assert res["unreliable"].passed is False


def test_score_test_retest_missing_metric_is_nan() -> None:
    n = 5
    ds_a = _make_dataset({"present": [0.0] * n}, n)
    ds_b = _make_dataset({"present": [0.0] * n}, n)
    res = score_test_retest(ds_a, ds_b, ["absent"])
    assert res["absent"].passed is False
    assert math.isnan(res["absent"].score)


# ---------------------------------------------------------------------------
# score_invariance (uses real registered NR metrics via eval_metric)
# ---------------------------------------------------------------------------
def test_invariance_flip_invariant_metrics_have_zero_violations() -> None:
    g = torch.Generator().manual_seed(11)
    refs = [(f"r{i}", torch.rand(1, 12, 12, generator=g)) for i in range(3)]
    # These NR metrics are computed from gradient/Laplacian magnitudes and are
    # exactly invariant to a horizontal flip.
    names = ["laplacian_variance", "tenengrad_variance"]
    res = score_invariance(refs, names, rel_tol=0.05)
    for name in names:
        assert res[name].tier == "invariance"
        assert res[name].detail["violations"] == 0
        assert res[name].detail["n"] == 3
        assert res[name].passed is True
        assert res[name].detail["max_rel_change"] <= 0.05


def test_invariance_structure_with_unknown_metric() -> None:
    g = torch.Generator().manual_seed(12)
    refs = [("r0", torch.rand(1, 8, 8, generator=g))]
    # An unregistered metric → eval_metric returns NaN on both → no opinion.
    res = score_invariance(refs, ["__definitely_not_a_metric__"], rel_tol=0.05)
    r = res["__definitely_not_a_metric__"]
    assert r.detail["n"] == 0
    assert r.passed is False  # no opinion is not a pass
    assert math.isnan(r.score)


# ---------------------------------------------------------------------------
# redundancy_gate
# ---------------------------------------------------------------------------
def test_redundancy_gate_prunes_duplicate_keeps_independent() -> None:
    n = 25
    g = torch.Generator().manual_seed(21)
    a = torch.randn(n, generator=g).double()
    dup = (a + 1e-6 * torch.randn(n, generator=g).double())  # near-perfect corr
    indep = torch.randn(n, generator=g).double()

    ds = _make_dataset(
        {"a": a.tolist(), "a_copy": dup.tolist(), "indep": indep.tolist()}, n
    )
    out = redundancy_gate(ds, ["a", "a_copy", "indep"], corr_threshold=0.95)

    assert "a" in out["kept"]
    assert "indep" in out["kept"]
    assert "a_copy" in out["pruned"]
    assert out["pruned"]["a_copy"] == "a"  # duplicates the first kept metric
    assert "a|" not in out["corr"]  # 'a' is first → never compared as duplicate


def test_redundancy_gate_absent_metric() -> None:
    n = 10
    ds = _make_dataset({"present": list(range(n))}, n)
    out = redundancy_gate(ds, ["present", "missing"], corr_threshold=0.95)
    assert "present" in out["kept"]
    assert out["pruned"]["missing"] == "__absent__"


# ---------------------------------------------------------------------------
# score_construct_validity combiner
# ---------------------------------------------------------------------------
def test_score_construct_validity_combines_icc_and_invariance() -> None:
    n = 40
    g = torch.Generator().manual_seed(31)
    base = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    noise = 0.001 * torch.randn(n, generator=g).double()

    # Use a real flip-invariant metric NAME so invariance passes; the dataset
    # columns are synthetic (ICC reads columns, invariance reads clean_volumes).
    ds_a = _make_dataset({"laplacian_variance": base.tolist()}, n)
    ds_b = _make_dataset({"laplacian_variance": (base + noise).tolist()}, n)

    refs = [(f"r{i}", torch.rand(1, 10, 10, generator=g)) for i in range(2)]
    res = score_construct_validity(
        ds_a, ds_b, refs, ["laplacian_variance"], icc_threshold=0.75, rel_tol=0.05
    )
    r = res["laplacian_variance"]
    assert r.tier == "construct_validity"
    assert r.passed is True
    assert r.detail["invariance_passed"] is True
    assert r.detail["invariance_violations"] == 0
    assert r.detail["icc_2_1"] > 0.75
    assert r.detail["n"] == n


def test_score_construct_validity_fails_on_bad_icc() -> None:
    n = 40
    g = torch.Generator().manual_seed(41)
    a = torch.randn(n, generator=g).double().tolist()
    b = torch.randn(n, generator=g).double().tolist()
    ds_a = _make_dataset({"laplacian_variance": a}, n)
    ds_b = _make_dataset({"laplacian_variance": b}, n)
    refs = [("r0", torch.rand(1, 10, 10, generator=g))]
    res = score_construct_validity(
        ds_a, ds_b, refs, ["laplacian_variance"]
    )
    # Invariance passes but ICC fails → combined fail.
    assert res["laplacian_variance"].passed is False
    assert res["laplacian_variance"].detail["invariance_passed"] is True
