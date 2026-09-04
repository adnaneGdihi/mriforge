"""Tests for the §8.7 ``nr_quality_index`` aggregator meta-metric.

CPU-canonical and deterministic: every synthetic dataset is built with explicit
seeds. The tests assert the fit recovers a strong linear severity signal,
exercise the sklearn-absent closed-form fallback, and verify the registered
metric round-trips (feature dict ⇒ float; no dict ⇒ NaN; idempotent re-register).
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.application.use_cases.nr_validation.aggregator import (
    NRQualityIndexModel,
    fit_nr_quality_index,
    register_nr_quality_index,
)
from spectramr.core.metrics import get_metric
from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
)
from spectramr.core.metrics.registry import MetricsRegistry


@pytest.fixture(autouse=True)
def _restore_metric_registry():
    """Snapshot and restore the global ``MetricsRegistry`` around each test.

    ``register_nr_quality_index`` mutates the process-wide registry singleton
    (adding the ``nr_quality_index`` metric plus its capability metadata).
    Without restoration that registration leaks into every later test in the
    session and breaks registry-coverage assertions elsewhere (e.g.
    ``test_sim2rank_scoring.TestMetricsList.test_covers_registry``). Restore the
    backing dicts in place so any external reference to a class-var stays valid.
    """
    attrs = ("_metrics", "_aliases", "_needs", "_requires_context", "_requires_reference")
    snapshot = {attr: dict(getattr(MetricsRegistry, attr)) for attr in attrs}
    try:
        yield
    finally:
        for attr, saved in snapshot.items():
            live = getattr(MetricsRegistry, attr)
            live.clear()
            live.update(saved)


# ---------------------------------------------------------------------------
# synthetic dataset: two NR columns linear in theta + small seeded noise
# ---------------------------------------------------------------------------
def _build_linear_dataset(
    *,
    n_content: int = 6,
    severities: int = 12,
    families: tuple[str, ...] = ("rician", "gaussian_noise"),
    seed: int = 0,
    noise: float = 0.02,
) -> MetricEvaluationDataset:
    """Build a dataset where ``nr_a`` and ``nr_b`` are linear in theta.

    A handful of content ids across two families, with severities sampled on a
    grid in ``[0, 1]``. Two NR columns track theta (one positive, one negative
    slope) plus small seeded Gaussian noise; a third column is pure seeded noise
    (a distractor feature).
    """
    gen = torch.Generator().manual_seed(seed)
    samples: list[DegradationSample] = []
    nr_a: list[float] = []
    nr_b: list[float] = []
    nr_noise: list[float] = []
    thetas = torch.linspace(0.05, 0.95, severities)
    for c in range(n_content):
        cid = f"subj_{c:02d}"
        for fam in families:
            for theta in thetas.tolist():
                clean = torch.zeros(1, 8, 8)
                degraded = torch.zeros(1, 8, 8)
                samples.append(
                    DegradationSample(
                        clean=clean,
                        degraded=degraded,
                        degradation_family=fam,
                        severity={"theta": float(theta)},
                        content_id=cid,
                        seed=seed,
                    )
                )
                eps = torch.randn(3, generator=gen) * noise
                nr_a.append(2.0 * theta + 0.1 + float(eps[0]))
                nr_b.append(-1.5 * theta + 0.5 + float(eps[1]))
                nr_noise.append(float(eps[2]))
    return MetricEvaluationDataset(
        samples=samples,
        metric_values={"nr_a": nr_a, "nr_b": nr_b, "nr_noise": nr_noise},
    )


# ---------------------------------------------------------------------------
# fit quality
# ---------------------------------------------------------------------------
def test_fit_recovers_strong_severity_signal() -> None:
    torch.manual_seed(0)
    ds = _build_linear_dataset(seed=1)
    model, metrics = fit_nr_quality_index(
        ds, ["nr_a", "nr_b", "nr_noise"], target="theta", seed=3
    )

    assert isinstance(model, NRQualityIndexModel)
    assert metrics["feature_names"] == ["nr_a", "nr_b", "nr_noise"]
    assert metrics["n_train"] > 0 and metrics["n_test"] > 0

    assert math.isfinite(metrics["r2_test"])
    assert metrics["r2_test"] > 0.5, metrics["r2_test"]

    assert math.isfinite(metrics["spearman_test"])
    assert metrics["spearman_test"] > 0.5, metrics["spearman_test"]

    assert math.isfinite(metrics["ece"])
    assert 0.0 <= metrics["ece"] <= 1.0

    assert math.isfinite(metrics["auroc"])
    assert metrics["auroc"] > 0.5, metrics["auroc"]


def test_group_split_no_content_overlap() -> None:
    # With two families the cross_degradation transfer probe is populated.
    ds = _build_linear_dataset(seed=2)
    _model, metrics = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=5)
    cd = metrics["cross_degradation"]
    assert cd is not None
    assert set(cd["train_families"]).isdisjoint(set(cd["held_out_families"]))
    assert math.isfinite(cd["r2"]) or cd["r2"] != cd["r2"]  # finite or NaN, never raises


def test_single_family_cross_degradation_is_null() -> None:
    ds = _build_linear_dataset(seed=4, families=("rician",))
    _model, metrics = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=7)
    assert metrics["cross_degradation"] is None


def test_insufficient_data_returns_nan_model_with_reason() -> None:
    # One content id => cannot form a group split.
    gen = torch.Generator().manual_seed(11)
    samples = [
        DegradationSample(
            clean=torch.zeros(1, 4, 4),
            degraded=torch.zeros(1, 4, 4),
            degradation_family="rician",
            severity={"theta": float(t)},
            content_id="only_subj",
            seed=11,
        )
        for t in torch.linspace(0.1, 0.9, 6).tolist()
    ]
    vals = (2.0 * torch.linspace(0.1, 0.9, 6) + 0.05 * torch.randn(6, generator=gen)).tolist()
    ds = MetricEvaluationDataset(samples=samples, metric_values={"nr_a": vals})
    model, metrics = fit_nr_quality_index(ds, ["nr_a"], seed=0)
    assert isinstance(model, NRQualityIndexModel)
    assert metrics["r2_test"] != metrics["r2_test"]  # NaN
    assert "reason" in metrics
    assert metrics["n_train"] == 0 and metrics["n_test"] == 0


def test_no_features_returns_nan_model() -> None:
    ds = _build_linear_dataset(seed=6)
    model, metrics = fit_nr_quality_index(ds, [], seed=0)
    assert metrics["reason"] == "no NR feature columns supplied"
    assert metrics["r2_test"] != metrics["r2_test"]
    assert model.feature_names == []


# ---------------------------------------------------------------------------
# sklearn-absent closed-form fallback
# ---------------------------------------------------------------------------
def test_closed_form_fallback_produces_finite_model() -> None:
    ds = _build_linear_dataset(seed=8)
    model, metrics = fit_nr_quality_index(
        ds, ["nr_a", "nr_b", "nr_noise"], seed=9, force_closed_form=True
    )
    assert metrics["backend"] == "ridge_closed_form"
    assert torch.isfinite(model.weights).all()
    assert math.isfinite(model.intercept)
    assert math.isfinite(metrics["r2_test"])
    assert metrics["r2_test"] > 0.5


def test_closed_form_fallback_via_import_monkeypatch(monkeypatch) -> None:
    """Force ``import sklearn`` to raise and confirm the fallback path runs."""
    import builtins

    real_import = builtins.__import__

    def _no_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("sklearn forced absent for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_sklearn)

    ds = _build_linear_dataset(seed=10)
    model, metrics = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=12)
    assert metrics["backend"] == "ridge_closed_form"
    assert torch.isfinite(model.weights).all()


def test_default_backend_reflects_path() -> None:
    ds = _build_linear_dataset(seed=13)
    _model, metrics = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=14)
    assert metrics["backend"] in ("sklearn_ridge", "ridge_closed_form")


# ---------------------------------------------------------------------------
# persistence round-trip
# ---------------------------------------------------------------------------
def test_to_dict_from_dict_round_trip() -> None:
    ds = _build_linear_dataset(seed=15)
    model, _metrics = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=16)
    payload = model.to_dict()
    # JSON-able: only python scalars / lists / strings.
    assert isinstance(payload["weights"], list)
    assert all(isinstance(v, float) for v in payload["weights"])
    rebuilt = NRQualityIndexModel.from_dict(payload)
    row = torch.tensor([[0.5, 0.2]], dtype=torch.float64)
    assert torch.allclose(model.predict(row), rebuilt.predict(row))


# ---------------------------------------------------------------------------
# registration as a metric
# ---------------------------------------------------------------------------
def test_register_and_call_with_features() -> None:
    ds = _build_linear_dataset(seed=17)
    model, _metrics = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=18)
    register_nr_quality_index(model)

    assert MetricsRegistry.is_registered("nr_quality_index")
    assert MetricsRegistry.requires_reference("nr_quality_index") is False
    assert MetricsRegistry.needs_context("nr_quality_index") is False

    metric = get_metric("nr_quality_index")
    assert metric.higher_is_better is False

    val = metric(torch.zeros(1, 4, 4), nr_features={"nr_a": 0.5, "nr_b": 0.2})
    assert isinstance(val, float)
    assert math.isfinite(val)

    # No feature dict => NaN (research-mode meta-metric).
    nan_val = metric(torch.zeros(1, 4, 4))
    assert nan_val != nan_val  # NaN

    # Empty dict => NaN.
    empty_val = metric(torch.zeros(1, 4, 4), nr_features={})
    assert empty_val != empty_val


def test_register_is_idempotent() -> None:
    ds = _build_linear_dataset(seed=19)
    model1, _ = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=20)
    register_nr_quality_index(model1)
    # Re-fit (different seed) and re-register — must not raise the duplicate guard.
    model2, _ = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=21)
    register_nr_quality_index(model2)
    register_nr_quality_index(model2)  # third time, still fine

    metric = get_metric("nr_quality_index")
    val = metric(torch.zeros(1, 4, 4), nr_features={"nr_a": 0.7, "nr_b": 0.1})
    assert math.isfinite(val)


def test_predict_imputes_non_finite_features() -> None:
    ds = _build_linear_dataset(seed=22)
    model, _ = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=23)
    # A NaN feature must be imputed (to the train mean), not propagated.
    row = torch.tensor([[float("nan"), 0.3]], dtype=torch.float64)
    out = model.predict(row)
    assert torch.isfinite(out).all()


def test_predict_clamped_to_unit_interval() -> None:
    ds = _build_linear_dataset(seed=24)
    model, _ = fit_nr_quality_index(ds, ["nr_a", "nr_b"], seed=25)
    # Extreme feature values must not push the prediction outside [0, 1].
    row = torch.tensor([[100.0, -100.0]], dtype=torch.float64)
    out = model.predict(row)
    assert float(out.min().item()) >= 0.0
    assert float(out.max().item()) <= 1.0
