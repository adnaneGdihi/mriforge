"""Tests for the NR-validation shared contract + Tier-1 concordance scorer.

Covers :mod:`mriforge.application.use_cases.nr_validation.cards`:
``TierResult`` / ``MetricCard`` / ``build_cards`` folding, the NaN-safe rank
statistics (``spearman`` / ``kendall_tau``), the per-image ``eval_metric``
evaluator, and ``score_concordance`` (spec §8.1) including the constant-trajectory
guard that defeats the repo's tie-blind Spearman.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.application.use_cases.nr_validation.cards import (
    MetricCard,
    TierResult,
    build_cards,
    eval_metric,
    kendall_tau,
    score_concordance,
    spearman,
)
from mriforge.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
)


def _sample(family: str, theta: float, idx: int) -> DegradationSample:
    clean = torch.zeros(1, 8, 8)
    return DegradationSample(
        clean=clean,
        degraded=clean + theta,
        degradation_family=family,
        severity={"theta": float(theta)},
        content_id=f"c{idx % 2}",
        seed=idx,
    )


def _dataset() -> MetricEvaluationDataset:
    families = ["rician", "gibbs"]
    thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
    samples: list[DegradationSample] = []
    monotone: list[float] = []
    flat: list[float] = []
    noisy: list[float] = []
    g = torch.Generator().manual_seed(0)
    i = 0
    for fam in families:
        for t in thetas:
            samples.append(_sample(fam, t, i))
            monotone.append(t)  # perfectly tracks severity
            flat.append(1.0)  # constant — zero discrimination
            noisy.append(float(torch.rand(1, generator=g).item()))
            i += 1
    return MetricEvaluationDataset(
        samples=samples,
        metric_values={"monotone": monotone, "flat": flat, "noisy": noisy},
    )


# --------------------------------------------------------------------------- stats
def test_spearman_perfect_and_anti():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0, abs=1e-9)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0, abs=1e-9)


def test_spearman_nan_safe():
    # < 3 finite pairs -> 0.0, never raises
    assert spearman([1.0, float("nan")], [1.0, 2.0]) == 0.0


def test_kendall_tau_monotone():
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0, abs=1e-9)
    assert kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0, abs=1e-9)
    # too few finite -> 0.0
    assert kendall_tau([1.0, float("nan")], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------- TierResult
def test_tier_result_to_dict_nan_becomes_none():
    tr = TierResult("concordance", True, float("nan"), {"k": 1})
    d = tr.to_dict()
    assert d["tier"] == "concordance"
    assert d["passed"] is True
    assert d["score"] is None  # NaN serialised as null
    assert d["detail"] == {"k": 1}


def test_metric_card_overall_pass():
    card = MetricCard("m")
    assert card.overall_pass is False  # no tiers -> not passing
    card.tiers["a"] = TierResult("a", True, 0.9)
    card.tiers["b"] = TierResult("b", True, 0.8)
    assert card.overall_pass is True
    card.tiers["b"] = TierResult("b", False, 0.1)
    assert card.overall_pass is False


def test_build_cards_folds_per_tier():
    per_tier = {
        "concordance": {"m1": TierResult("concordance", True, 0.7)},
        "fr_proxy": {
            "m1": TierResult("fr_proxy", True, 0.8),
            "m2": TierResult("fr_proxy", False, 0.2),
        },
    }
    cards = build_cards(per_tier)
    assert set(cards) == {"m1", "m2"}
    assert set(cards["m1"].tiers) == {"concordance", "fr_proxy"}
    assert cards["m1"].overall_pass is True
    assert cards["m2"].overall_pass is False  # only ran fr_proxy, which failed


# ---------------------------------------------------------------------- eval_metric
def test_eval_metric_finite_for_registered_metric():
    pred = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(1))
    ref = pred.clone()
    val = eval_metric("psnr", pred, ref)
    assert isinstance(val, float)
    assert math.isfinite(val)  # identical images -> finite (high) PSNR


def test_eval_metric_unknown_metric_returns_nan():
    pred = torch.rand(1, 1, 8, 8)
    assert math.isnan(eval_metric("__definitely_not_a_metric__", pred, pred))


# ------------------------------------------------------------------ concordance §8.1
def test_score_concordance_passes_monotone_fails_flat():
    ds = _dataset()
    out = score_concordance(ds, ["monotone", "flat", "noisy"], rho_threshold=0.6)
    assert out["monotone"].passed is True
    assert out["monotone"].score == pytest.approx(1.0, abs=1e-9)
    # flat metric: constant-trajectory guard forces rho=0 -> fails (no spurious 1.0)
    assert out["flat"].passed is False
    assert out["flat"].score == pytest.approx(0.0, abs=1e-9)
    # detail carries the per-family diagnostics
    assert "monotone_fraction" in out["monotone"].detail
    assert out["monotone"].detail["n_families"] == 2


def test_score_concordance_missing_metric_records_reason():
    ds = _dataset()
    out = score_concordance(ds, ["not_in_dataset"], rho_threshold=0.6)
    assert out["not_in_dataset"].passed is False
    assert math.isnan(out["not_in_dataset"].score)
    assert "reason" in out["not_in_dataset"].detail
