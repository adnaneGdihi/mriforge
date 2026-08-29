"""Unit tests for models/stability/stability_linter.py.

Covers: StabilityAnalyzer — analyze_architecture, analyze_training_stability,
        run_full_analysis (scoring + keys).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from mriforge.models.stability.stability_linter import StabilityAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_bn_model() -> nn.Module:
    """Tiny model with BN so normalization score is rewarded."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def _tiny_no_norm_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_analyzer_canary():
    """StabilityAnalyzer initialises and returns a non-trivial architecture score."""
    analyzer = StabilityAnalyzer(_small_bn_model(), model_type="unet")
    results = analyzer.analyze_architecture()
    assert "score" in results
    score = results["score"]
    assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type,expected_key_in_recs",
    [
        ("diffusion_unet", "diffusion"),
        ("gan_generator", "spectral"),
        ("vit_encoder", "dropout"),
    ],
)
def test_training_stability_model_type_recommendations(model_type, expected_key_in_recs):
    """Recommendations include the expected keyword for known model types."""
    analyzer = StabilityAnalyzer(_tiny_no_norm_model(), model_type=model_type)
    results = analyzer.analyze_training_stability()
    combined = " ".join(results.get("optimizer_recommendations", []))
    combined += " ".join(results.get("regularization_needs", []))
    combined += " ".join(analyzer.recommendations)
    assert expected_key_in_recs.lower() in combined.lower(), (
        f"Expected '{expected_key_in_recs}' in recommendations for {model_type}; got: {combined}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("model_fn", [_small_bn_model, _tiny_no_norm_model])
def test_architecture_score_bounded(model_fn):
    """Architecture score is always in [0, 100]."""
    analyzer = StabilityAnalyzer(model_fn(), model_type="standard")
    results = analyzer.analyze_architecture()
    assert 0 <= results["score"] <= 100


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_full_analysis_returns_all_keys():
    """run_full_analysis returns all expected top-level keys."""
    analyzer = StabilityAnalyzer(_tiny_no_norm_model(), model_type="standard")
    results = analyzer.run_full_analysis()
    for key in ("architecture_score", "parameter_score", "gradient_score",
                "training_score", "overall_score"):
        assert key in analyzer.analysis_results, f"Missing key: {key}"


@pytest.mark.unit
def test_no_norm_model_penalized():
    """Model without normalization layers gets a lower architecture score."""
    analyzer_no_norm = StabilityAnalyzer(_tiny_no_norm_model(), model_type="test")
    analyzer_bn = StabilityAnalyzer(_small_bn_model(), model_type="test")
    score_no_norm = analyzer_no_norm.analyze_architecture()["score"]
    score_bn = analyzer_bn.analyze_architecture()["score"]
    assert score_no_norm <= score_bn, (
        f"Expected no-norm ({score_no_norm}) <= bn ({score_bn})"
    )


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_analyze_training_stability_score_inversely_proportional_to_sensitivity():
    """Training score = 100 - lr_sensitivity (the code makes this explicit)."""
    analyzer = StabilityAnalyzer(_tiny_no_norm_model(), model_type="standard")
    results = analyzer.analyze_training_stability()
    expected = max(0, min(100, 100 - results["learning_rate_sensitivity"]))
    assert results["score"] == expected
