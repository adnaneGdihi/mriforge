"""Tests for the canonical perceptual-metric input finiteness guard.

Perceptual metrics (LPIPS, FID) normalise their inputs to a fixed range and
expand channels to 3 before handing them to a frozen VGG / Inception backbone.
When a model diverges and emits NaN/Inf, the silent ``torch.clamp`` inside the
old per-metric normaliser passed the NaN straight through, so the failure only
surfaced deep inside torchmetrics as a confusing *"values in range [nan, nan] …
expected [-1, 1]"* range error (the 2026-06-24 ``exp_hm_06`` / ``exp_hm_10``
crash). The guard catches non-finite inputs at the metric boundary and raises a
precise, actionable error pointing at the real cause — model instability —
per CLAUDE.md pitfall #9 (no silent fallbacks).
"""

import pytest
import torch

from spectramr.core.metrics.metric_input_prep import assert_finite_metric_input


class TestAssertFiniteMetricInput:
    @pytest.mark.unit
    def test_passes_silently_on_finite_inputs(self):
        x = torch.randn(1, 1, 16, 16)
        # Must not raise and must return None (pure guard, no transform).
        assert assert_finite_metric_input("lpips", preds=x, target=x.clone()) is None

    @pytest.mark.unit
    def test_raises_on_nan_pred(self):
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            assert_finite_metric_input("lpips", preds=x, target=torch.zeros_like(x))

    @pytest.mark.unit
    def test_raises_on_inf_target(self):
        x = torch.randn(1, 1, 16, 16)
        with pytest.raises(ValueError, match="non-finite"):
            assert_finite_metric_input(
                "lpips", preds=x, target=torch.full_like(x, float("inf"))
            )

    @pytest.mark.unit
    def test_error_names_the_metric_and_the_offending_field(self):
        x = torch.full((1, 1, 4, 4), float("nan"))
        with pytest.raises(ValueError) as exc:
            assert_finite_metric_input("fid", preds=x, target=torch.zeros_like(x))
        msg = str(exc.value)
        assert "fid" in msg
        assert "preds" in msg

    @pytest.mark.unit
    def test_error_reports_nan_and_inf_counts(self):
        x = torch.zeros(2, 1, 4, 4)
        x[0, 0, 0, 0] = float("nan")
        x[0, 0, 0, 1] = float("inf")
        with pytest.raises(ValueError) as exc:
            assert_finite_metric_input("lpips", preds=x, target=torch.zeros_like(x))
        msg = str(exc.value)
        assert "1 NaN" in msg
        assert "1 Inf" in msg

    @pytest.mark.unit
    def test_ignores_none_tensors(self):
        x = torch.randn(1, 1, 8, 8)
        # A None field (e.g. an optional reference) must be skipped, not crash.
        assert assert_finite_metric_input("lpips", preds=x, target=None) is None
