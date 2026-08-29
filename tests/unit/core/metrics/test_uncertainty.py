"""Tests for ``UncertaintyMetrics``.

Targets ``mriforge.core.metrics.uncertainty``. Three calibration / sharpness
helpers used by validation pipelines for evidential / MC-dropout
uncertainty:

- ``expected_calibration_error`` — bin-by-confidence ECE
- ``sharpness`` — mean uncertainty (lower is sharper)
- ``uncertainty_correlation`` — Pearson correlation between
  uncertainty and absolute error

Categories:

- ``ECE == 0`` for a perfectly-calibrated Bernoulli classifier
- ``sharpness`` returns the mean of the uncertainty tensor
- ``uncertainty_correlation == 1`` when uncertainty linearly tracks error
"""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.uncertainty import UncertaintyMetrics

# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated_classifier() -> None:
    """Always-correct + zero uncertainty → ECE = 0."""
    pred = torch.tensor([1, 0, 1, 0])
    target = torch.tensor([1, 0, 1, 0])
    uncertainties = torch.zeros(4)  # confidence = 1.0
    ece = UncertaintyMetrics.expected_calibration_error(pred, target, uncertainties)
    # All predictions correct, all confidences = 1, so ECE = 0
    assert pytest.approx(ece, abs=1e-6) == 0.0


def test_ece_returns_float() -> None:
    """Return type is plain Python float (not tensor)."""
    pred = torch.tensor([1, 0])
    target = torch.tensor([0, 1])
    uncertainties = torch.tensor([0.5, 0.5])
    ece = UncertaintyMetrics.expected_calibration_error(pred, target, uncertainties)
    assert isinstance(ece, float)


def test_ece_no_samples_in_some_bins() -> None:
    """Empty bins are silently skipped (no NaN / divide-by-zero)."""
    pred = torch.zeros(100, dtype=torch.long)
    target = torch.zeros(100, dtype=torch.long)
    # All confidences in a single bin (uncertainty ≈ 0).
    uncertainties = torch.zeros(100) + 0.001
    ece = UncertaintyMetrics.expected_calibration_error(pred, target, uncertainties, n_bins=10)
    assert torch.isfinite(torch.tensor(ece))


# ---------------------------------------------------------------------------
# sharpness
# ---------------------------------------------------------------------------


def test_sharpness_returns_mean() -> None:
    """``sharpness`` is just the mean of the uncertainty tensor."""
    uncertainties = torch.tensor([0.1, 0.3, 0.2])
    expected = uncertainties.mean().item()
    assert UncertaintyMetrics.sharpness(uncertainties) == expected


def test_sharpness_zero_for_zero_uncertainty() -> None:
    """Zero uncertainty everywhere → sharpness = 0."""
    assert UncertaintyMetrics.sharpness(torch.zeros(8)) == 0.0


# ---------------------------------------------------------------------------
# uncertainty_correlation
# ---------------------------------------------------------------------------


def test_correlation_identity_perfect() -> None:
    """``uncertainty == errors`` → Pearson r = 1."""
    errors = torch.linspace(0.0, 1.0, 16)
    uncertainties = errors.clone()
    corr = UncertaintyMetrics.uncertainty_correlation(uncertainties, errors)
    assert pytest.approx(corr, abs=1e-5) == 1.0


def test_correlation_anticorrelated() -> None:
    """Inversely-related → Pearson r = -1."""
    errors = torch.linspace(0.0, 1.0, 16)
    uncertainties = -errors
    corr = UncertaintyMetrics.uncertainty_correlation(uncertainties, errors)
    assert pytest.approx(corr, abs=1e-5) == -1.0


def test_correlation_flat_input_returns_nan() -> None:
    """Constant input has zero variance → Pearson is NaN."""
    errors = torch.linspace(0.0, 1.0, 16)
    uncertainties = torch.zeros_like(errors)
    corr = UncertaintyMetrics.uncertainty_correlation(uncertainties, errors)
    # torch.corrcoef on a constant row yields nan.
    import math

    assert math.isnan(corr)


# ---------------------------------------------------------------------------
# Regression: the final confidence bin is closed on the right (<=), so
# confidence == 1.0 (uncertainty == 0.0) samples are COUNTED, not dropped.
# ---------------------------------------------------------------------------


def _ece_half_open_reference(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """White-box reference of the PRE-FIX (buggy) ECE.

    Every bin is half-open ``[lo, hi)`` — including the final one — so a
    sample with ``confidence == 1.0`` (``uncertainty == 0.0``) falls in no bin
    and is silently dropped. The production code closes the top bin on the
    right; this helper reproduces the old behavior to pin the difference.
    """
    confidence = 1.0 - uncertainties
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
        if in_bin.sum() == 0:
            continue
        bin_acc = (predictions[in_bin] == targets[in_bin]).float().mean()
        bin_confidence = confidence[in_bin].mean()
        ece += in_bin.float().mean() * abs(bin_acc - bin_confidence)
    return float(ece)


@pytest.mark.unit
class TestEceTopBinClosedRight:
    """``expected_calibration_error`` counts the ``confidence == 1.0`` samples."""

    def test_confidence_one_calibrated_samples_yield_zero_ece(self) -> None:
        """conf == 1.0 (unc == 0.0) AND pred == target everywhere → ECE ~ 0.

        These perfectly-calibrated, maximally-confident samples land in the
        final bin ``[0.9, 1.0]``. Because the bin is now closed on the right
        they are counted, the bin's accuracy (1.0) matches its confidence
        (1.0), and the metric is exactly zero — not undefined / skipped.
        """
        pred = torch.tensor([1, 0, 1, 0, 1, 1])
        target = torch.tensor([1, 0, 1, 0, 1, 1])
        uncertainties = torch.zeros(6)  # confidence == 1.0 for every sample

        ece = UncertaintyMetrics.expected_calibration_error(pred, target, uncertainties, n_bins=10)
        assert pytest.approx(ece, abs=1e-6) == 0.0

    def test_top_bin_samples_are_counted_not_dropped(self) -> None:
        """The decisive regression: counting vs. dropping the conf == 1.0 bin.

        Two samples sit at ``confidence == 1.0`` (``uncertainty == 0.0``) but
        are WRONG (a miscalibrated top bin); two more sit at
        ``confidence == 0.0`` and are correct (a calibrated bottom bin).

        * Fixed code (top bin closed on the right): the conf == 1.0 samples are
          counted, contribute ``|acc - conf| = |0 - 1| = 1`` over the whole
          set → ECE == 1.0.
        * Pre-fix code (top bin half-open): those samples are dropped, leaving
          only the calibrated bottom bin → ECE == 0.5.

        Asserting the production value equals the all-counted result (1.0) and
        differs from the half-open reference (0.5) proves the fix.
        """
        pred = torch.tensor([1, 1, 0, 0])
        target = torch.tensor([0, 0, 0, 0])
        uncertainties = torch.tensor([0.0, 0.0, 1.0, 1.0])

        ece = UncertaintyMetrics.expected_calibration_error(pred, target, uncertainties, n_bins=10)
        dropped = _ece_half_open_reference(pred, target, uncertainties, n_bins=10)

        # The conf == 1.0 samples are counted: production matches all-counted.
        assert pytest.approx(ece, abs=1e-6) == 1.0
        # ... and is strictly NOT the buggy (top-bin-dropped) value.
        assert pytest.approx(dropped, abs=1e-6) == 0.5
        assert abs(ece - dropped) > 0.1

    def test_mixed_calibrated_with_max_confidence_stays_zero(self) -> None:
        """Sanity: a calibrated mix that includes conf == 1.0 samples → ECE ~ 0.

        Confirms the closed top bin does not spuriously inflate ECE for a
        well-calibrated dataset that happens to contain maximally-confident
        (uncertainty == 0.0) correct predictions alongside lower-confidence,
        correctly-graded ones.
        """
        # conf == 1.0 correct, conf == 0.0 correctly-graded-as-wrong: calibrated.
        pred = torch.tensor([1, 1, 0, 1])
        target = torch.tensor([1, 1, 1, 0])
        uncertainties = torch.tensor([0.0, 0.0, 1.0, 1.0])

        ece = UncertaintyMetrics.expected_calibration_error(pred, target, uncertainties, n_bins=10)
        assert pytest.approx(ece, abs=1e-6) == 0.0
