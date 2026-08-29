"""Tests for region-restricted metrics (banding_region_mse)."""

import pytest

torch = pytest.importorskip("torch")

# E402 is correct in general and wrong here: importing mriforge pulls in torch, so these
# must sit BELOW the importorskip or a torch-less environment gets an ImportError at
# collection instead of a clean skip.
from mriforge.core.metrics.outcome import (  # noqa: E402
    MetricNotApplicableError,
    NotApplicableReason,
)
from mriforge.core.metrics.registry import get_metric, list_available  # noqa: E402


def test_banding_region_mse_registered():
    """The metric resolves through the registry (registration is live)."""
    assert "banding_region_mse" in list_available()
    metric = get_metric("banding_region_mse")
    assert metric.name == "banding_region_mse"
    assert metric.higher_is_better is False
    # alias resolves to the same class
    assert type(get_metric("banding_mse")) is type(metric)


def test_banding_region_mse_restricts_to_mask():
    """Error outside the mask is ignored; only in-ROI error counts."""
    metric = get_metric("banding_region_mse")
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    # Put a unit error only in the top-left quadrant.
    pred[0, 0, 0, 0] = 1.0
    # ROI mask covering only the bottom-right pixel (no error there).
    mask = torch.zeros(4, 4)
    mask[3, 3] = 1.0
    assert metric(pred, target, region_mask=mask) == pytest.approx(0.0)
    # ROI mask covering the errored pixel → MSE 1.0 over a single pixel.
    mask2 = torch.zeros(4, 4)
    mask2[0, 0] = 1.0
    assert metric(pred, target, region_mask=mask2) == pytest.approx(1.0)


def test_banding_region_mse_without_a_mask_raises_instead_of_reporting_full_fov():
    """Regression for B2: it used to return a full-FOV MSE behind a log warning.

    That number is a *different physical quantity* from the one the metric's name
    promises -- and the full-FOV average is exactly what the banding ROI exists to
    see through, so the fallback silently answered the opposite question. It must
    raise, with a machine-readable reason (pitfall #9).
    """
    metric = get_metric("banding_region_mse")
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 0, 0] = 1.0

    with pytest.raises(MetricNotApplicableError) as exc:
        metric(pred, target)

    assert exc.value.reason is NotApplicableReason.MISSING_REQUIRED_ROI_MASK
    assert exc.value.metric == "banding_region_mse"
    # The old behaviour: the full-FOV MSE (1 unit error over 16 px) came back as a
    # confident number. Pin that it is gone.
    assert "full-FOV" in exc.value.detail


def test_banding_region_mse_complex_uses_magnitude():
    """Complex inputs are reduced to magnitude before the squared error."""
    metric = get_metric("banding_region_mse")
    pred = torch.zeros(1, 1, 2, 2, dtype=torch.complex64)
    target = torch.zeros(1, 1, 2, 2, dtype=torch.complex64)
    pred[0, 0, 0, 0] = 3 + 4j  # magnitude 5
    mask = torch.ones(2, 2)
    # only one of four pixels has |.|=5 → mean of [25,0,0,0] = 6.25
    assert metric(pred, target, region_mask=mask) == pytest.approx(6.25)


def test_banding_region_mse_shape_mismatch_raises():
    metric = get_metric("banding_region_mse")
    with pytest.raises(ValueError, match="same shape"):
        metric(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2))
