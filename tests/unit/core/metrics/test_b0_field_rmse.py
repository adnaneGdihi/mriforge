"""b0_field_rmse — absolute Hz RMSE between an estimated B0 field and a real one.

The Hz-domain headline metric for exp_vf_29. It grades only Hz-field-vs-Hz-field;
a complex/2-channel *image* estimate (the bssfp_peak_finder facade) or a shape
mismatch returns NaN — the runtime half of the two-lock parametrization guard
(pitfall #16). Registered, so it is reachable via validation.metrics.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mriforge.core.metrics.b0_field_rmse import B0FieldRMSE
from mriforge.core.metrics.registry import MetricsRegistry


def test_registered_and_lower_is_better():
    m = MetricsRegistry.get("b0_field_rmse")
    assert m.name == "b0_field_rmse"
    assert m.higher_is_better is False


def test_perfect_match_is_zero():
    m = B0FieldRMSE()
    field = torch.full((2, 1, 16, 16), 25.0)
    assert m(field, field.clone()) == 0.0


def test_rmse_value_in_hz():
    m = B0FieldRMSE()
    est = torch.full((2, 1, 8, 8), 30.0)
    ref = torch.full((2, 1, 8, 8), 20.0)
    assert abs(m(est, ref) - 10.0) < 1e-4  # constant 10 Hz offset → 10 Hz RMSE


def test_complex_image_estimate_is_skipped_not_graded():
    # The bssfp_peak_finder facade: a complex *image* graded as a B0 field.
    m = B0FieldRMSE()
    est_image = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    ref_field = torch.zeros(2, 1, 16, 16)
    assert math.isnan(m(est_image, ref_field))


def test_shape_mismatch_is_skipped_not_graded():
    # A 2-channel (e.g. real/imag) image vs a 1-channel field → not comparable.
    m = B0FieldRMSE()
    est = torch.zeros(2, 2, 16, 16)
    ref = torch.zeros(2, 1, 16, 16)
    assert math.isnan(m(est, ref))
