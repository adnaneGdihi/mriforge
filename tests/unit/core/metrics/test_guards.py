"""Shared metric guards — the parametrization comparability predicate.

These guard the field/trajectory metrics (``b0_field_rmse``,
``k_space_trajectory_rmse``) against the pitfall-#16 facade: grading an
estimate against a reference that is a *different physical quantity*
(an image graded as a Hz field; a Cartesian per-line ``Δk`` graded as a
spiral ``Δk(t)``). The predicate implements the spec's ``comparable(u, v)``
(shape ∧ kind ∧ units); the metric returns ``nan`` and the strategy seam
returns ``{}`` whenever it does not hold.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics._guards import ComparabilityVerdict, field_comparability


def _t(shape):
    return torch.zeros(*shape)


def test_comparable_when_shape_kind_units_all_match():
    v = field_comparability(
        _t((2, 1, 16, 16)),
        _t((2, 1, 16, 16)),
        estimate_kind="field",
        reference_kind="field",
        estimate_units="Hz",
        reference_units="Hz",
    )
    assert isinstance(v, ComparabilityVerdict)
    assert v.ok is True
    assert v.reason  # a non-empty human-readable reason is always set


def test_not_comparable_on_kind_mismatch_image_vs_field():
    # The bssfp_peak_finder facade: a corrected *image* graded as a B0 *field*.
    v = field_comparability(
        _t((2, 1, 16, 16)),
        _t((2, 1, 16, 16)),
        estimate_kind="image",
        reference_kind="field",
        estimate_units="Hz",
        reference_units="Hz",
    )
    assert v.ok is False
    assert "kind" in v.reason


def test_not_comparable_on_units_mismatch():
    v = field_comparability(
        _t((2, 1, 16, 16)),
        _t((2, 1, 16, 16)),
        estimate_kind="field",
        reference_kind="field",
        estimate_units="rad",
        reference_units="Hz",
    )
    assert v.ok is False
    assert "unit" in v.reason


def test_not_comparable_on_shape_mismatch_cartesian_vs_spiral():
    # The diff_trajectory_opt facade: Cartesian per-line [B,2,H] vs spiral [N,2].
    v = field_comparability(
        _t((1, 2, 16)),
        _t((512, 2)),
        estimate_kind="trajectory",
        reference_kind="trajectory",
        estimate_units="cycles_per_fov",
        reference_units="cycles_per_fov",
    )
    assert v.ok is False
    assert "shape" in v.reason


def test_not_comparable_when_estimate_missing():
    v = field_comparability(
        None,
        _t((2, 1, 16, 16)),
        estimate_kind="field",
        reference_kind="field",
        estimate_units="Hz",
        reference_units="Hz",
    )
    assert v.ok is False
    assert "missing" in v.reason


def test_not_comparable_when_reference_missing():
    v = field_comparability(
        _t((2, 1, 16, 16)),
        None,
        estimate_kind="field",
        reference_kind="field",
        estimate_units="Hz",
        reference_units="Hz",
    )
    assert v.ok is False
    assert "missing" in v.reason


def test_verdict_is_frozen():
    v = ComparabilityVerdict(ok=True, reason="ok")
    with pytest.raises((AttributeError, TypeError)):
        v.ok = False  # type: ignore[misc]
