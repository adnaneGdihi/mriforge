"""Tests for time-varying classifier-free-guidance schedules.

The schedules live in :mod:`spectramr.models.blocks.cfg_schedule` and
plug into :meth:`MultiAxisCFGMixin.cfg_sample_step` via the new
optional ``schedule=`` / ``t=`` kwargs (multi-contrast Item 3,
``TODO/backlog_multi_contrast_remaining.md``).

Two non-negotiable invariants:

* Identity (no schedule supplied) reproduces the original
  constant-weight behaviour **bit-exact**. This is what makes the
  extension non-breaking.
* The YAML-dispatch factory rejects unknown ``type`` values rather
  than silently falling back to a constant -- CLAUDE.md pitfall #9.
"""
from __future__ import annotations

import math

import pytest

from spectramr.models.blocks.cfg_schedule import (
    CosineCFGSchedule,
    LinearCFGSchedule,
    OscillatingCosineCFGSchedule,
    PerAxisSchedule,
    StepCFGSchedule,
    make_per_axis_schedule,
    make_schedule,
)


def test_linear_schedule_endpoints() -> None:
    s = LinearCFGSchedule(start=2.0, end=0.5)
    assert s(0.0) == pytest.approx(2.0)
    assert s(1.0) == pytest.approx(0.5)
    assert s(0.5) == pytest.approx(1.25)


def test_linear_schedule_clips_outside_unit_interval() -> None:
    """Slight numerical excursions outside [0, 1] must saturate, not extrapolate."""
    s = LinearCFGSchedule(start=2.0, end=0.5)
    assert s(-0.1) == pytest.approx(2.0)
    assert s(1.1) == pytest.approx(0.5)


def test_cosine_ramp_endpoints_match_linear() -> None:
    """The cosine ramp must share endpoints with a same-spec linear."""
    s = CosineCFGSchedule(start=3.0, end=0.0)
    assert s(0.0) == pytest.approx(3.0)
    assert s(1.0) == pytest.approx(0.0)
    # Midpoint is the average (smooth half-cosine).
    assert s(0.5) == pytest.approx(1.5)


def test_oscillating_cosine_endpoints() -> None:
    """The oscillating cosine returns to the baseline + amplitude at t=0 and t=1."""
    s = OscillatingCosineCFGSchedule(amplitude=0.5, baseline=1.5)
    # cos(0) = cos(2*pi) = 1 -> baseline + amplitude
    assert s(0.0) == pytest.approx(2.0)
    assert s(1.0) == pytest.approx(2.0)
    # cos(pi) = -1 -> baseline - amplitude
    assert s(0.5) == pytest.approx(1.0)


def test_step_schedule_transition() -> None:
    s = StepCFGSchedule(high=4.0, low=1.0, transition=0.6)
    assert s(0.0) == 4.0
    assert s(0.5) == 4.0
    assert s(0.6) == 1.0  # transition is the boundary; >= transition => low
    assert s(1.0) == 1.0


def test_per_axis_default_is_identity_for_missing_axes() -> None:
    """Axes absent from the schedule default to multiplier 1.0."""
    bundle = PerAxisSchedule(schedules={"anatomy": LinearCFGSchedule(2.0, 0.5)})
    assert bundle.weight_at("pathology", 0.5) == pytest.approx(1.0)
    assert bundle.weight_at("anatomy", 0.0) == pytest.approx(2.0)


def test_make_schedule_dispatches_known_types() -> None:
    assert isinstance(make_schedule({"type": "linear", "start": 1, "end": 0}), LinearCFGSchedule)
    assert isinstance(
        make_schedule({"type": "cosine_ramp", "start": 1, "end": 0}), CosineCFGSchedule
    )
    assert isinstance(
        make_schedule({"type": "cosine", "amplitude": 0.5, "baseline": 1.5}),
        OscillatingCosineCFGSchedule,
    )


def test_make_schedule_rejects_unknown_type_loudly() -> None:
    """Pitfall #9: a typo in the YAML must raise, not silently fall back."""
    with pytest.raises(ValueError, match="Unknown CFG schedule type"):
        make_schedule({"type": "exponential", "rate": 0.5})


def test_make_schedule_requires_type_key() -> None:
    with pytest.raises(ValueError, match="must declare"):
        make_schedule({"start": 1.0, "end": 0.0})


def test_make_per_axis_schedule_yaml_round_trip() -> None:
    """The factory must build the same bundle the YAML reference expects."""
    yaml_block = {
        "anatomy": {"type": "linear", "start": 6.0, "end": 1.0},
        "pathology": {"type": "linear", "start": 1.0, "end": 6.0},
        "contrast": {"type": "cosine", "amplitude": 0.5, "baseline": 1.5},
    }
    bundle = make_per_axis_schedule(yaml_block)
    assert bundle is not None
    assert bundle.weight_at("anatomy", 0.0) == pytest.approx(6.0)
    assert bundle.weight_at("pathology", 1.0) == pytest.approx(6.0)
    assert bundle.weight_at("contrast", 0.5) == pytest.approx(1.0)
    # Unknown axis still defaults to identity.
    assert bundle.weight_at("vendor", 0.5) == pytest.approx(1.0)


def test_make_per_axis_schedule_returns_none_for_empty() -> None:
    """Empty or None spec returns None so callers can pass it as schedule=None."""
    assert make_per_axis_schedule(None) is None
    assert make_per_axis_schedule({}) is None


def test_cfg_sample_step_identity_when_no_schedule() -> None:
    """Sampling with ``schedule=None`` must match the prior constant-weight output bit-exact.

    This is the non-breaking-by-construction guarantee that makes the
    extension safe to land without touching any existing strategy.
    """
    import torch

    from spectramr.models.blocks.multi_axis_cfg import MultiAxisCFGMixin

    class _StubMixin(MultiAxisCFGMixin):
        def _predict_noise(self, x, conditions):
            # Predict a value that depends on which axes are active so
            # the CFG combination is non-trivial.
            score = sum(
                1.0 for v in conditions.values() if v is not None
            )
            return torch.full_like(x, score)

    model = _StubMixin()
    x = torch.zeros(2, 1, 4, 4)
    conditions = {
        "anatomy": torch.zeros(2, 4),
        "pathology": torch.zeros(2, 4),
    }
    weights = {"anatomy": 2.0, "pathology": 1.0}

    out_no_schedule = model.cfg_sample_step(x, conditions, weights)
    identity_bundle = PerAxisSchedule(schedules={})
    out_identity = model.cfg_sample_step(
        x, conditions, weights, schedule=identity_bundle, t=0.7
    )
    assert torch.allclose(out_no_schedule, out_identity, atol=0.0)
