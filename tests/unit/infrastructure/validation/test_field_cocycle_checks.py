"""Tier-1 audit guard tests for the field_cocycle arm (MICCAI MRIxFields2026, 4.2)."""

from __future__ import annotations

from types import SimpleNamespace

# Import the model so it is registered when the single-model guard queries the
# registry (get_model_class).
import mriforge.models.generators.field_cocycle_generator  # noqa: F401
from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker


def _checker() -> ConfigHealthChecker:
    return object.__new__(ConfigHealthChecker)


def _cfg(**overrides):
    fc = dict(
        cocycle_weight=0.1,
        identity_weight=0.5,
        adversarial_weight=0.1,
        reference_field_tesla=3.0,
        field_min_tesla=0.1,
        field_max_tesla=7.0,
    )
    fc.update(overrides.pop("field_cocycle", {}))
    model_type = overrides.pop("model_type", "field_cocycle_generator")
    mode = overrides.pop("mode", "field_cocycle")
    return SimpleNamespace(
        training=SimpleNamespace(
            training_mode=mode,
            strategy_class=mode,
            field_cocycle=SimpleNamespace(**fc),
        ),
        model=SimpleNamespace(model_type=model_type),
    )


def _by_name(results):
    return {r.check_name: r for r in results}


def test_valid_arm_all_pass() -> None:
    results = _checker().check_field_cocycle_arm(_cfg())
    by = _by_name(results)
    assert by["field_cocycle_fidelity_nonzero"].passed
    assert by["field_cocycle_reference_in_range"].passed
    assert by["field_cocycle_single_model"].passed


def test_not_applicable_for_other_mode() -> None:
    results = _checker().check_field_cocycle_arm(_cfg(mode="reconstruction"))
    assert len(results) == 1
    assert results[0].passed and results[0].check_name == "field_cocycle_arm"


def test_fidelity_nonzero_fires_when_adversarial_zero() -> None:
    results = _checker().check_field_cocycle_arm(
        _cfg(field_cocycle={"adversarial_weight": 0.0})
    )
    r = _by_name(results)["field_cocycle_fidelity_nonzero"]
    assert not r.passed and r.severity == "error"


def test_reference_out_of_range_fires() -> None:
    results = _checker().check_field_cocycle_arm(
        _cfg(field_cocycle={"reference_field_tesla": 10.0})
    )
    r = _by_name(results)["field_cocycle_reference_in_range"]
    assert not r.passed and r.severity == "error"


def test_single_model_guard_fires_on_wrong_model() -> None:
    results = _checker().check_field_cocycle_arm(
        _cfg(model_type="anatomy_field_renderer")
    )
    r = _by_name(results)["field_cocycle_single_model"]
    assert not r.passed and r.severity == "error"
