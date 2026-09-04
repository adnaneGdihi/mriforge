"""Tests for the per-strategy knob registry (Tier-B B1 seam).

Verifies the mechanism that closes the ``extra='allow'`` facade hole:
registration (explicit + decorator), inertness for unregistered strategies
(so it can never false-reject an unmigrated arm), flagging of undeclared knobs,
and integration with ``ValidatorRegistry``.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.strategy_knobs import (
    build_strategy_knob_rule,
    known_knobs,
    register_strategy_knob_names,
    register_strategy_knobs,
    validate_strategy_knobs,
)
from spectramr.config.schemas.strictness import StrictSchema

_FAKE = "tests.fake.SomeStrategy"


def test_register_and_lookup():
    register_strategy_knob_names(_FAKE + ".lookup", ["a", "b"])
    assert known_knobs(_FAKE + ".lookup") == frozenset({"a", "b"})


def test_register_merges_union():
    register_strategy_knob_names(_FAKE + ".merge", ["a"])
    register_strategy_knob_names(_FAKE + ".merge", ["b"])
    assert known_knobs(_FAKE + ".merge") == frozenset({"a", "b"})


def test_empty_registration_raises():
    with pytest.raises(ValueError, match="empty"):
        register_strategy_knob_names(_FAKE + ".empty", [])


def test_decorator_uses_schema_fields():
    @register_strategy_knobs(_FAKE + ".decorated")
    class _Knobs(StrictSchema):
        sampler_steps: int = 28
        multistep_cold_sampling: bool = False

    assert known_knobs(_FAKE + ".decorated") == frozenset(
        {"sampler_steps", "multistep_cold_sampling"}
    )


def test_unregistered_strategy_is_inert():
    cfg = {"training": {"strategy_class": "tests.never.Registered", "weird_knob": 1}}
    assert validate_strategy_knobs(cfg) == []  # inert → no false rejection


def test_no_strategy_class_is_inert():
    assert validate_strategy_knobs({"training": {"foo": 1}}) == []
    assert validate_strategy_knobs({}) == []


def test_flags_undeclared_knob():
    register_strategy_knob_names(_FAKE + ".flag", ["real_knob"])
    cfg = {
        "training": {
            "strategy_class": _FAKE + ".flag",
            "real_knob": 1,
            "typoed_knob": 2,
        }
    }
    msgs = validate_strategy_knobs(cfg)
    assert len(msgs) == 1
    assert "typoed_knob" in msgs[0]
    assert "real_knob" not in msgs[0]


def test_declared_training_fields_are_allowed():
    # strategy_class itself is a declared field of TrainingStrategyConfigSchema,
    # so it must never be flagged even though it is not a registered "knob".
    register_strategy_knob_names(_FAKE + ".declared", ["real_knob"])
    cfg = {"training": {"strategy_class": _FAKE + ".declared", "real_knob": 1}}
    assert validate_strategy_knobs(cfg) == []


def test_integrates_with_validator_registry():
    from spectramr.config.schemas.validator_registry import ValidatorRegistry

    register_strategy_knob_names(_FAKE + ".registry", ["real_knob"])
    rule = build_strategy_knob_rule(severity="warning")
    reg = ValidatorRegistry()
    reg.register(rule)
    # validate() returns (rule_name, message) tuples.
    results = reg.validate({"training": {"strategy_class": _FAKE + ".registry", "bogus": 1}})
    assert any(name == "strategy_knobs_are_declared" and "bogus" in msg for name, msg in results)
