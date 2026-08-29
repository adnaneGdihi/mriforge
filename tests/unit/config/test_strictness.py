"""Unit tests for the shared strictness bases (``schemas/strictness.py``)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.strictness import (
    CompatSchema,
    StrategySchema,
    StrictSchema,
)


class _Strict(StrictSchema):
    a: int = 1


class _Compat(CompatSchema):
    a: int = 1


class _Strategy(StrategySchema):
    a: int = 1


def test_strict_rejects_unknown_key():
    with pytest.raises(ValidationError):
        _Strict(a=1, unknown=2)


def test_compat_ignores_unknown_key():
    inst = _Compat(a=1, unknown=2)
    assert inst.a == 1
    assert not hasattr(inst, "unknown")  # dropped, not stored


def test_strategy_keeps_unknown_key():
    inst = _Strategy(a=1, unknown=2)
    assert inst.a == 1
    assert inst.unknown == 2  # forwarded for downstream getattr access


@pytest.mark.parametrize("base", [_Strict, _Compat, _Strategy])
def test_all_bases_are_frozen(base):
    inst = base(a=1)
    with pytest.raises(ValidationError):  # frozen → mutation raises
        inst.a = 2


@pytest.mark.parametrize(
    "base,expected",
    [(StrictSchema, "forbid"), (CompatSchema, "ignore"), (StrategySchema, "allow")],
)
def test_extra_policy_matches_tier(base, expected):
    assert base.model_config.get("extra") == expected
    assert base.model_config.get("frozen") is True
    assert base.model_config.get("protected_namespaces") == ()


def test_protected_namespace_disabled_allows_model_fields():
    # A field named model_* must not warn under the bases.
    class _WithModelField(StrictSchema):
        model_type: str = "unet"

    assert _WithModelField().model_type == "unet"
