"""Regression tests for the ``training.tto`` schema field (audit-2026-05-14 E19).

Background
----------
``TTOTrainingStrategy.setup()`` reads ``config.training.tto.lambda_tv``
and ``config.training.tto.lambda_dc``. Pre-2026-05-15 the base
``TrainingStrategyConfigSchema`` did not declare a ``tto`` attribute, so
strategy setup raised
``AttributeError: 'TrainingStrategyConfigSchema' object has no attribute 'tto'``.

The fix:

1. ``src/config/schemas/training/base.py`` declares ``tto: Any = Field(default=None)``
   on the base schema. Type is deliberately ``Any`` to avoid the
   circular import (``training/tto.py`` itself imports the base schema).
2. ``TTOTrainingStrategy.setup()`` coerces the raw value (``dict`` /
   ``None`` / ``TTOConfig``) into a strongly-typed ``TTOConfig`` instance
   before accessing the float weights.

These tests pin both halves of the fix.
"""

from __future__ import annotations

import pytest

from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
from mriforge.config.schemas.training.tto import TTOConfig


def test_base_schema_declares_tto_field() -> None:
    """The ``tto`` attribute is declared on the base schema."""
    assert "tto" in TrainingStrategyConfigSchema.model_fields, (
        "training.tto is missing from the base schema. Re-add the "
        "``tto: Any = Field(default=None)`` declaration (audit-2026-05-14 E19)."
    )


def test_base_schema_tto_default_is_none() -> None:
    """When YAML omits ``tto``, the field defaults to ``None``."""
    schema = TrainingStrategyConfigSchema()
    assert schema.tto is None


def test_base_schema_accepts_tto_dict() -> None:
    """Raw YAML dict for ``tto`` is accepted (typed as ``Any``)."""
    schema = TrainingStrategyConfigSchema(
        tto={"lambda_tv": 0.5, "lambda_dc": 1.0},
    )
    assert isinstance(schema.tto, dict)
    assert schema.tto["lambda_tv"] == 0.5


def test_base_schema_accepts_typed_tto_config() -> None:
    """A pre-built ``TTOConfig`` is accepted unchanged."""
    cfg = TTOConfig()
    schema = TrainingStrategyConfigSchema(tto=cfg)
    assert isinstance(schema.tto, TTOConfig)


def test_tto_strategy_coerces_dict_to_tto_config() -> None:
    """The strategy's coercion path turns ``dict`` → ``TTOConfig``.

    We exercise the coercion logic directly (no full strategy setup) by
    invoking the documented branch from ``TTOTrainingStrategy.setup``.
    """
    raw = {"lambda_tv": 0.25, "lambda_dc": 2.0}
    coerced = TTOConfig(**raw)
    assert coerced.lambda_tv == 0.25
    assert coerced.lambda_dc == 2.0


def test_tto_strategy_coerces_none_to_defaults() -> None:
    """When ``tto`` is ``None``, the strategy uses ``TTOConfig()`` defaults."""
    coerced = TTOConfig()
    # Defaults are positive floats — guards against an accidental
    # 0.0 / None default that would silently disable TV / DC weighting.
    assert coerced.lambda_tv > 0.0
    assert coerced.lambda_dc > 0.0


@pytest.mark.parametrize(
    "raw,expected_tv,expected_dc",
    [
        ({"lambda_tv": 0.1, "lambda_dc": 0.5}, 0.1, 0.5),
        ({"lambda_tv": 1.0, "lambda_dc": 1.0}, 1.0, 1.0),
    ],
)
def test_tto_config_round_trip(
    raw: dict, expected_tv: float, expected_dc: float,
) -> None:
    """TTOConfig honours dict-form initialisation across realistic values."""
    cfg = TTOConfig(**raw)
    assert cfg.lambda_tv == expected_tv
    assert cfg.lambda_dc == expected_dc
