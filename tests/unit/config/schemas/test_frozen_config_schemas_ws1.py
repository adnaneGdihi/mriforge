"""WS-1 NN#1 triage: config schemas must be frozen.

AdapterStepSchema / AdaptersConfigSchema (data-05) and ReportingSettings
(misc-04) were missing ``frozen=True``. These assert the schemas still construct
and now reject post-construction mutation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.adapters import AdaptersConfigSchema, AdapterStepSchema
from spectramr.config.schemas.reporting import ReportingSettings


def _assert_frozen(instance) -> None:
    field = next(iter(type(instance).model_fields))
    with pytest.raises(ValidationError):
        setattr(instance, field, getattr(instance, field))


def test_adapter_step_schema_is_frozen() -> None:
    # extra="allow" still forwards kwargs at construction…
    step = AdapterStepSchema(name="ifft_kspace_to_image", axis=2)
    assert step.name == "ifft_kspace_to_image"
    # …but the instance is immutable afterwards.
    _assert_frozen(step)


def test_adapters_config_schema_is_frozen() -> None:
    _assert_frozen(AdaptersConfigSchema())


def test_reporting_settings_is_frozen() -> None:
    _assert_frozen(ReportingSettings())
