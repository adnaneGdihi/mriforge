"""``TrainingSettings.metadata`` is the typed ``ExperimentMetadataSchema`` (one owner)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.base import ExperimentMetadataSchema
from spectramr.config.settings import TrainingSettings


def _minimal(metadata: dict) -> dict:
    """The settings fixture ``test_settings.py`` already owns, plus a metadata block."""
    from tests.unit.config.test_settings import _minimal_config

    cfg = _minimal_config()
    cfg["metadata"] = metadata
    return cfg


def test_metadata_is_typed_after_load() -> None:
    settings = TrainingSettings(**_minimal({"name": "arm", "status": "ready", "note": "prose"}))
    assert isinstance(settings.metadata, ExperimentMetadataSchema)
    assert settings.metadata.name == "arm"
    assert settings.metadata.status == "ready"
    assert settings.metadata.model_dump()["note"] == "prose"  # prose keys survive


def test_free_text_status_is_refused_at_load() -> None:
    with pytest.raises(ValidationError):
        TrainingSettings(**_minimal({"name": "arm", "status": "testable_on_real_b0"}))


def test_tags_status_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match=r"tags\.status is retired"):
        TrainingSettings(**_minimal({"name": "arm", "tags": {"status": "ready"}}))
