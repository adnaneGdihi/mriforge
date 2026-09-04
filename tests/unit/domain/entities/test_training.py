"""Tests for domain training entities — DomainEvent dataclass chain (WS-3).

Regression: the event subclasses are ``@dataclass`` but ``DomainEvent`` was a
plain class whose hand-written ``__init__`` was never chained, so ``timestamp``
(and, for ``ModelValidatedEvent``, ``event_id``) was never set and ``to_dict()``
raised ``AttributeError``. The base is now ``@dataclass(kw_only=True)``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from spectramr.domain.entities.training import (
    ModelValidatedEvent,
    TrainingCompletedEvent,
    TrainingStartedEvent,
)


def test_training_started_event_to_dict() -> None:
    e = TrainingStartedEvent(
        event_id="evt1", session_id="s1", model_id="m1", config={"epochs": 10}
    )
    assert isinstance(e.timestamp, datetime)
    d = e.to_dict()
    assert d["event_id"] == "evt1"
    assert d["event_type"] == "TrainingStartedEvent"
    assert d["session_id"] == "s1"
    assert d["model_id"] == "m1"
    assert d["config"] == {"epochs": 10}
    assert d["timestamp"] == e.timestamp.isoformat()


def test_training_completed_event_to_dict() -> None:
    e = TrainingCompletedEvent(
        event_id="evt2", session_id="s2", final_metrics={"loss": 0.1}, success=True
    )
    assert isinstance(e.timestamp, datetime)
    d = e.to_dict()
    assert d["success"] is True
    assert d["final_metrics"] == {"loss": 0.1}
    assert d["event_type"] == "TrainingCompletedEvent"


def test_model_validated_event_has_event_id_and_timestamp() -> None:
    # Regression: ModelValidatedEvent previously declared neither event_id nor
    # inherited a working timestamp.
    e = ModelValidatedEvent(
        event_id="evt3", model_id="m3", validation_results={"psnr": 30.0}
    )
    assert e.event_id == "evt3"
    assert isinstance(e.timestamp, datetime)
    d = e.to_dict()
    assert d["event_id"] == "evt3"
    assert d["model_id"] == "m3"
    assert d["validation_results"] == {"psnr": 30.0}


def test_explicit_timestamp_is_respected() -> None:
    ts = datetime(2020, 1, 1)
    e = TrainingStartedEvent(
        event_id="e", session_id="s", model_id="m", config={}, timestamp=ts
    )
    assert e.timestamp == ts


def test_events_are_keyword_only() -> None:
    # kw_only=True: positional construction must fail.
    with pytest.raises(TypeError):
        TrainingStartedEvent("e", "s", "m", {})  # type: ignore[misc]
