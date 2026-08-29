"""Subject laziness: cheap surfaces must not pay for expensive ones."""

from __future__ import annotations

import pytest

from mriforge.infrastructure.validation.witness.registry import Subject
from mriforge.infrastructure.validation.witness.subject import (
    WitnessSubject,
    WitnessSubjectUnavailableError,
)


def test_ci_subject_serves_the_raw_config_without_any_build():
    subject = WitnessSubject.for_ci("a.yaml", {"model": {"model_type": "unet"}})
    assert subject.get(Subject.CONFIG)["model"]["model_type"] == "unet"


def test_missing_kind_raises_and_names_what_was_unavailable():
    """A mis-scheduled witness must fail loudly, never silently pass.

    Returning "nothing to check" here would be a detector that never fires.
    """
    subject = WitnessSubject.for_ci("a.yaml", {})
    with pytest.raises(WitnessSubjectUnavailableError, match="module_tree"):
        subject.get(Subject.MODULE_TREE)


def test_provides_answers_without_invoking_the_factory():
    """Pre-filtering must not trigger the very build it is avoiding."""
    calls = []

    subject = WitnessSubject.for_ci("a.yaml", {})
    subject._factories[Subject.MODULE_TREE] = lambda: calls.append(1)

    assert subject.provides(frozenset({Subject.MODULE_TREE})) is True
    assert subject.provides(frozenset({Subject.TENSOR})) is False
    assert calls == [], "provides() built the subject it was only asked about"


def test_factory_is_invoked_at_most_once():
    calls = []
    subject = WitnessSubject.for_ci("a.yaml", {})
    subject._factories[Subject.MODULE_TREE] = lambda: (calls.append(1), "model")[1]

    assert subject.get(Subject.MODULE_TREE) == "model"
    assert subject.get(Subject.MODULE_TREE) == "model"
    assert len(calls) == 1, "an expensive build ran twice"


def test_none_kind_needs_nothing():
    """Pure-introspection witnesses must run on any surface, config or not."""
    assert WitnessSubject.for_ci(None, {}).get(Subject.NONE) is None
    assert WitnessSubject.for_ci(None, {}).provides(frozenset({Subject.NONE})) is True
