"""Planted violations for ``strategy_class_matches_training_mode``."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.infrastructure.validation.witness.checks import strategy_override_checks as soc
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


class _Generic:  # stands in for ReconstructionTrainingStrategy
    pass


_Generic.__name__ = "ReconstructionTrainingStrategy"


class _Mechanism:
    pass


class _Specialised(_Mechanism):
    pass


class _Sibling:
    pass


def test_generic_base_under_a_mechanism_mode_is_the_facade_shape() -> None:
    message = soc.strategy_override_defect(_Generic, _Mechanism, None)
    assert message is not None and "generic base" in message


def test_a_specialisation_of_the_mapped_class_passes() -> None:
    assert soc.strategy_override_defect(_Specialised, _Mechanism, None) is None


def test_a_sibling_mechanism_declared_by_name_passes() -> None:
    assert soc.strategy_override_defect(_Sibling, _Mechanism, None) is None


def test_same_class_passes() -> None:
    assert soc.strategy_override_defect(_Mechanism, _Mechanism, None) is None


def test_an_override_reason_is_the_documented_escape() -> None:
    assert soc.strategy_override_defect(_Generic, _Mechanism, "runs the loss-only variant") is None


def test_the_witness_reads_the_settings(monkeypatch) -> None:
    monkeypatch.setattr(soc, "_resolve", lambda settings: (_Generic, _Mechanism))
    settings = SimpleNamespace(
        training=SimpleNamespace(strategy_class="x.Generic", training_mode="symplectic_bloch"),
        metadata=SimpleNamespace(tags={}),
    )
    verdict = soc.strategy_class_matches_training_mode(
        WitnessSubject.for_audit("arm.yaml", settings)
    )
    assert verdict.passed is False and verdict.yaml_keys == (
        "training.strategy_class",
        "training.training_mode",
    )
    settings.metadata.tags["strategy_override_reason"] = "documented"
    assert (
        soc.strategy_class_matches_training_mode(
            WitnessSubject.for_audit("arm.yaml", settings)
        ).passed
        is True
    )
