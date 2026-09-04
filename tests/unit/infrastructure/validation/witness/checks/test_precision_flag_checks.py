"""Planted violations for ``no_dead_precision_flag``."""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.checks import precision_flag_checks as pfc
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


def _subject(raw: dict) -> WitnessSubject:
    return WitnessSubject.for_ci("arm.yaml", raw)


def test_the_dead_spelling_is_an_error_even_when_it_agrees() -> None:
    """Planted violation: agreement is luck, not a reader."""
    verdict = pfc.no_dead_precision_flag(
        _subject(
            {
                "training": {"enable_mixed_precision": False},
                "optimization": {"precision": {"enabled": False}},
            }
        )
    )
    assert verdict.passed is False and "agrees with" in verdict.message
    assert verdict.yaml_keys == (pfc.DEAD_FLAG, pfc.LIVE_FLAG)


def test_a_disagreement_names_the_live_value_that_wins() -> None:
    message = pfc.dead_precision_flag_defect(
        {
            "training": {"enable_mixed_precision": False},
            "optimization": {"precision": {"enabled": True}},
        }
    )
    assert message is not None and "DISAGREES with" in message and "enabled=True" in message


def test_an_arm_without_the_flag_passes() -> None:
    verdict = pfc.no_dead_precision_flag(_subject({"training": {"epochs": 5}}))
    assert verdict.passed is True


def test_no_live_block_means_the_default_false() -> None:
    message = pfc.dead_precision_flag_defect({"training": {"enable_mixed_precision": True}})
    assert message is not None and "DISAGREES with" in message and "enabled=False" in message
