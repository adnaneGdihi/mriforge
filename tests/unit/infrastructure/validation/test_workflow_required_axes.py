"""Tests for check_workflow_required_axes — the pitfall-#19 killer."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.config.schemas.enums import Regime
from spectramr.config.schemas.workflow import WorkflowConfigSchema
from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


def _canonical(dataset_type: str) -> str:
    """Reject a ``dataset_type`` no validated config could ever hold.

    The checker reads ``config.data.dataset_type`` *after*
    ``DataConfigSchema.validate_dataset_type`` has lower-cased it, mapped aliases
    onto canonical names and rejected the rest. Reaching the checker through a
    ``SimpleNamespace`` skips all of that, so a test can assert on a string the
    production path cannot produce — which is exactly how this file's
    ``"2d"`` case stayed green against a ``DATASET_TYPE_AXES`` full of dead
    keys. ``"2d"`` is an alias normalised to ``"image"`` before the check ever
    runs. Routing every case through the real validator keeps these tests on the
    seam.
    """
    normalised = DataConfigSchema.validate_dataset_type(dataset_type)
    assert normalised == dataset_type, (
        f"{dataset_type!r} is not a canonical dataset_type: the validator "
        f"normalises it to {normalised!r}, so no loaded config reaches "
        f"check_workflow_required_axes carrying {dataset_type!r}."
    )
    return dataset_type


def _check(regime: Regime | None, dataset_type: str | None):
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    workflow = WorkflowConfigSchema(regime=regime) if regime is not None else None
    cfg = SimpleNamespace(
        workflow=workflow,
        data=SimpleNamespace(
            dataset_type=_canonical(dataset_type) if dataset_type else dataset_type
        ),
    )
    return checker.check_workflow_required_axes(cfg)


def test_functional_on_structural_data_errors() -> None:
    # mri_functional requires TEMPORAL; dataset_type "image" exposes none.
    r = _check(Regime.FUNCTIONAL, "image")
    assert not r.passed
    assert r.severity == "error"
    assert "temporal" in r.message


def test_quantitative_needs_echo_on_image_data_errors() -> None:
    # mri_quantitative requires ECHO; "image" is annotated as exposing none.
    # Was written against "2d", which the validator normalises to "image".
    r = _check(Regime.QUANTITATIVE, "image")
    assert not r.passed
    assert "echo" in r.message


def test_structural_on_structural_data_passes() -> None:
    # mri_structural requires no non-spatial axes.
    r = _check(Regime.STRUCTURAL, "image")
    assert r.passed


def test_unannotated_dataset_type_is_skipped() -> None:
    # "synthetic" is deliberately not in the axis table → skip, never guess.
    r = _check(Regime.FUNCTIONAL, "synthetic")
    assert r.passed
    assert "skipped" in r.message


def test_no_workflow_is_skipped() -> None:
    r = _check(None, "image")
    assert r.passed


# ── the DECLARED route: per-arm bart_dim_map, not a per-type annotation ──────
#
# Until 2026-08-05 every bart_kspace arm SKIPPED this check: exposed_axes_for is
# keyed on dataset_type, bart_kspace has no row (its arms disagree), so the one
# rule whose job is to consume a declared axis never saw the eight arms that were
# declaring one. These two tests pin that it now fires in BOTH directions — a
# mechanism proven only to accept is a mechanism that has not been shown to work.

_ECHO_ARM_DIM_MAP = {"readout": 1, "spoke": 2, "coil": 3, "echo": 6, "slice": 10, "repetition": 13}
_FLIP_ARM_DIM_MAP = {"readout": 0, "phase": 1, "coil": 3, "flip": 11}
#: Neither echo nor flip: a map that varies NO physical parameter. This is what
#: keeps the rule honest now that two axes satisfy it (#1020).
_NO_PARAMETER_AXIS_DIM_MAP = {"readout": 0, "phase": 1, "coil": 3}


def _check_bart(regime: Regime, dim_map: dict[str, int], *, enabled: bool = True):
    """Drive the check with a REAL validated DataConfigSchema.

    Not a SimpleNamespace: ``declared_axes_for`` reads ``data.bart``, which is
    exactly the attribute a hand-rolled stub would fake into existence, and
    BartConfigSchema's own validation (non-empty map, known roles, no duplicate
    slots) is part of what makes the declaration trustworthy.
    """
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    cfg = SimpleNamespace(
        workflow=WorkflowConfigSchema(regime=regime),
        data=DataConfigSchema(
            dataset_type="bart_kspace",
            bart={
                "enabled": enabled,
                "bart_dim_map": dim_map,
                "sampling": "radial",
                "trajectory_source": "golden_angle",
            },
        ),
    )
    return checker.check_workflow_required_axes(cfg)


def test_quantitative_accepts_a_declared_echo_axis() -> None:
    """The 5 dual-echo B0 arms' shape: mri_quantitative is now admissible on it."""
    r = _check_bart(Regime.QUANTITATIVE, _ECHO_ARM_DIM_MAP)
    assert r.passed
    assert "bart_dim_map" in r.message, (
        "the message must say the answer came from the DECLARATION, not the "
        f"dataset_type table; got: {r.message}"
    )


def test_quantitative_accepts_a_declared_flip_axis() -> None:
    """The 3 double-angle B1 arms' shape. This assertion is INVERTED (#1020).

    It previously asserted refusal, on the reasoning that ``flip`` "is a real
    declared axis with no ``Axis`` member, so it must not be silently counted as
    satisfying anything". The premise was right and the conclusion has been
    overtaken: ``flip`` is no longer memberless, and it is not counted silently
    -- ``Axis.FLIP_ANGLE`` exists and ``mri_quantitative`` names it explicitly.

    A B1+ transmit map is a physical parameter map measured by varying the flip
    angle, exactly as a B0 map is measured by varying the echo time. Refusing it
    made three unambiguously quantitative arms undeclarable while the only other
    option, ``mri_structural``, would have been false science.
    """
    r = _check_bart(Regime.QUANTITATIVE, _FLIP_ARM_DIM_MAP)
    assert r.passed
    assert "flip_angle" in r.message
    assert "echo" not in r.message.split("needs any of")[0], (
        "the message must name what SATISFIED the rule, not the whole required "
        f"set -- this arm declares no echo axis; got: {r.message}"
    )


def test_quantitative_still_rejects_a_map_that_varies_no_parameter() -> None:
    """The teeth the inverted test above used to provide, kept intact.

    Widening a requirement is how a rule quietly stops rejecting anything. An
    acquisition with neither echo nor flip encoding varies no physical
    parameter, so it cannot support a parameter-mapping regime, and must still
    be refused.
    """
    r = _check_bart(Regime.QUANTITATIVE, _NO_PARAMETER_AXIS_DIM_MAP)
    assert not r.passed
    assert r.severity == "error"
    assert "echo" in r.message and "flip_angle" in r.message


def test_any_of_semantics_matches_the_message_the_check_prints() -> None:
    """``required_axes`` is at-least-one-of, and the message always said so.

    The check computed ``required - exposed`` (all-of) while printing "expose
    none of them" (any-of). The two agreed only while every profile declared 0
    or 1 axis -- true of all 14 until ``mri_quantitative`` needed both spellings.
    """
    from spectramr.domain.workflows import WORKFLOW_PROFILES

    required = WORKFLOW_PROFILES[Regime.QUANTITATIVE].required_axes
    assert len(required) == 2, "precondition: the only multi-axis profile"
    # Each alternative satisfies it ALONE, which all-of semantics would refuse.
    assert _check_bart(Regime.QUANTITATIVE, _ECHO_ARM_DIM_MAP).passed
    assert _check_bart(Regime.QUANTITATIVE, _FLIP_ARM_DIM_MAP).passed


def test_the_declaration_is_what_answers_not_the_type_table() -> None:
    """``bart_kspace`` is unannotated, so a pass can only have come from the map."""
    from spectramr.data.datasets.axis_exposure import exposed_axes_for

    assert exposed_axes_for("bart_kspace") is None
    assert _check_bart(Regime.QUANTITATIVE, _ECHO_ARM_DIM_MAP).passed


def test_disabled_bart_falls_back_to_the_skip() -> None:
    """Composition must not turn the historical SKIP into a reject."""
    r = _check_bart(Regime.QUANTITATIVE, _ECHO_ARM_DIM_MAP, enabled=False)
    assert r.passed
    assert "skipped" in r.message
