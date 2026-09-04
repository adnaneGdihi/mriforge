"""Tests for check_knob_applicability (regime-gated inert-knob trap)."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.config.schemas.enums import Regime
from spectramr.config.schemas.workflow import WorkflowConfigSchema
from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


def _check(regime: Regime | None, **blocks):
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    workflow = WorkflowConfigSchema(regime=regime) if regime is not None else None
    fields = {"workflow": workflow, "mrf": None, "data": None}
    fields.update(blocks)
    return checker.check_knob_applicability(SimpleNamespace(**fields))


def _errors(results):
    return [r for r in results if not r.passed]


def test_mrf_under_structural_is_inert_knob() -> None:
    results = _check(Regime.STRUCTURAL, mrf=SimpleNamespace(n_timepoints=100))
    errs = _errors(results)
    assert len(errs) == 1
    assert "mrf" in errs[0].message


def test_mrf_under_fingerprinting_ok() -> None:
    results = _check(Regime.FINGERPRINTING, mrf=SimpleNamespace(n_timepoints=100))
    assert _errors(results) == []


def test_quantitative_maps_under_structural_error() -> None:
    data = SimpleNamespace(quantitative=SimpleNamespace(enabled=True))
    results = _check(Regime.STRUCTURAL, data=data)
    assert len(_errors(results)) == 1


def test_quantitative_maps_disabled_is_ok() -> None:
    data = SimpleNamespace(quantitative=SimpleNamespace(enabled=False))
    results = _check(Regime.STRUCTURAL, data=data)
    assert _errors(results) == []


def test_no_workflow_is_skipped() -> None:
    results = _check(None, mrf=SimpleNamespace(n_timepoints=100))
    assert _errors(results) == []


# ---------------------------------------------------------------------------
# Phase-contrast (mri_flow) and perfusion (mri_perfusion) knob scopes.
# ---------------------------------------------------------------------------


def test_phase_contrast_under_structural_is_inert_knob() -> None:
    """venc/4-point encoding is meaningless outside phase-contrast flow.

    Without the KnobScope this is the quiet failure mode: a structural arm
    declares data.phase_contrast, the audit says nothing, and the block does
    nothing — an advertised knob that never fires (pitfall #15).
    """
    data = SimpleNamespace(
        quantitative=SimpleNamespace(enabled=False),
        phase_contrast=SimpleNamespace(enabled=True),
        perfusion=SimpleNamespace(enabled=False),
    )
    errs = _errors(_check(Regime.STRUCTURAL, data=data))
    assert len(errs) == 1
    assert "data.phase_contrast" in errs[0].message


def test_phase_contrast_under_flow_ok() -> None:
    data = SimpleNamespace(
        quantitative=SimpleNamespace(enabled=False),
        phase_contrast=SimpleNamespace(enabled=True),
        perfusion=SimpleNamespace(enabled=False),
    )
    assert _errors(_check(Regime.FLOW, data=data)) == []


def test_phase_contrast_disabled_is_ok_anywhere() -> None:
    """The default block is inert, so it must never redden an unrelated arm."""
    data = SimpleNamespace(
        quantitative=SimpleNamespace(enabled=False),
        phase_contrast=SimpleNamespace(enabled=False),
        perfusion=SimpleNamespace(enabled=False),
    )
    assert _errors(_check(Regime.STRUCTURAL, data=data)) == []


def test_perfusion_under_flow_is_inert_knob() -> None:
    """The two new blocks are scoped to different regimes and must not overlap."""
    data = SimpleNamespace(
        quantitative=SimpleNamespace(enabled=False),
        phase_contrast=SimpleNamespace(enabled=False),
        perfusion=SimpleNamespace(enabled=True),
    )
    errs = _errors(_check(Regime.FLOW, data=data))
    assert len(errs) == 1
    assert "data.perfusion" in errs[0].message


def test_perfusion_under_perfusion_ok() -> None:
    data = SimpleNamespace(
        quantitative=SimpleNamespace(enabled=False),
        phase_contrast=SimpleNamespace(enabled=False),
        perfusion=SimpleNamespace(enabled=True),
    )
    assert _errors(_check(Regime.PERFUSION, data=data)) == []


def test_every_knob_scope_names_a_real_schema_path() -> None:
    """A KnobScope pointing at a path nothing mounts can never fire.

    DataConfigSchema is extra="ignore", so an unmounted `data.<x>` block is
    silently dropped from the config — the scope would then resolve to None and
    the check would pass vacuously forever. Walk the real schema.
    """
    from spectramr.config.settings import TrainingSettings
    from spectramr.domain.workflows.knobs import KNOB_APPLICABILITY

    for scope in KNOB_APPLICABILITY:
        model = TrainingSettings
        for segment in scope.path.split("."):
            assert segment in model.model_fields, (
                f"KnobScope({scope.path!r}) names {segment!r}, which is not a "
                "field on the schema — the scope could never fire."
            )
            annotation = model.model_fields[segment].annotation
            model = next(
                (
                    arg
                    for arg in getattr(annotation, "__args__", (annotation,))
                    if hasattr(arg, "model_fields")
                ),
                annotation,
            )
