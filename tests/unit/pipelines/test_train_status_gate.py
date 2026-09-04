"""``metadata.status`` launch gate (cohort review 2026-09-02, T0.4).

Planted violation first: an arm whose metadata says ``needs_implementation``
used to train and report a number nothing could tell from a real result.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.config.schemas.base import EXPERIMENT_STATUSES, LAUNCH_REFUSED_STATUSES
from spectramr.pipelines.train import ExperimentStatusRefusedError, _refuse_unlaunchable_status


def _cfg(status: str | None, reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(metadata=SimpleNamespace(status=status, status_reason=reason))


@pytest.mark.parametrize("status", sorted(LAUNCH_REFUSED_STATUSES))
def test_refused_statuses_do_not_launch(status: str) -> None:
    with pytest.raises(ExperimentStatusRefusedError, match=f"metadata.status={status!r}"):
        _refuse_unlaunchable_status(_cfg(status, "identity Jacobian"))


def test_the_reason_is_named_in_the_refusal() -> None:
    with pytest.raises(ExperimentStatusRefusedError, match="identity Jacobian"):
        _refuse_unlaunchable_status(_cfg("inert", "identity Jacobian"))


@pytest.mark.parametrize("status", sorted(set(EXPERIMENT_STATUSES) - LAUNCH_REFUSED_STATUSES))
def test_launchable_statuses_pass_and_are_returned(status: str) -> None:
    assert _refuse_unlaunchable_status(_cfg(status)) == status


def test_no_status_declared_passes() -> None:
    assert _refuse_unlaunchable_status(_cfg(None)) is None
    assert _refuse_unlaunchable_status(SimpleNamespace(metadata=None)) is None


def test_allow_status_must_name_the_exact_status() -> None:
    """``--allow-status inert`` does not unlock a ``needs_implementation`` arm."""
    assert _refuse_unlaunchable_status(_cfg("inert"), allow_status="inert") == "inert"
    with pytest.raises(ExperimentStatusRefusedError):
        _refuse_unlaunchable_status(_cfg("needs_implementation"), allow_status="inert")


def test_run_training_pipeline_exposes_the_knob() -> None:
    import inspect

    from spectramr.pipelines.train import run_training_pipeline

    assert "allow_status" in inspect.signature(run_training_pipeline).parameters
