"""Runtime maturity enforcement — STUB/EVAL_ONLY raise from the right pipelines."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mriforge.config.schemas.enums import Maturity, Regime
from mriforge.config.schemas.workflow import WorkflowConfigSchema
from mriforge.domain.exceptions import WorkflowNotImplementedError
from mriforge.domain.workflows import (
    WORKFLOW_PROFILES,
    enforce_pipeline_maturity,
    enforce_pipeline_maturity_for_config,
)


def test_stub_raises_from_every_pipeline() -> None:
    for pipeline in ("train", "evaluate", "predict"):
        with pytest.raises(WorkflowNotImplementedError):
            enforce_pipeline_maturity(Regime.CT, pipeline)  # type: ignore[arg-type]


def test_eval_only_raises_only_from_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """EVAL_ONLY blocks train but permits evaluate/predict.

    Tested against a SYNTHETIC profile rather than whichever regime happens to
    sit at EVAL_ONLY today. This test used to pick `mri_spectroscopy` — and when
    that regime was completed to LIVE (2026-07-16) the test broke, because
    completing a regime had deleted the only fixture for a rule that still needs
    guarding. EVAL_ONLY is now empty by design
    (`test_maturity_ledger.py::test_no_mr_regime_with_real_physics_is_left_at_eval_only`),
    so there is no live regime to borrow.

    The rule is a property of the CODE, not of the ledger's current contents, and
    the test should say so.
    """
    import dataclasses

    from mriforge.domain.workflows import profiles as profiles_module

    live = WORKFLOW_PROFILES[Regime.SPECTROSCOPIC]
    eval_only = dataclasses.replace(live, maturity=Maturity.EVAL_ONLY)
    monkeypatch.setitem(
        profiles_module.WORKFLOW_PROFILES, Regime.SPECTROSCOPIC, eval_only
    )

    with pytest.raises(WorkflowNotImplementedError, match="EVAL_ONLY"):
        enforce_pipeline_maturity(Regime.SPECTROSCOPIC, "train")
    enforce_pipeline_maturity(Regime.SPECTROSCOPIC, "evaluate")
    enforce_pipeline_maturity(Regime.SPECTROSCOPIC, "predict")


def test_spectroscopy_can_now_be_trained() -> None:
    """The behavioural consequence of completing the last EVAL_ONLY regime.

    Paired with the synthetic test above: that one proves the RULE still works,
    this one proves this regime is no longer subject to it.
    """
    for pipeline in ("train", "evaluate", "predict"):
        enforce_pipeline_maturity(Regime.SPECTROSCOPIC, pipeline)  # type: ignore[arg-type]


def test_live_and_partial_never_raise() -> None:
    for regime in (Regime.STRUCTURAL, Regime.QUANTITATIVE):
        for pipeline in ("train", "evaluate", "predict"):
            enforce_pipeline_maturity(regime, pipeline)  # type: ignore[arg-type]


def test_none_regime_is_noop() -> None:
    # Arms predating the workflow: block declare no regime — never blocked here
    # (that is the audit's job, not the runtime's).
    enforce_pipeline_maturity(None, "train")


def test_for_config_reads_workflow_name() -> None:
    stub_cfg = SimpleNamespace(workflow=WorkflowConfigSchema(regime=Regime.ULTRASOUND))
    with pytest.raises(WorkflowNotImplementedError):
        enforce_pipeline_maturity_for_config(stub_cfg, "train")

    live_cfg = SimpleNamespace(workflow=WorkflowConfigSchema(regime=Regime.STRUCTURAL))
    enforce_pipeline_maturity_for_config(live_cfg, "train")

    # Missing workflow attr / None → no-op
    enforce_pipeline_maturity_for_config(SimpleNamespace(workflow=None), "train")
    enforce_pipeline_maturity_for_config(SimpleNamespace(), "train")
