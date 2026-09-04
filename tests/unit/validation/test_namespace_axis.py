"""Phase 0 regression tests — ``check_namespace_axis``.

The deletion ledger (commit d8ccb8452) found training_mode, dataset-type,
strategy, and block tokens misfiled as ``model.model_type``. This check
flags exactly those cross-axis reuses with a precise fix hint, while never
blocking a string that legitimately resolves as a model.
See TODO/deleted_model_types_reimplementation_plan.md Phase 0.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.validation.config_health_checker import (  # noqa: E402
    ConfigHealthChecker,
)


def _settings(model_type) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(model_type=model_type))


@pytest.mark.registry_contract
def test_training_mode_token_rejected() -> None:
    checker = ConfigHealthChecker()
    res = checker.check_namespace_axis(_settings("gan"))
    assert not res.passed
    assert res.severity == "error"
    assert "training.training_mode" in res.fix_hint


@pytest.mark.registry_contract
def test_dataset_type_token_rejected() -> None:
    checker = ConfigHealthChecker()
    res = checker.check_namespace_axis(_settings("dicom_series"))
    # Only asserts the axis-aware behaviour: if dicom_series is a real
    # DatasetType value it is flagged with a data.dataset_type hint.
    if not res.passed:
        assert "data.dataset_type" in res.fix_hint
    else:
        # If the enum doesn't carry that value in this build, the check
        # must at least not crash and must return an info pass.
        assert res.severity == "info"


@pytest.mark.registry_contract
def test_real_model_is_accepted() -> None:
    checker = ConfigHealthChecker()
    res = checker.check_namespace_axis(_settings("unet"))
    assert res.passed
    assert res.severity == "info"


@pytest.mark.registry_contract
def test_unknown_nonforeign_token_is_deferred() -> None:
    """A name that is neither a model nor a foreign-axis token passes here
    (membership failure is owned by check_model_registry)."""
    checker = ConfigHealthChecker()
    res = checker.check_namespace_axis(_settings("totally_unknown_token_xyz"))
    assert res.passed
    assert res.severity == "info"


@pytest.mark.registry_contract
def test_empty_model_type_is_skipped() -> None:
    checker = ConfigHealthChecker()
    res = checker.check_namespace_axis(_settings(""))
    assert res.passed
