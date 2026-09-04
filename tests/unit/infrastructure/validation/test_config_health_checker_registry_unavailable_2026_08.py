"""D14#3: an unavailable registry is a failed check, not a passed one.

Both model checks used to return ``passed=True, severity="info"`` when the
registry could not be loaded. ``audit`` renders only non-passing results, so an
import regression in ``ModelFactory`` silently converted the audit's single
model-existence gate into a no-op: a typo'd ``model_type`` cleared pre-flight
and failed later, at build time, on the cluster.

Measurement behind the polarity flip (the corpus is irrelevant here -- the
degrade is environment-conditional, not config-conditional): in the environment
``spectramr audit`` runs in, ``_lazy_load_registries`` loads **332** model names
and **206** strategy keys, and ``_model_registry`` is emptied only by an
``ImportError``/``AttributeError`` that is already logged as a warning. So these
results now fire on a genuine import regression and nothing else -- which
``test_registry_is_actually_available_here`` pins.
"""

import types

import pytest

from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


def _config(model_type: str = "unet"):
    return types.SimpleNamespace(model=types.SimpleNamespace(model_type=model_type))


def test_registry_is_actually_available_here():
    """The premise of the flip: nothing in a normal environment trips it."""
    checker = ConfigHealthChecker()
    checker._lazy_load_registries()
    assert checker._model_registry, "model registry empty — the flip below would fire"
    assert checker._strategy_registry


def test_empty_model_registry_is_an_error_not_a_skip():
    checker = ConfigHealthChecker()
    checker._model_registry = set()  # short-circuits the lazy load

    result = checker.check_model_registry(_config())

    assert result.passed is False
    assert result.severity == "error"
    assert "registry unavailable" in result.message


def test_missing_model_type_still_reported_before_the_registry_check():
    """Polarity of the pre-existing branch is untouched."""
    checker = ConfigHealthChecker()
    checker._model_registry = set()

    result = checker.check_model_registry(_config(model_type=None))
    assert result.passed is False
    assert "model_type not specified" in result.message


def test_populated_registry_still_passes_for_a_registered_model():
    checker = ConfigHealthChecker()
    checker._model_registry = {"unet"}

    result = checker.check_model_registry(_config("unet"))
    assert result.passed is True
    assert result.severity == "info"


def test_populated_registry_still_fails_for_an_unregistered_model():
    checker = ConfigHealthChecker()
    checker._model_registry = {"unet"}

    result = checker.check_model_registry(_config("not_a_model"))
    assert result.passed is False
    assert result.severity == "error"


def test_resolve_check_reports_an_unimportable_registry_as_error(monkeypatch):
    import spectramr.models.init_registry as init_registry
    import spectramr.models.registry as registry

    monkeypatch.setattr(registry, "MODEL_REGISTRY", {})

    def _boom():
        raise ImportError("simulated registry import regression")

    monkeypatch.setattr(init_registry, "populate_model_registry", _boom)

    result = ConfigHealthChecker().check_registered_model_resolves(_config())

    assert result.passed is False
    assert result.severity == "error"
    assert "could NOT run" in result.message


def test_resolve_check_without_a_model_type_is_still_a_clean_skip():
    """Nothing to resolve is genuinely nothing to report -- unchanged."""
    result = ConfigHealthChecker().check_registered_model_resolves(_config(None))
    assert result.passed is True
    assert result.severity == "info"


@pytest.mark.parametrize("check", ["check_model_registry", "check_registered_model_resolves"])
def test_neither_check_can_pass_while_saying_unavailable(check):
    """The shape the fix targets: a message that admits the check did not run
    paired with ``passed=True``."""
    checker = ConfigHealthChecker()
    checker._model_registry = set()
    result = getattr(checker, check)(_config())
    if "unavailable" in result.message or "NOT run" in result.message:
        assert result.passed is False
