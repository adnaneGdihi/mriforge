"""Phase 0 regression tests — ``check_registered_model_resolves``.

Guards the re-introduction of the deletion-ledger failure mode: an
advertised ``model_type`` that passes Tier-0 (membership) but maps to a
non-buildable class (abstract base / no ``forward`` / missing class
object). See TODO/deleted_model_types_reimplementation_plan.md Phase 0.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.validation.config_health_checker import (  # noqa: E402
    ConfigHealthChecker,
)


def _settings(model_type) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(model_type=model_type))


@pytest.mark.registry_contract
def test_concrete_registered_model_passes() -> None:
    checker = ConfigHealthChecker()
    res = checker.check_registered_model_resolves(_settings("unet"))
    assert res.passed, res.message


@pytest.mark.registry_contract
def test_wave2_alias_resolves() -> None:
    """A restored alias (vq_vae -> vqvae) resolves to a concrete class."""
    checker = ConfigHealthChecker()
    res = checker.check_registered_model_resolves(_settings("vq_vae"))
    assert res.passed, res.message


@pytest.mark.registry_contract
def test_empty_model_type_is_skipped() -> None:
    checker = ConfigHealthChecker()
    res = checker.check_registered_model_resolves(_settings(None))
    assert res.passed
    assert res.severity == "info"


@pytest.mark.registry_contract
def test_unknown_name_deferred_to_membership_check() -> None:
    """A name absent from the decorator registry is deferred (not failed
    here) because check_model_registry owns membership."""
    checker = ConfigHealthChecker()
    res = checker.check_registered_model_resolves(
        _settings("definitely_not_a_real_model_xyz")
    )
    assert res.passed
    assert res.severity == "info"


@pytest.mark.registry_contract
def test_abstract_class_is_rejected(monkeypatch) -> None:
    """A registry entry pointing at an abstract class fails the resolve check."""
    import abc

    from mriforge.models import registry as reg

    class _AbstractModel(abc.ABC):
        @abc.abstractmethod
        def forward(self, x):  # pragma: no cover - never instantiated
            ...

    reg.MODEL_REGISTRY["__abstract_probe__"] = {
        "class": _AbstractModel,
        "mode": "reconstruction",
    }
    try:
        checker = ConfigHealthChecker()
        res = checker.check_registered_model_resolves(_settings("__abstract_probe__"))
        assert not res.passed
        assert res.severity == "error"
        assert "abstract" in res.message.lower()
    finally:
        reg.MODEL_REGISTRY.pop("__abstract_probe__", None)
