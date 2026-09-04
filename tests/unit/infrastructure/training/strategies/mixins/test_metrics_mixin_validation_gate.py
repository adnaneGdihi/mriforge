"""``validation.scoring.compute`` is validated against registry-or-emitted (cohort review T0.5)."""

from __future__ import annotations

import logging

import pytest

from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.models.capabilities import StrategyCapabilities


def _host(emitted: frozenset[str]):
    class _Host(MetricsMixin):
        capabilities = StrategyCapabilities(emitted_metrics=emitted)

    return _Host()


def _registered(name: str) -> bool:
    return name in {"psnr", "ssim", "hfen"}


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    import spectramr.core.metrics.registry as reg

    monkeypatch.setattr(reg.MetricsRegistry, "is_registered", staticmethod(_registered))


def test_emitted_names_are_accepted_and_removed_from_the_computer_list() -> None:
    host = _host(frozenset({"val_field_mse", "val_field_bias"}))
    assert host._registry_backed_validation_metrics(["psnr", "field_mse", "field_bias"]) == ["psnr"]


def test_declaring_strategy_raises_on_an_ownerless_name() -> None:
    """The planted violation: a typo that used to become a missing column."""
    host = _host(frozenset({"val_field_mse"}))
    with pytest.raises(ValueError, match="neither registered nor emitted"):
        host._registry_backed_validation_metrics(["psnr", "field_typo"])


def test_silent_strategy_logs_a_census_line_and_keeps_the_registry_part(caplog) -> None:
    host = _host(frozenset())
    with caplog.at_level(logging.INFO):
        kept = host._registry_backed_validation_metrics(["psnr", "crlb_objective"])
    assert kept == ["psnr"]
    assert any("declares no emitted_metrics" in r.getMessage() for r in caplog.records)


def test_val_loss_is_universally_emitted() -> None:
    host = _host(frozenset({"val_field_mse"}))
    assert host._registry_backed_validation_metrics(["val_loss", "ssim"]) == ["ssim"]
