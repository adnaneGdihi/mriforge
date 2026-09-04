"""``validation_metric_names_resolve`` (cohort review 2026-09-02, T0.5). Planted violation first."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.infrastructure.validation.witness.checks import validation_metric_checks as vmc
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

_REGISTERED = {"psnr", "ssim", "hfen"}


def _registered(name: str) -> bool:
    return name in _REGISTERED


def _settings(compute=(), best=None, early=None):
    return SimpleNamespace(
        validation=SimpleNamespace(scoring=SimpleNamespace(compute=list(compute))),
        metrics=SimpleNamespace(best_metric_name=best),
        early_stopping=SimpleNamespace(metric=early),
    )


def _subject(settings, emitted: frozenset[str] | None, monkeypatch) -> WitnessSubject:
    name = "DeclaringStrategy" if emitted else "SilentStrategy"
    monkeypatch.setattr(vmc, "_emitted_for", lambda _s: (emitted or frozenset(), name))
    import spectramr.core.metrics.registry as reg

    monkeypatch.setattr(reg.MetricsRegistry, "is_registered", staticmethod(_registered))
    return WitnessSubject.for_audit(None, settings)


# --- the predicate --------------------------------------------------------------


def test_registered_and_emitted_names_resolve() -> None:
    names = ["psnr", "field_mse", "val_field_abs_bias", "val_psnr_mean", "val_psnr_2x", "val_loss"]
    assert (
        vmc.unresolved_metric_names(
            names, _registered, frozenset({"val_field_mse", "val_field_abs_bias"})
        )
        == []
    )


def test_a_selector_no_path_produces_is_unresolved() -> None:
    """The planted violation: exp_vf_22 once selected on val_nmse and never early-stopped."""
    assert vmc.unresolved_metric_names(["val_nmse"], _registered, frozenset()) == ["val_nmse"]


def test_cascade_and_heldout_suffixes_are_stripped() -> None:
    assert (
        vmc.unresolved_metric_names(
            ["val_hfen_heldout_64x", "val_ssim_median"], _registered, frozenset()
        )
        == []
    )


# --- the witness ----------------------------------------------------------------


def test_declaring_strategy_with_an_ownerless_name_is_an_error(monkeypatch) -> None:
    subject = _subject(
        _settings(compute=["psnr", "field_typo"]), frozenset({"val_field_mse"}), monkeypatch
    )
    verdict = vmc.validation_metric_names_resolve(subject)
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert "field_typo" in verdict.message


def test_declaring_strategy_with_resolved_names_passes(monkeypatch) -> None:
    subject = _subject(
        _settings(compute=["field_bias", "field_mse"], best="val_field_abs_bias"),
        frozenset({"val_field_bias", "val_field_mse", "val_field_abs_bias"}),
        monkeypatch,
    )
    assert vmc.validation_metric_names_resolve(subject).passed is True


def test_silent_strategy_is_reported_unverified_not_failed(monkeypatch) -> None:
    """The census step of the ratchet."""
    subject = _subject(_settings(compute=["crlb_objective"]), None, monkeypatch)
    verdict = vmc.validation_metric_names_resolve(subject)
    assert verdict.passed is True and verdict.severity is Severity.INFO
    assert verdict.message.startswith("UNVERIFIED")


def test_no_names_declared_is_a_pass(monkeypatch) -> None:
    assert (
        vmc.validation_metric_names_resolve(_subject(_settings(), None, monkeypatch)).passed is True
    )


def test_the_qmri_strategy_declares_the_keys_its_arms_name() -> None:
    """Pins the declaration the 10 vf qMRI arms rely on, at its source."""
    from spectramr.infrastructure.training.strategies.multi_acquisition_strategy import (
        ConcreteMultiAcquisitionStrategy,
    )

    emitted = ConcreteMultiAcquisitionStrategy.capabilities.emitted_metrics
    assert {"val_field_mse", "val_field_bias", "val_field_abs_bias", "val_field_mae"} <= emitted


def test_witness_is_registered_after_discovery() -> None:
    import spectramr.infrastructure.validation.witness  # noqa: F401

    assert get_witness_registry().get("validation_metric_names_resolve") is not None
