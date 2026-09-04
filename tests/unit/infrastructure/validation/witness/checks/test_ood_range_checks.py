"""Planted violations for ``ood_acceleration_range_is_read`` (VF review 2026-09-03)."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.infrastructure.validation.witness.checks import ood_range_checks as orc
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


class _Reader:
    reads_ood_acceleration_range = True


class _ReaderSubclass(_Reader):
    pass


class _Plain:
    pass


def _settings(rng=(16.0, 32.0), *, enabled=True, undersampling=True):
    return SimpleNamespace(
        physics=SimpleNamespace(
            digital_twin=SimpleNamespace(
                enabled=enabled,
                enable_undersampling=undersampling,
                acceleration=4.0,
                ood_acceleration_range=list(rng) if rng is not None else None,
            )
        )
    )


def _subject(settings, strategy_cls, monkeypatch) -> WitnessSubject:
    monkeypatch.setattr(orc, "_resolve_strategy", lambda _s: strategy_cls)
    return WitnessSubject.for_audit(None, settings)


def test_a_range_on_a_strategy_that_does_not_read_it_is_an_error(monkeypatch) -> None:
    """Planted violation: the two cold-diffusion and two DDPM baseline arms."""
    verdict = orc.ood_acceleration_range_is_read(_subject(_settings(), _Plain, monkeypatch))
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert "does not read it" in verdict.message
    assert verdict.yaml_keys == (orc.KEY,)


def test_the_intra_block_conditions_belong_to_the_schema_not_the_witness(monkeypatch) -> None:
    """One owner (non-negotiable 17): a twin that is disabled or does not undersample
    fails at load in ``DigitalTwinConfig``; the witness does not re-ask."""
    for settings in (_settings(undersampling=False), _settings(enabled=False)):
        assert orc.ood_range_defect(settings, _Reader) is None


def test_an_unresolved_strategy_is_an_error_not_a_pass(monkeypatch) -> None:
    verdict = orc.ood_acceleration_range_is_read(_subject(_settings(), None, monkeypatch))
    assert verdict.passed is False and "could not be resolved" in verdict.message


def test_a_reader_with_the_twin_undersampling_passes(monkeypatch) -> None:
    verdict = orc.ood_acceleration_range_is_read(_subject(_settings(), _Reader, monkeypatch))
    assert verdict.passed is True and "2 rung(s)" in verdict.message and "4.0x" in verdict.message


def test_the_reader_flag_is_inherited(monkeypatch) -> None:
    """MRO scan: a subclass of a reader (ib_vf) is a reader."""
    verdict = orc.ood_acceleration_range_is_read(
        _subject(_settings(), _ReaderSubclass, monkeypatch)
    )
    assert verdict.passed is True


def test_no_range_is_a_pass_without_resolving_the_strategy(monkeypatch) -> None:
    monkeypatch.setattr(orc, "_resolve_strategy", lambda _s: (_ for _ in ()).throw(AssertionError))
    verdict = orc.ood_acceleration_range_is_read(WitnessSubject.for_audit(None, _settings(None)))
    assert verdict.passed is True and verdict.severity is Severity.INFO


def test_the_readers_declare_the_flag_at_their_source() -> None:
    from spectramr.infrastructure.training.strategies.ib_vf_strategy import IBVFTrainingStrategy
    from spectramr.infrastructure.training.strategies.vf_admm_strategy import (
        ConcreteVFADMMStrategy,
    )
    from spectramr.infrastructure.training.strategies.virtual_fiducial_strategy import (
        ConcreteVirtualFiducialStrategy,
    )

    assert orc._strategy_reads_it(ConcreteVirtualFiducialStrategy)
    assert orc._strategy_reads_it(ConcreteVFADMMStrategy)
    assert orc._strategy_reads_it(IBVFTrainingStrategy)


def test_registered_on_the_ladder() -> None:
    spec = get_witness_registry().get("ood_acceleration_range_is_read")
    assert spec is not None and spec.severity is Severity.ERROR
