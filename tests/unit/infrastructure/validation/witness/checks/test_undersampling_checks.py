"""``undersampling_block_is_applied`` (cohort review 2026-09-02, T0.6). Planted violation first."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.validation.witness.checks import undersampling_checks as uc
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


class _Masking:
    applies_undersampling = True


class _Plain:
    applies_undersampling = False


def _settings(
    *,
    dataset_type="nifti_paired",
    trajectory=None,
    image_undersampling=False,
    dynamic=False,
    twin=False,
    block=True,
):
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_type=dataset_type,
            trajectory=trajectory,
            image_undersampling=image_undersampling,
        ),
        undersampling=SimpleNamespace(base_acceleration=4.0, enable_dynamic_mask=dynamic)
        if block
        else None,
        physics=SimpleNamespace(digital_twin=SimpleNamespace(apply_as_transform=twin)),
    )


def _subject(settings, strategy_cls, monkeypatch, raw_block=None) -> WitnessSubject:
    monkeypatch.setattr(uc, "_resolve_strategy", lambda _s: strategy_cls)
    subject = WitnessSubject.for_audit(None, settings)
    raw = {"undersampling": {"base_acceleration": 4.0} if raw_block is None else raw_block}
    if getattr(settings, "undersampling", None) is None:
        raw = {}
    subject.raw_config.update(raw)
    return subject


def test_image_domain_block_with_no_consumer_is_an_error(monkeypatch) -> None:
    """The planted violation: hilbert_mamba / mrf_2026 / fmri_2026 on 2026-09-02."""
    verdict = uc.undersampling_block_is_applied(_subject(_settings(), _Plain, monkeypatch))
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert "nothing applies it" in verdict.message


@pytest.mark.parametrize(
    "kw",
    [
        {"trajectory": "cartesian"},
        {"image_undersampling": True},
        {"dynamic": True},
        {"twin": True},
    ],
)
def test_each_declared_route_counts_as_a_consumer(monkeypatch, kw) -> None:
    verdict = uc.undersampling_block_is_applied(_subject(_settings(**kw), _Plain, monkeypatch))
    assert verdict.passed is True and "applied via" in verdict.message


def test_a_masking_strategy_counts_as_a_consumer(monkeypatch) -> None:
    verdict = uc.undersampling_block_is_applied(_subject(_settings(), _Masking, monkeypatch))
    assert verdict.passed is True and "_Masking.applies_undersampling" in verdict.message


def test_kspace_domain_without_a_route_is_reported_unverified_not_failed(monkeypatch) -> None:
    """The census step of the ratchet: a k-space arm on an undeclared strategy."""
    verdict = uc.undersampling_block_is_applied(
        _subject(_settings(dataset_type="kspace"), _Plain, monkeypatch)
    )
    assert verdict.passed is True and verdict.severity is Severity.INFO
    assert verdict.message.startswith("UNVERIFIED")


def test_no_block_is_a_pass(monkeypatch) -> None:
    verdict = uc.undersampling_block_is_applied(
        _subject(_settings(block=False), _Plain, monkeypatch)
    )
    assert verdict.passed is True


def test_masking_strategies_declare_the_flag() -> None:
    """The declarations the witness relies on, pinned at their source."""
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
    from spectramr.infrastructure.training.strategies.loupe_strategy import LOUPEStrategy
    from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    assert BaseTrainingStrategy.applies_undersampling is False
    # the mixin is universal (every strategy has it) and masks nothing by itself
    assert "applies_undersampling" not in KspaceMixin.__dict__
    assert ReconstructionTrainingStrategy.applies_undersampling is False
    assert DiffusionTrainingStrategy.applies_undersampling is True
    assert LOUPEStrategy.applies_undersampling is True  # reads undersampling.adaptive


def test_witness_is_registered_after_discovery() -> None:
    import spectramr.infrastructure.validation.witness  # noqa: F401

    assert get_witness_registry().get("undersampling_block_is_applied") is not None


def test_an_empty_raw_block_advertises_nothing(monkeypatch) -> None:
    """``undersampling: {}`` (17 vf / vf_ulf arms): defaults fill in, nothing is declared."""
    verdict = uc.undersampling_block_is_applied(
        _subject(_settings(), _Plain, monkeypatch, raw_block={})
    )
    assert verdict.passed is True and "empty" in verdict.message


def test_a_masking_mixin_behind_a_non_masking_base_still_counts(monkeypatch) -> None:
    """MRO trap: BaseTrainingStrategy declares False before KspaceMixin's True."""

    class _Base:
        applies_undersampling = False

    class _Mixin:
        applies_undersampling = True

    class _Strategy(_Base, _Mixin):
        pass

    assert _Strategy.applies_undersampling is False  # the plain read that hid it
    consumers = uc.undersampling_consumers(_settings(), _Strategy)
    assert consumers and "_Mixin.applies_undersampling" in consumers[0]


def test_the_fully_sampled_declaration_has_nothing_to_apply(monkeypatch) -> None:
    settings = _settings()
    settings.undersampling.base_acceleration = 1.0
    settings.undersampling.declares_no_acceleration = True
    verdict = uc.undersampling_block_is_applied(_subject(settings, _Plain, monkeypatch))
    assert verdict.passed is True and "declare no acceleration" in verdict.message


def test_base_one_alone_is_not_a_declaration(monkeypatch) -> None:
    """Planted violation: base 1.0 with the default max is a 1x-to-8x ladder."""
    settings = _settings()
    settings.undersampling.base_acceleration = 1.0
    settings.undersampling.declares_no_acceleration = False
    verdict = uc.undersampling_block_is_applied(_subject(settings, _Plain, monkeypatch))
    assert verdict.passed is False


def test_twin_driven_strategies_do_not_claim_the_block() -> None:
    """VF review 2026-09-03: the twin undersamples at its own ``acceleration``; the
    top-level block reaches nothing on these strategies, so a block on their arms
    must be reported, not blessed by a class constant."""
    from spectramr.infrastructure.training.strategies.ib_vf_strategy import IBVFTrainingStrategy
    from spectramr.infrastructure.training.strategies.vf_admm_strategy import (
        ConcreteVFADMMStrategy,
    )
    from spectramr.infrastructure.training.strategies.virtual_fiducial_strategy import (
        ConcreteVirtualFiducialStrategy,
    )

    for cls in (ConcreteVirtualFiducialStrategy, ConcreteVFADMMStrategy, IBVFTrainingStrategy):
        assert cls.applies_undersampling is False
        assert "applies_undersampling" not in cls.__dict__
        assert uc.undersampling_consumers(_settings(dataset_type="kspace"), cls) == []
