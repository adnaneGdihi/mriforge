"""Planted violations for ``image_losses_reach_the_objective`` (mrixfields review 2026-09-03)."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.infrastructure.validation.witness.checks import loss_ownership_checks as loc
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


class _ScoreMatching:
    """Owns its objective, computes no image loss, folds nothing (doob_bridge's shape)."""

    inline_losses = frozenset()
    folds_image_losses = False


class _InlineL1:
    inline_losses = frozenset({"l1"})
    folds_image_losses = False


class _Folder:
    inline_losses = frozenset({"l1"})
    folds_image_losses = True


class _Undeclared:
    pass


def _settings(names):
    return SimpleNamespace(
        losses=SimpleNamespace(image_losses=[{"name": n, "weight": 1.0} for n in names])
    )


def _subject(settings, strategy_cls, monkeypatch) -> WitnessSubject:
    monkeypatch.setattr(loc, "_resolve_strategy", lambda _s: strategy_cls)
    return WitnessSubject.for_audit(None, settings)


def test_a_decoy_l1_on_a_score_matching_strategy_is_an_error(monkeypatch) -> None:
    """Planted violation: 26 mrixfields arms on 2026-09-03."""
    verdict = loc.image_losses_reach_the_objective(
        _subject(_settings(["l1"]), _ScoreMatching, monkeypatch)
    )
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert "['l1']" in verdict.message and "folds nothing" in verdict.message
    assert verdict.yaml_keys == (loc.KEY,)


def test_an_extra_entry_on_a_non_folding_inline_strategy_is_an_error(monkeypatch) -> None:
    """Planted violation: ``[l1, ssim]`` on a strategy that computes l1 and folds nothing."""
    verdict = loc.image_losses_reach_the_objective(
        _subject(_settings(["l1", "ssim"]), _InlineL1, monkeypatch)
    )
    assert verdict.passed is False and "['ssim']" in verdict.message


def test_an_unregistered_name_on_a_folder_is_an_error(monkeypatch) -> None:
    verdict = loc.image_losses_reach_the_objective(
        _subject(_settings(["l1", "no_such_loss_xyz"]), _Folder, monkeypatch)
    )
    assert verdict.passed is False and "no_such_loss_xyz" in verdict.message


def test_inline_and_folded_registered_entries_pass(monkeypatch) -> None:
    verdict = loc.image_losses_reach_the_objective(
        _subject(_settings(["l1", "hfen", "ms_ssim"]), _Folder, monkeypatch)
    )
    assert verdict.passed is True and "all 3 declared" in verdict.message


def test_an_alias_of_an_inline_term_counts_as_inline(monkeypatch) -> None:
    """``mae`` canonicalises to ``l1``."""
    verdict = loc.image_losses_reach_the_objective(
        _subject(_settings(["mae"]), _InlineL1, monkeypatch)
    )
    assert verdict.passed is True


def test_an_undeclared_strategy_is_reported_unverified_not_passed(monkeypatch) -> None:
    """The ratchet's census step."""
    verdict = loc.image_losses_reach_the_objective(
        _subject(_settings(["l1"]), _Undeclared, monkeypatch)
    )
    assert verdict.passed is True and verdict.severity is Severity.INFO
    assert verdict.message.startswith("UNVERIFIED")


def test_no_declared_losses_is_a_pass_without_resolving(monkeypatch) -> None:
    monkeypatch.setattr(loc, "_resolve_strategy", lambda _s: (_ for _ in ()).throw(AssertionError))
    verdict = loc.image_losses_reach_the_objective(WitnessSubject.for_audit(None, _settings([])))
    assert verdict.passed is True


def test_registered_on_the_ladder() -> None:
    spec = get_witness_registry().get("image_losses_reach_the_objective")
    assert spec is not None and spec.severity is Severity.ERROR
