"""Regression tests for the 2026-06 PaDNet-strategy audit fixes.

Covers the BROKEN_STRATEGY_ONLY finding from the 2026-05-31 strategy audit:

- ``tr``/``te`` physics knobs silently defaulting to 500.0/10.0. The dataset
  must supply real sequence params; the Bloch physics loss must never train
  against fabricated TR/TE (pitfall #1/#15). The fix replaces the hardcoded
  defaults with a missing-sentinel and a hard ``raise``.

Each test invokes the fixed method on an instance built via ``__new__`` so we
do not need the full DI/optimizer machinery — the guard fires *before* any
``self``-dependent work on the tested path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies.padnet_strategy import (  # noqa: E402
    PaDNetTrainingStrategy,
)


def _make_strategy() -> PaDNetTrainingStrategy:
    """Build a bare strategy instance without running __init__ / DI wiring."""
    return PaDNetTrainingStrategy.__new__(PaDNetTrainingStrategy)


def _input() -> torch.Tensor:
    return torch.zeros(2, 1, 8, 8)


def test_missing_tr_and_te_raises() -> None:
    """A batch with neither tr nor te must raise instead of fabricating params."""
    strat = _make_strategy()
    with pytest.raises(ValueError, match=r"requires 'tr'/'te' in batch metadata"):
        strat._compute_losses_impl(
            _input(),
            _input(),
            epoch=0,
            batch={},
        )


def test_missing_te_only_raises() -> None:
    """tr present but te absent must still raise (both are required)."""
    strat = _make_strategy()
    with pytest.raises(ValueError, match=r"refusing to fabricate sequence params"):
        strat._compute_losses_impl(
            _input(),
            _input(),
            epoch=0,
            batch={"tr": 500.0},
        )


def test_missing_tr_only_raises() -> None:
    """te present but tr absent must still raise (both are required)."""
    strat = _make_strategy()
    with pytest.raises(ValueError, match=r"requires 'tr'/'te'"):
        strat._compute_losses_impl(
            _input(),
            _input(),
            epoch=0,
            batch={"te": 10.0},
        )


def test_metadata_object_batch_missing_raises() -> None:
    """Non-dict batch whose .metadata lacks tr/te must also raise."""

    class _FakeBatch:
        metadata: dict[str, float] = {}

    strat = _make_strategy()
    with pytest.raises(ValueError, match=r"requires 'tr'/'te' in batch metadata"):
        strat._compute_losses_impl(
            _input(),
            _input(),
            epoch=0,
            batch=_FakeBatch(),
        )


def test_present_tr_te_passes_the_guard() -> None:
    """With both tr and te present the tr/te guard does NOT raise.

    We let execution proceed past the guard; on a bare ``__new__`` instance the
    strategy next reaches into ``self.env.generator``, which raises
    ``AttributeError`` (not the guard's ``ValueError``) — proving the tr/te
    guard itself did not fire when both params are supplied.
    """
    strat = _make_strategy()
    with pytest.raises(AttributeError):
        strat._compute_losses_impl(
            _input(),
            _input(),
            epoch=0,
            batch={"tr": 500.0, "te": 10.0},
        )
