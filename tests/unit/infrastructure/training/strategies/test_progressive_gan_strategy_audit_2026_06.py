"""Regression tests for the 2026-06 progressive_gan_strategy audit fixes.

Covers three findings from the strategy audit, all fixed inside
``progressive_gan_strategy.py`` (silent-fallback hardening):

1. ``_num_phases`` must RAISE (not return a 1-phase no-op) when the resolved
   generator lacks an integer ``num_phases``.
2. ``_require_phase_capability`` (invoked from ``__init__``) must RAISE when the
   generator — or the discriminator, when present — lacks a callable
   ``set_phase``, instead of letting ``_apply_phase`` silently skip growth.
3. ``_resolve_phase_schedule`` must RAISE when an explicit ``phase_schedule``
   length does not equal the model's ``num_phases`` (no silent truncation /
   mid-training ``IndexError``).

The full ``GANTrainingStrategy.__init__`` needs a heavily-mocked environment, so
these tests build the strategy via ``__new__`` and exercise the pure helper
methods, mirroring the sibling ``test_progressive_gan_strategy.py`` idiom.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mriforge.infrastructure.training.strategies.progressive_gan_strategy import (
    ProgressiveGANStrategy,
)


class _GrowableModel:
    """Minimal phase-growable stand-in for ProgressiveGAN / discriminator."""

    def __init__(self, num_phases: int) -> None:
        self.num_phases = num_phases
        self.phase: int | None = None
        self.alpha: float | None = None

    def set_phase(self, phase: int, alpha: float = 1.0) -> None:
        self.phase = phase
        self.alpha = alpha


class _PlainModel:
    """A non-growable model: no ``num_phases``, no ``set_phase``."""


def _bare_strategy(gen: object, disc: object | None) -> ProgressiveGANStrategy:
    """Build a strategy without running the GAN __init__, env wired by hand."""
    strat = ProgressiveGANStrategy.__new__(ProgressiveGANStrategy)
    strat.env = SimpleNamespace(generator=gen, discriminator=disc)
    return strat


def _wire_config(strat: ProgressiveGANStrategy, schedule, max_iters: int = 0) -> None:
    gan_cfg = SimpleNamespace(phase_schedule=schedule) if schedule is not None else None
    strat.config = SimpleNamespace(
        training=SimpleNamespace(gan=gan_cfg, max_iterations=max_iters)
    )


# --------------------------------------------------------------------------- #
# Finding 1: _num_phases must raise instead of returning a 1-phase no-op.
# --------------------------------------------------------------------------- #
class TestNumPhasesRaises:
    def test_raises_when_generator_lacks_num_phases(self) -> None:
        strat = _bare_strategy(_PlainModel(), None)
        with pytest.raises(TypeError, match="num_phases"):
            strat._num_phases()

    def test_raises_when_num_phases_not_int(self) -> None:
        gen = SimpleNamespace(num_phases="four", set_phase=lambda *_a, **_k: None)
        strat = _bare_strategy(gen, None)
        with pytest.raises(TypeError, match="num_phases"):
            strat._num_phases()

    def test_no_silent_one_fallback(self) -> None:
        # The old code returned 1; verify the constant fallback is gone.
        strat = _bare_strategy(_PlainModel(), None)
        with pytest.raises(TypeError):
            strat._num_phases()

    def test_returns_value_for_growable_model(self) -> None:
        strat = _bare_strategy(_GrowableModel(3), None)
        assert strat._num_phases() == 3


# --------------------------------------------------------------------------- #
# Finding 2: __init__ must reject models that cannot grow (no set_phase).
# --------------------------------------------------------------------------- #
class TestRequirePhaseCapability:
    def test_raises_when_generator_has_no_set_phase(self) -> None:
        strat = _bare_strategy(_PlainModel(), None)
        with pytest.raises(TypeError, match="generator.*set_phase"):
            strat._require_phase_capability()

    def test_raises_when_discriminator_has_no_set_phase(self) -> None:
        strat = _bare_strategy(_GrowableModel(2), _PlainModel())
        with pytest.raises(TypeError, match="discriminator.*set_phase"):
            strat._require_phase_capability()

    def test_accepts_growable_generator_and_discriminator(self) -> None:
        strat = _bare_strategy(_GrowableModel(2), _GrowableModel(2))
        # Must not raise.
        strat._require_phase_capability()

    def test_accepts_growable_generator_without_discriminator(self) -> None:
        strat = _bare_strategy(_GrowableModel(2), None)
        strat._require_phase_capability()

    def test_apply_phase_no_longer_silently_skips(self) -> None:
        # With a growable model, _apply_phase mutates both models (no hasattr skip).
        gen, disc = _GrowableModel(3), _GrowableModel(3)
        strat = _bare_strategy(gen, disc)
        _wire_config(strat, [10, 10, 10])
        strat._phase_schedule = strat._resolve_phase_schedule()
        strat._phase_boundaries = strat._cumulative(strat._phase_schedule)
        phase, alpha = strat._apply_phase(15)
        assert gen.phase == phase == 1
        assert disc.phase == phase
        assert gen.alpha == alpha and disc.alpha == alpha


# --------------------------------------------------------------------------- #
# Finding 3: explicit phase_schedule length must equal num_phases.
# --------------------------------------------------------------------------- #
class TestScheduleLengthValidation:
    def test_raises_on_too_short_schedule(self) -> None:
        strat = _bare_strategy(_GrowableModel(3), None)
        _wire_config(strat, [10, 20])  # len 2 != num_phases 3
        with pytest.raises(ValueError, match="phase_schedule length"):
            strat._resolve_phase_schedule()

    def test_raises_on_too_long_schedule(self) -> None:
        strat = _bare_strategy(_GrowableModel(2), None)
        _wire_config(strat, [10, 20, 30])  # len 3 != num_phases 2
        with pytest.raises(ValueError, match="phase_schedule length"):
            strat._resolve_phase_schedule()

    def test_accepts_matching_length(self) -> None:
        strat = _bare_strategy(_GrowableModel(3), None)
        _wire_config(strat, [10, 20, 30])
        assert strat._resolve_phase_schedule() == [10, 20, 30]

    def test_positivity_check_still_enforced(self) -> None:
        strat = _bare_strategy(_GrowableModel(2), None)
        _wire_config(strat, [10, 0])
        with pytest.raises(ValueError, match="positive"):
            strat._resolve_phase_schedule()
