"""Unit tests for the progressive-GAN phase scheduler.

The full :class:`GANTrainingStrategy` requires a heavily-mocked training
environment, so these tests exercise the scheduler logic (phase / alpha
computation and phase application) on an instance built via ``__new__``
with the minimal attributes the pure methods touch. This isolates the
load-bearing growth schedule from the parent's optimisation machinery.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.training.strategies.progressive_gan_strategy import (
    ProgressiveGANStrategy,
)


class _FakeModel:
    def __init__(self, num_phases: int) -> None:
        self.num_phases = num_phases
        self.phase: int | None = None
        self.alpha: float | None = None

    def set_phase(self, phase: int, alpha: float = 1.0) -> None:
        self.phase = phase
        self.alpha = alpha


def _make_strategy(num_phases: int, schedule: list[int] | None, max_iters: int = 0):
    """Build a ProgressiveGANStrategy without running the GAN __init__."""
    strat = ProgressiveGANStrategy.__new__(ProgressiveGANStrategy)
    gen = _FakeModel(num_phases)
    disc = _FakeModel(num_phases)
    strat.env = SimpleNamespace(generator=gen, discriminator=disc)
    gan_cfg = SimpleNamespace(phase_schedule=schedule) if schedule is not None else None
    strat.config = SimpleNamespace(
        training=SimpleNamespace(gan=gan_cfg, max_iterations=max_iters)
    )
    strat._last_step_metrics = {}
    # Reproduce the relevant __init__ body.
    strat._phase_schedule = strat._resolve_phase_schedule()
    strat._phase_boundaries = strat._cumulative(strat._phase_schedule)
    return strat, gen, disc


class TestPhaseSchedule:
    def test_explicit_schedule_used(self) -> None:
        strat, _, _ = _make_strategy(3, [10, 20, 30])
        assert strat._phase_schedule == [10, 20, 30]
        assert strat._phase_boundaries == [0, 10, 30, 60]

    def test_uniform_schedule_from_max_iters(self) -> None:
        strat, _, _ = _make_strategy(4, None, max_iters=40)
        assert strat._phase_schedule == [10, 10, 10, 10]

    def test_rejects_nonpositive_entries(self) -> None:
        with pytest.raises(ValueError):
            _make_strategy(2, [10, 0])

    def test_phase0_no_fade(self) -> None:
        strat, _, _ = _make_strategy(3, [10, 10, 10])
        phase, alpha = strat._phase_and_alpha(0)
        assert phase == 0
        assert alpha == 1.0

    def test_alpha_ramps_within_phase(self) -> None:
        strat, _, _ = _make_strategy(2, [10, 10])
        # Phase 1 spans steps [10, 20); fade over first half (5 steps).
        p_start, a_start = strat._phase_and_alpha(10)
        assert p_start == 1 and a_start == pytest.approx(0.0)
        _, a_mid = strat._phase_and_alpha(12)
        assert 0.0 < a_mid < 1.0
        _, a_end = strat._phase_and_alpha(18)
        assert a_end == pytest.approx(1.0)

    def test_advances_through_phases(self) -> None:
        strat, _, _ = _make_strategy(3, [10, 10, 10])
        assert strat._phase_and_alpha(0)[0] == 0
        assert strat._phase_and_alpha(15)[0] == 1
        assert strat._phase_and_alpha(25)[0] == 2

    def test_apply_phase_mutates_both_models(self) -> None:
        strat, gen, disc = _make_strategy(3, [10, 10, 10])
        phase, alpha = strat._apply_phase(15)
        assert gen.phase == phase == 1
        assert disc.phase == phase
        assert gen.alpha == alpha
        assert disc.alpha == alpha

    def test_clamps_step_beyond_total(self) -> None:
        strat, _, _ = _make_strategy(2, [5, 5])
        phase, alpha = strat._phase_and_alpha(10_000)
        assert phase == 1
        assert alpha == pytest.approx(1.0)
