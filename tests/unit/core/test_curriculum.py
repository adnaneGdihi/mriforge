"""Declared vs effective timestep curriculum (#1296).

The bypass itself is correct and stays. What is pinned here is that an arm
which DECLARES a curriculum and then does not get one can be told apart from
one that never asked -- because before this, every artifact a run produced
looked identical in the two cases.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.core.curriculum import (
    DEFAULT_MAX_ITERATIONS,
    SHORT_RUN_BYPASS_ITERATIONS,
    resolve_curriculum_state,
)


def _config(
    start: int | None = 4,
    rate: float | None = 0.005,
    max_iterations: int | None = 30_000,
) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            curriculum_start_timestep=start,
            curriculum_ramp_rate=rate,
            max_iterations=max_iterations,
        )
    )


def test_a_long_run_with_both_knobs_gets_its_curriculum() -> None:
    state = resolve_curriculum_state(_config())
    assert state.declared and state.effective
    assert state.suppressed_by is None


def test_the_short_run_bypass_suppresses_a_declared_curriculum() -> None:
    """The #1296 case. The bypass is deliberate -- a 5k ramp never reaches
    meaningful acceleration -- but nothing said so."""
    state = resolve_curriculum_state(_config(max_iterations=2_000))
    assert state.declared
    assert not state.effective
    assert "short-run bypass" in (state.suppressed_by or "")


def test_the_bypass_boundary_is_inclusive() -> None:
    """`<=`, matching the strategy. One iteration either side must differ."""
    at = resolve_curriculum_state(_config(max_iterations=SHORT_RUN_BYPASS_ITERATIONS))
    above = resolve_curriculum_state(_config(max_iterations=SHORT_RUN_BYPASS_ITERATIONS + 1))
    assert not at.effective
    assert above.effective


def test_declaring_neither_knob_is_silence_not_suppression() -> None:
    """No curriculum was asked for, so there is nothing to warn about."""
    state = resolve_curriculum_state(_config(start=None, rate=None))
    assert not state.declared and not state.effective
    assert state.suppressed_by is None
    assert state.describe() == "curriculum=off"


@pytest.mark.parametrize(("start", "rate"), [(4, None), (None, 0.005)])
def test_half_a_curriculum_is_reported_not_swallowed(start, rate) -> None:
    """`start_t + iteration * rate` needs both, so one alone does nothing.

    It is not `declared` -- it cannot run -- but it must not read as a plain
    `off` either, because someone set a knob and got nothing.
    """
    state = resolve_curriculum_state(_config(start=start, rate=rate))
    assert not state.declared and not state.effective
    assert "both are required" in (state.suppressed_by or "")
    assert "declared-but-off" in state.describe()


def test_an_unset_horizon_falls_back_rather_than_tripping_the_bypass() -> None:
    """`max_iterations=None` must not compare as 0 and silently suppress."""
    state = resolve_curriculum_state(_config(max_iterations=None))
    assert state.effective
    assert state.max_iterations == DEFAULT_MAX_ITERATIONS


def test_the_description_distinguishes_off_from_declared_but_off() -> None:
    """The whole point: these two states used to be indistinguishable."""
    off = resolve_curriculum_state(_config(start=None, rate=None)).describe()
    suppressed = resolve_curriculum_state(_config(max_iterations=100)).describe()
    on = resolve_curriculum_state(_config()).describe()
    assert off != suppressed
    assert on.startswith("curriculum=on(")
    assert "t0=4" in on and "rate=0.005" in on


def test_resolution_reads_only_config_and_never_raises_on_a_bare_object() -> None:
    """It runs at strategy construction and inside the startup knob line, so a
    partially-built config must degrade to `off`, not abort the run."""
    state = resolve_curriculum_state(SimpleNamespace())
    assert not state.declared and not state.effective
    assert state.max_iterations == DEFAULT_MAX_ITERATIONS
