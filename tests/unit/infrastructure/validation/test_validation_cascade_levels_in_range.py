"""Unit tests for the validation_cascade_levels_in_range audit check.

Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.
"""

from __future__ import annotations

from types import SimpleNamespace

from mriforge.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


class _Accel:
    """Minimal duck-typed acceleration block.

    The audit reads attributes via ``getattr`` so a plain object suffices —
    we deliberately don't build a full ``AccelerationConfig`` Pydantic
    model here to keep the test fast and decoupled from schema churn.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Training:
    def __init__(self, diffusion=True):
        self.diffusion = object() if diffusion else None


class _Config:
    def __init__(self, accel, training, validation=None):
        # Phase 11 renamed the top-level block `acceleration:` -> `undersampling:`
        # and the checker was migrated with it; this stand-in was not, so every
        # check short-circuited on "acceleration or training section absent."
        # and reported a vacuous pass.
        self.undersampling = accel
        self.training = training
        # Absent unless a test declares one — `getattr(config, "validation",
        # None)` then resolves to the framework default ladder, which is what
        # every arm written before #1394 gets.
        self.validation = validation


def _check(accel, *, diffusion=True, levels=None):
    checker = ConfigHealthChecker()
    validation = (
        None if levels is None else SimpleNamespace(cascade=SimpleNamespace(levels=levels))
    )
    cfg = _Config(
        accel=accel, training=_Training(diffusion=diffusion), validation=validation
    )
    return checker.check_validation_cascade_levels_in_range(cfg)


class TestPassingCases:
    def test_all_cascade_levels_present(self):
        """experiment_11 declares every cascade level — must pass."""
        accel = _Accel(
            schedule_type="step",
            acceleration_range=[2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
        )
        result = _check(accel)
        assert result.passed
        assert "all cascade levels" in result.message

    def test_linear_schedule_skips(self):
        accel = _Accel(schedule_type="linear", acceleration_range=[2.0, 32.0])
        result = _check(accel)
        assert result.passed
        assert "closed-form inverse" in result.message

    def test_power_law_schedule_skips(self):
        accel = _Accel(schedule_type="power_law", acceleration_range=[2.0])
        result = _check(accel)
        assert result.passed
        assert "closed-form inverse" in result.message

    def test_step_without_range_skips(self):
        accel = _Accel(schedule_type="step")
        result = _check(accel)
        assert result.passed
        assert "binary fallback" in result.message

    def test_non_diffusion_skips(self):
        accel = _Accel(
            schedule_type="step", acceleration_range=[2.0, 4.0]
        )
        result = _check(accel, diffusion=False)
        assert result.passed
        assert "cascade does not run" in result.message


class TestFailingCases:
    def test_missing_level_8x_flagged(self):
        accel = _Accel(
            schedule_type="step",
            acceleration_range=[2.0, 4.0, 16.0, 32.0],  # no 8.0
        )
        result = _check(accel)
        assert not result.passed
        assert result.severity == "warning"
        # #1394: the ladder now comes from `resolve_cascade_levels`, which
        # keeps an integral rung as `int` so the flat column names stay
        # `val_psnr_8x` rather than `val_psnr_8.0x`. The message renders the
        # same spelling it tells the reader to look for, so this pins "[8]"
        # and not "8.0" -- anchored on the bracket, a stronger pin than the
        # substring it replaces.
        assert "[8]" in result.message

    def test_missing_level_2x_flagged(self):
        accel = _Accel(
            schedule_type="step",
            acceleration_range=[8.0, 32.0],  # no 2.0 (also no 8.0 check)
        )
        result = _check(accel)
        assert not result.passed
        assert "2.0" in result.message

    def test_enum_schedule_type_resolved(self):
        """The pydantic schema yields ``AccelerationSchedule.STEP`` (enum), not
        the bare string ``'step'``. The check must unwrap ``.value``."""
        class _FakeEnum:
            value = "step"

            def __str__(self):  # noqa: D401 — mimic enum's verbose ``str``
                return "AccelerationSchedule.STEP"

        accel = _Accel(
            schedule_type=_FakeEnum(),
            acceleration_range=[2.0],  # 8 + 32 missing
        )
        result = _check(accel)
        assert not result.passed
        # If the check stringified the enum naively it would have
        # short-circuited to the "closed-form" branch and passed.
        # #1394: the ladder now comes from `resolve_cascade_levels`, which
        # keeps an integral rung as `int` so the flat column names stay
        # `val_psnr_8x` rather than `val_psnr_8.0x`. The message renders the
        # same spelling it tells the reader to look for, so this pins "[8]"
        # and not "8.0" -- anchored on the bracket, a stronger pin than the
        # substring it replaces.
        assert "[8, 32]" in result.message


class TestDeclaredLadderIsWhatGetsChecked:
    """The check reads `validation.cascade.levels`, not a private copy (#1394).

    Both of these were RED before the check stopped holding its own
    ``(2.0, 8.0, 32.0)`` — and they fail in OPPOSITE directions, which is the
    point. A checker frozen on the old tuple does not merely go quiet: it
    reports on rungs the run never evaluates *and* stays silent about the ones
    it does. Testing only the default ladder would score the old code green.
    """

    def test_a_narrow_declared_ladder_passes_a_narrow_acceleration_range(self):
        """Declared [2, 4] against range [2, 4] is complete.

        The old code demanded 8 and 32 — rungs this arm never runs — so it
        failed a correct config and told the author to widen a range that
        needed no widening.
        """
        accel = _Accel(schedule_type="step", acceleration_range=[2.0, 4.0])
        result = _check(accel, levels=[2, 4])
        assert result.passed, result.message

    def test_a_declared_rung_outside_the_range_is_flagged(self):
        """Declared [2, 16] against range [2, 8, 32] must name 16.

        The old code checked 2/8/32 — all present — and passed, so the one
        rung that would actually be skipped at runtime went unreported. The
        symptom for the user is a missing column, not a failed audit.
        """
        accel = _Accel(schedule_type="step", acceleration_range=[2.0, 8.0, 32.0])
        result = _check(accel, levels=[2, 16])
        assert not result.passed
        assert "16" in result.message
        assert "32" not in result.message.split("acceleration_range")[0]

    def test_an_undeclared_ladder_still_checks_the_framework_default(self):
        """No `validation:` block — behaviour is exactly what it was before."""
        accel = _Accel(schedule_type="step", acceleration_range=[2.0, 4.0])
        result = _check(accel)
        assert not result.passed
        assert "8" in result.message and "32" in result.message

    def test_an_illegal_declared_ladder_is_reported_not_defaulted(self):
        """A ladder the resolver rejects must NOT fall back to (2, 8, 32).

        Checking a ladder the run will never evaluate is worse than checking
        nothing: it produces a verdict about rungs that do not exist
        (non-negotiable 3).
        """
        accel = _Accel(schedule_type="step", acceleration_range=[2.0, 8.0, 32.0])
        result = _check(accel, levels=[0.5])
        assert not result.passed
        assert "not a legal ladder" in result.message
