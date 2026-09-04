"""Integration test: experiment_11 cascade routes through the schedule-aware inverse.

Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

The diffusion strategy's cascading validation picks three representative
acceleration levels (``[2, 8, 32]``) and asks the mask generator for a
mask at each. Prior to 2026-05-28 the strategy used a hard-coded linear
inverse to pick the timestep — for the experiment_11 YAML's
``schedule_type: step`` declaration the linear inverse silently produced
timesteps the step schedule decoded as a *different* acceleration. The
``val_*_<R>x`` metric columns were mislabeled.

This integration test exercises the mask-generator path the strategy
uses at validation time and pins the round-trip: each cascade level's
timestep, when fed back into the same mask generator, must reproduce
the requested acceleration within tolerance.

We hard-code the experiment_11 acceleration block here rather than
loading the YAML so the test is resilient to unrelated YAML edits and
so it does not need a full ``TrainingSettings`` round-trip.
"""

from __future__ import annotations

import pytest

from spectramr.core.cascading_validation import check_round_trip, training_band
from spectramr.infrastructure.training.utils.kspace_masks import (
    create_kspace_mask_generator,
)

# Mirrors the acceleration block of
# experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml
# (re-synced 2026-08-16 for #1157, which widened that arm's ladder from 7 rungs
# to 28 -- one per timestep -- and dropped min_center_fraction to 0.02).
#
# NUM_TIMESTEPS deliberately does NOT mirror the arm: the arm runs T=28, and
# these cases assert that the cascade round-trip holds for a horizon far longer
# than the ladder, which is where an off-by-one in the index map would show.
# Everything else here is the arm's live block; re-sync it when that block moves.
EXPERIMENT_11_ACCEL = {
    "max_acceleration": 32.0,
    "base_acceleration": 2.0,
    "center_fraction": 0.08,
    "min_center_fraction": 0.02,
    "acceleration_range": [
        2.0, 2.226, 2.485, 2.753, 3.048, 3.413, 3.765, 4.197, 4.655, 5.224,
        5.818, 6.4, 7.111, 8.0, 8.828, 9.846, 10.667, 11.636, 12.8, 14.222,
        16.0, 18.286, 19.692, 21.333, 23.273, 25.6, 28.444, 32.0,
    ],
    "acceleration_schedule": "step",
    "seed": 42,
}
NUM_TIMESTEPS = 1000


def _mask_generator(pattern: str = "uniform_cartesian"):
    return create_kspace_mask_generator(
        num_timesteps=NUM_TIMESTEPS,
        default_pattern=pattern,
        accelerator_kwargs=EXPERIMENT_11_ACCEL,
    )


class TestCascadeRoundTrip:
    """The cascade levels [2, 8, 32] each round-trip cleanly under
    ``schedule_type='step'`` with the experiment_11 ``acceleration_range``."""

    @pytest.mark.parametrize("cascade_level", [2.0, 8.0, 32.0])
    def test_cascade_level_round_trip(self, cascade_level):
        mg = _mask_generator()
        accel = mg._get_accelerator(None)
        t = accel.timestep_for_acceleration(cascade_level)
        # Forward schedule must reproduce the level — anything else means
        # the val_*_<R>x column is mislabeled.
        assert accel.get_acceleration_factor(t) == pytest.approx(
            cascade_level, abs=1e-6
        ), (
            f"Cascade level R={cascade_level} round-tripped to "
            f"R={accel.get_acceleration_factor(t)} via t={t}"
        )

    def test_step_schedule_rejects_off_grid_cascade(self):
        """If a future contributor adds an off-grid level to ``_CASCADING_LEVELS``,
        the strategy will see a ``ValueError`` and skip the level — the test
        guarantees the inverse refuses to silently snap to a neighbouring bucket."""
        mg = _mask_generator()
        accel = mg._get_accelerator(None)
        # 6.0 is not in EXPERIMENT_11_ACCEL["acceleration_range"].
        with pytest.raises(ValueError, match="cannot represent R=6.0"):
            accel.timestep_for_acceleration(6.0)


class TestLinearScheduleFallback:
    """When the YAML uses ``schedule_type='linear'`` the cascade should
    behave identically to the historical hard-coded formula."""

    def test_linear_inverse_matches_legacy_formula(self):
        accel_kwargs = dict(EXPERIMENT_11_ACCEL)
        accel_kwargs["acceleration_schedule"] = "linear"
        accel_kwargs.pop("acceleration_range", None)
        mg = create_kspace_mask_generator(
            num_timesteps=NUM_TIMESTEPS,
            default_pattern="uniform_cartesian",
            accelerator_kwargs=accel_kwargs,
        )
        accel = mg._get_accelerator(None)
        base = EXPERIMENT_11_ACCEL["base_acceleration"]
        max_a = EXPERIMENT_11_ACCEL["max_acceleration"]
        span = max_a - base
        for r in (2.0, 8.0, 32.0):
            legacy_t = int(round(NUM_TIMESTEPS * (r - base) / span))
            new_t = accel.timestep_for_acceleration(r)
            assert abs(new_t - legacy_t) <= 1, (
                f"R={r}: legacy_t={legacy_t}, new_t={new_t}"
            )


# ── #1295: the band clamp that re-pointed two of the three rungs ────────────
#
# The cascade took `timestep_for_acceleration`'s answer and clamped it into
# `[T//10, T - T//10]` before using it. On the attention-shootout arms that
# band is [2, 27], and the ladder's exact 2/8/32 rungs sit at t=1, 14 and 28 --
# so two of the three were moved onto a neighbour while `acceleration_level`
# kept naming the level that had been requested.
#
# T here is the arm's real horizon, not the 1000 used above: the clamp is a
# function of T, so a long horizon hides the whole defect.

#: `experiments/inprogress/kspace_filling/attention_shootout/
#: experiment_11_attention_none.yaml` -- 29 rungs, one per timestep, base 1.0.
ATTENTION_NONE_ACCEL = {
    "max_acceleration": 32.0,
    "base_acceleration": 1.0,
    "center_fraction": 0.08,
    "min_center_fraction": 0.02,
    "acceleration_range": [
        1.0, 2.0, 2.226, 2.485, 2.753, 3.048, 3.413, 3.765, 4.197, 4.655,
        5.224, 5.818, 6.4, 7.111, 8.0, 8.828, 9.846, 10.667, 11.636, 12.8,
        14.222, 16.0, 18.286, 19.692, 21.333, 23.273, 25.6, 29.444, 32.0,
    ],
    "acceleration_schedule": "step",
    "seed": 42,
}
ATTENTION_NONE_T = 29


def _attention_none_accelerator():
    return create_kspace_mask_generator(
        num_timesteps=ATTENTION_NONE_T,
        default_pattern="uniform_cartesian",
        accelerator_kwargs=ATTENTION_NONE_ACCEL,
    )._get_accelerator(None)


class TestAttentionArmBandClamp:
    """The witness numbers from the filed issue, at the arm's own horizon."""

    def test_the_band_is_two_to_twentyseven(self):
        assert training_band(ATTENTION_NONE_T) == (2, 27)

    @pytest.mark.parametrize(
        ("level", "expected_t"), [(2.0, 1), (8.0, 14), (32.0, 28)]
    )
    def test_the_inverse_lands_on_an_exact_rung(self, level, expected_t):
        """The ladder was built so 2/8/32 are exact literals -- the arm's own
        comment says so, because an off-grid level makes the cascade skip the
        rung entirely."""
        accel = _attention_none_accelerator()
        assert accel.timestep_for_acceleration(level) == expected_t
        assert accel.get_acceleration_factor(expected_t) == pytest.approx(level)

    @pytest.mark.parametrize(
        ("level", "clamped_t", "realized"),
        [(2.0, 2, 2.226), (32.0, 27, 29.444)],
    )
    def test_the_removed_clamp_would_have_mislabelled_the_column(
        self, level, clamped_t, realized
    ):
        """t=1 -> 2 and t=28 -> 27, and those buckets are different rungs.

        Note both clamped timesteps are perfectly legal, trained timesteps that
        realise perfectly sensible accelerations. Nothing crashes and no metric
        looks wrong -- the row is simply about a different R than its label
        says, which is why this survived unnoticed.
        """
        accel = _attention_none_accelerator()
        min_t, max_t = training_band(ATTENTION_NONE_T)
        t_ideal = accel.timestep_for_acceleration(level)
        assert max(min_t, min(max_t, t_ideal)) == clamped_t
        assert accel.get_acceleration_factor(clamped_t) == pytest.approx(realized)

    @pytest.mark.parametrize("level", [2.0, 8.0, 32.0])
    def test_the_guard_accepts_the_unclamped_answer(self, level):
        accel = _attention_none_accelerator()
        t = accel.timestep_for_acceleration(level)
        rt = check_round_trip(
            requested=level,
            timestep=t,
            forward=accel.get_acceleration_factor,
            num_timesteps=ATTENTION_NONE_T,
        )
        assert rt.ok, rt.reason

    @pytest.mark.parametrize(("level", "clamped_t"), [(2.0, 2), (32.0, 27)])
    def test_the_guard_would_now_refuse_a_reintroduced_clamp(self, level, clamped_t):
        """The real regression guard. If anyone puts the band back on this
        path, the cascade skips the level with a warning instead of publishing
        a column whose label and content disagree.

        A residual bound alone would not catch the R=32 case -- the ladder's
        top rungs are 2.556 apart, wider than the error -- so this is pinning
        the local-optimality half of the check specifically.
        """
        accel = _attention_none_accelerator()
        rt = check_round_trip(
            requested=level,
            timestep=clamped_t,
            forward=accel.get_acceleration_factor,
            num_timesteps=ATTENTION_NONE_T,
        )
        assert not rt.ok
        assert not rt.locally_optimal
