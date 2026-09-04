"""Round-trip tests for ``KSpaceAccelerator.timestep_for_acceleration``.

Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

The previous validation cascade in
``infrastructure/training/strategies/diffusion.py`` used a hard-coded
linear inverse to map a desired acceleration ``R`` to a diffusion
timestep ``t``. For ``schedule_type='step'`` with a non-uniform
``acceleration_range`` the inverse silently disagreed with the forward
mask — ``R=8`` resolved to ``t=200`` which the step schedule decoded as
``R=4``. The ``val_*_<R>x`` metric columns therefore mislabeled the
acceleration that was actually being tested.

These tests cover the new
:meth:`KSpaceAccelerator.timestep_for_acceleration` inverse and verify
the round-trip ``R → t → R`` is honest under every supported
``acceleration_schedule``.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from spectramr.infrastructure.physics.sampling import (
    UniformCartesianKSpaceAccelerator,
)


def _sampler(**kw):
    defaults = dict(
        num_timesteps=1000,
        max_acceleration=32.0,
        base_acceleration=2.0,
        center_fraction=0.08,
        seed=42,
    )
    defaults.update(kw)
    return UniformCartesianKSpaceAccelerator(**defaults)


class TestLinearInverse:
    def test_endpoints(self):
        s = _sampler(acceleration_schedule="linear")
        assert s.timestep_for_acceleration(2.0) == 0
        assert s.timestep_for_acceleration(32.0) == 999

    def test_midpoint(self):
        s = _sampler(acceleration_schedule="linear")
        # R = 17 → ratio = (17-2)/30 = 0.5 → t ≈ 500
        t = s.timestep_for_acceleration(17.0)
        assert abs(t - 500) <= 1

    @pytest.mark.parametrize("r", [2.0, 4.0, 8.0, 17.0, 32.0])
    def test_round_trip(self, r):
        s = _sampler(acceleration_schedule="linear")
        t = s.timestep_for_acceleration(r)
        assert s.get_acceleration_factor(t) == pytest.approx(r, abs=0.1)


class TestPowerLawInverse:
    @pytest.mark.parametrize("r", [2.0, 8.0, 17.0, 32.0])
    def test_round_trip(self, r):
        s = _sampler(acceleration_schedule="power_law", schedule_power=2.0)
        t = s.timestep_for_acceleration(r)
        assert s.get_acceleration_factor(t) == pytest.approx(r, abs=0.5)


class TestExponentialInverse:
    @pytest.mark.parametrize("r", [2.0, 8.0, 17.0, 32.0])
    def test_round_trip(self, r):
        s = _sampler(acceleration_schedule="exponential")
        t = s.timestep_for_acceleration(r)
        assert s.get_acceleration_factor(t) == pytest.approx(r, abs=0.5)


class TestStepInverseWithExplicitRange:
    """The experiment_11 regression: cascade picks R values from the YAML's
    ``acceleration_range`` and must land in the matching bucket."""

    RANGE = [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]

    @pytest.mark.parametrize("r", RANGE)
    def test_round_trip(self, r):
        s = _sampler(
            acceleration_schedule="step",
            acceleration_range=self.RANGE,
        )
        t = s.timestep_for_acceleration(r)
        # Must land in the bucket whose forward mapping equals r.
        assert s.get_acceleration_factor(t) == pytest.approx(r, abs=1e-6)

    def test_off_grid_raises(self):
        s = _sampler(
            acceleration_schedule="step",
            acceleration_range=self.RANGE,
        )
        # 6.0 is not in the declared range — step schedule has no honest
        # inverse here. Linear-inverse fallback would silently snap to
        # bucket idx=1 (R=4) — exactly the experiment_11 mislabel.
        with pytest.raises(ValueError, match="cannot represent R=6.0"):
            s.timestep_for_acceleration(6.0)

    def test_experiment_11_cascade_round_trip(self):
        """The exact YAML config from experiment_11_kspace_cold_diffusion.

        Cascade levels [2, 8, 32] each must land in their declared
        ``acceleration_range`` bucket. Prior to this fix the linear
        inverse mapped R=8 to t=200, which the step schedule decoded
        as R=4 (the 8x column lied).
        """
        s = _sampler(
            acceleration_schedule="step",
            acceleration_range=self.RANGE,
        )
        for r in (2.0, 8.0, 32.0):
            t = s.timestep_for_acceleration(r)
            assert s.get_acceleration_factor(t) == pytest.approx(r, abs=1e-6), (
                f"Cascade level R={r} round-tripped to "
                f"R={s.get_acceleration_factor(t)} via t={t}; the "
                f"validation column would have lied."
            )


class TestStepInverseBinaryFallback:
    """Without an explicit ``acceleration_range`` the binary fallback
    returns base for ``R ≤ base`` and max for everything above. The
    inverse must agree."""

    def test_base_returns_zero(self):
        s = _sampler(acceleration_schedule="step")
        assert s.timestep_for_acceleration(2.0) == 0

    def test_max_returns_T_minus_1(self):
        s = _sampler(acceleration_schedule="step")
        assert s.timestep_for_acceleration(32.0) == 999

    def test_intermediate_routes_to_max(self):
        s = _sampler(acceleration_schedule="step")
        # No range → binary schedule; anything > base lands at max.
        assert s.timestep_for_acceleration(8.0) == 999


class TestLinearInverseRegressionMatchesOldCascade:
    """The historical hard-coded cascade formula ``t = T*(R-base)/(max-base)``
    is the linear inverse. ``timestep_for_acceleration`` must reproduce
    it byte-for-byte under ``schedule_type='linear'`` so legacy strategies
    that route through this method don't see a behavioral shift."""

    def test_matches_old_formula(self):
        s = _sampler(acceleration_schedule="linear")
        T = 1000
        base = 2.0
        max_a = 32.0
        span = max_a - base
        for r in (2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0):
            old_t = int(round(T * (r - base) / span))
            # The old code stored ``int(t_ideal)`` then clamped; we
            # ``round`` for stability — within 1 of the old formula.
            new_t = s.timestep_for_acceleration(r)
            assert abs(new_t - old_t) <= 1, (
                f"R={r}: legacy={old_t}, new={new_t}"
            )


class TestDeclaredIsNotTheSameAsRealised:
    """Issue #1171: membership in ``acceleration_range`` != reachable by any ``t``.

    The step forward index is ``min(int(ratio * steps), steps - 1)`` with
    ``ratio = t / (T - 1)``. It can only take ``T`` distinct values, so when the
    ladder declares MORE rungs than there are timesteps some entries are skipped
    entirely. The inverse checked membership in the declared list, which is still
    satisfied for a skipped rung, and so returned a timestep that decodes to a
    *neighbouring* rung.

    That is the exact failure the docstring promises to prevent, and it is
    load-bearing: ``_CASCADING_LEVELS = (2.0, 8.0, 32.0)`` reaches the validation
    cascade through this method, and the cascade *catches* ``ValueError`` to skip
    a rung it cannot test. A wrong-but-silent timestep makes it report a result
    under the wrong severity label instead.
    """

    #: One rung per timestep — the shape the kspace_filling ladder was widened to
    #: in #1155, and the shape every current arm satisfies (measured: all 7 arms
    #: under experiments/inprogress/ with len(range) != timesteps have FEWER
    #: rungs than timesteps, where nothing is skipped).
    LADDER_28: ClassVar[list[float]] = [
        2.0, 2.226, 2.485, 2.753, 3.048, 3.413, 3.765, 4.197, 4.655, 5.224,
        5.818, 6.4, 7.111, 8.0, 8.828, 9.846, 10.667, 11.636, 12.8, 14.222,
        16.0, 18.286, 19.692, 21.333, 23.273, 25.6, 28.444, 32.0,
    ]

    def _stepped(self, ladder, timesteps):
        return _sampler(
            acceleration_schedule="step",
            acceleration_range=list(ladder),
            num_timesteps=timesteps,
            base_acceleration=ladder[0],
        )

    def test_one_rung_per_timestep_round_trips_on_every_rung(self):
        """The baseline this guard must not disturb."""
        s = self._stepped(self.LADDER_28, 28)
        for r in self.LADDER_28:
            t = s.timestep_for_acceleration(r)
            assert s.get_acceleration_factor(t) == pytest.approx(r, abs=1e-6), (
                f"R={r} round-tripped to t={t} which decodes to "
                f"{s.get_acceleration_factor(t)}"
            )

    def test_fewer_rungs_than_timesteps_still_round_trips(self):
        """7 rungs over 28 timesteps: buckets are wide, nothing is skipped.

        This is the shape all seven mismatched arms in the corpus have, so the
        new raise must not fire for them.
        """
        s = self._stepped([2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0], 28)
        for r in (2.0, 8.0, 32.0):
            t = s.timestep_for_acceleration(r)
            assert s.get_acceleration_factor(t) == pytest.approx(r, abs=1e-6)

    def test_more_rungs_than_timesteps_raises_instead_of_snapping(self):
        """Prepending R=1 without widening ``timesteps`` skips the R=8 rung.

        29 rungs over 28 timesteps is the concrete edit that motivated this:
        extending the ladder down to a fully-sampled single-NEX rung.
        """
        s = self._stepped([1.0, *self.LADDER_28], 28)

        realised = {round(s.get_acceleration_factor(t), 6) for t in range(28)}
        assert 8.0 not in realised, (
            "premise of this test: with 29 rungs over 28 timesteps the forward "
            "index must skip the 8.0 entry"
        )

        with pytest.raises(ValueError, match="realised at NO timestep"):
            s.timestep_for_acceleration(8.0)

    def test_the_raise_is_a_value_error_so_the_cascade_still_skips(self):
        """The cascade catches ``ValueError`` specifically.

        Raising anything else would turn a skipped validation rung into a crashed
        training run — a worse outcome than the bug being fixed.
        """
        s = self._stepped([1.0, *self.LADDER_28], 28)
        with pytest.raises(ValueError):
            s.timestep_for_acceleration(8.0)

    def test_a_reachable_rung_still_resolves_on_an_oversized_ladder(self):
        """The guard is per-request, not a blanket refusal of the ladder.

        Rungs that ARE realised must keep working, so an over-long ladder
        degrades one rung at a time rather than disabling the inverse wholesale.
        """
        s = self._stepped([1.0, *self.LADDER_28], 28)
        realised = sorted({round(s.get_acceleration_factor(t), 6) for t in range(28)})
        assert realised, "the schedule must realise something"
        for r in (realised[0], realised[len(realised) // 2], realised[-1]):
            t = s.timestep_for_acceleration(r)
            assert s.get_acceleration_factor(t) == pytest.approx(r, abs=1e-6)

    def test_an_off_grid_request_still_names_the_declared_set(self):
        """The pre-existing off-grid message must not be shadowed by the new one.

        Two different authoring mistakes with two different fixes: 6.0 is simply
        not a declared rung, while a skipped 8.0 IS declared but unreachable.
        """
        s = self._stepped(self.LADDER_28, 28)
        with pytest.raises(ValueError, match=r"cannot .*represent"):
            s.timestep_for_acceleration(6.0)
