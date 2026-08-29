"""Trajectory routing for :class:`NonCartesianSimulationTransform` (#1097).

``data.trajectory`` used to be validated against a five-name tuple and then routed
through a two-name ``if``: ``golden_angle`` and ``epi`` passed validation, missed the
dispatch, and fell through to a ``_NoOpTransform()`` announced at ``debug`` level. An
arm asking for a golden-angle acquisition trained with **no non-Cartesian simulation at
all** and said so nowhere a human would look.

Meanwhile ``trajectories.get_trajectory`` — a complete, ``Literal``-typed dispatcher over
exactly the right four families — had no production callers, while the transform
hand-rolled a two-member copy of it.

The three properties this file pins:

1. **Live science does not move.** Six ``spiral`` and two ``radial`` arms are committed;
   routing through the shared generator must produce byte-identical trajectories, so
   the density derivation stays in the transform (budget policy) while construction
   moves to ``get_trajectory`` (geometry).
2. **The newly-reachable families are real**, not a facade (pitfall #16) — a wiring fix
   that made ``golden_angle`` construct the same tensor as ``radial`` would be worse
   than the NoOp, because it would look like it worked.
3. **The accept-set and the route-set are one list.** Asserted against the constant, so
   the two cannot drift apart the way they originally did.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.data.transforms.non_cartesian import NonCartesianSimulationTransform
from mriforge.infrastructure.physics.trajectories import (
    NON_CARTESIAN_TRAJECTORIES,
    TRAJECTORY_TYPES,
    TrajectoryFactory,
    get_trajectory,
)

IM_SIZE = (256, 256)
ACCEL = 4.0


def _built(pattern: str, im_size=IM_SIZE, acceleration=ACCEL):
    """Construct through the transform's own derivation, as the live path does."""
    tr = NonCartesianSimulationTransform(
        pattern=pattern, im_size=im_size, acceleration=acceleration
    )
    return get_trajectory(
        trajectory_type=pattern, im_size=im_size, **tr._density_kwargs(im_size)
    )


class TestLiveArmsAreUnmoved:
    """The behaviour-preservation contract, asserted on tensors rather than argued.

    The old dispatch called the factory methods directly with a density derived from
    ``acceleration``. If routing through ``get_trajectory`` changed either the callee
    or the derived parameter, eight committed arms would quietly train on a different
    sampling density — the exact failure a 'wiring fix' must not cause.
    """

    def test_radial_is_byte_identical_to_the_old_direct_call(self) -> None:
        old_t, old_d = TrajectoryFactory.get_radial_trajectory(
            im_size=IM_SIZE, num_spokes=int(max(IM_SIZE) / ACCEL)
        )
        new_t, new_d = _built("radial")
        assert torch.equal(old_t, new_t)
        assert torch.equal(old_d, new_d)

    def test_spiral_is_byte_identical_to_the_old_direct_call(self) -> None:
        old_t, old_d = TrajectoryFactory.get_spiral_trajectory(
            im_size=IM_SIZE, num_arms=max(1, int(48 / ACCEL))
        )
        new_t, new_d = _built("spiral")
        assert torch.equal(old_t, new_t)
        assert torch.equal(old_d, new_d)

    def test_the_derived_density_matches_the_original_arithmetic(self) -> None:
        """Pins the expressions themselves, so a later 'tidy-up' of the formulas
        fails here rather than silently re-tuning eight arms."""
        tr = NonCartesianSimulationTransform(im_size=IM_SIZE, acceleration=ACCEL)
        tr.pattern = "radial"
        assert tr._density_kwargs(IM_SIZE) == {"num_spokes": 64}  # 256 / 4
        tr.pattern = "spiral"
        assert tr._density_kwargs(IM_SIZE) == {"num_arms": 12}  # 48 / 4


class TestTheNewlyReachableFamiliesAreReal:
    """Not a facade: these must differ from what they would have been mistaken for."""

    def test_golden_angle_is_not_just_radial_under_another_name(self) -> None:
        radial, _ = _built("radial")
        golden, _ = _built("golden_angle")
        assert not torch.equal(radial, golden), (
            "golden_angle producing the radial trajectory would be a facade (#16): "
            "reachable, apparently working, and physically wrong."
        )

    def test_golden_angle_uses_the_winkelmann_increment(self) -> None:
        """~111.25 deg = pi(sqrt5 - 1)/2, the radial-MRI golden angle -- NOT the
        137.508 deg 2-D phyllotaxis angle, a confusion the factory calls out."""
        traj, _ = _built("golden_angle")
        spokes = traj.reshape(2, 64, -1)
        angles = torch.atan2(spokes[1, :, -1], spokes[0, :, -1])
        step = (math.degrees(angles[1].item()) - math.degrees(angles[0].item())) % 180
        assert step == pytest.approx(111.25, abs=0.1)

    def test_radial_uses_uniform_increments(self) -> None:
        """The contrast that makes the test above meaningful."""
        traj, _ = _built("radial")
        spokes = traj.reshape(2, 64, -1)
        angles = torch.atan2(spokes[1, :, -1], spokes[0, :, -1])
        step = (math.degrees(angles[1].item()) - math.degrees(angles[0].item())) % 180
        assert step == pytest.approx(180 / 64, abs=0.1)

    def test_epi_honours_the_acceleration_as_a_line_count(self) -> None:
        """EPI takes acceleration natively, so the budget must land as phase-encode
        lines rather than being dropped on the floor."""
        traj, _ = _built("epi")
        assert len(torch.unique(traj[1])) == IM_SIZE[0] // int(ACCEL)

    @pytest.mark.parametrize("pattern", NON_CARTESIAN_TRAJECTORIES)
    def test_every_declared_family_actually_constructs(self, pattern) -> None:
        """The ratchet. A name in the constant that cannot build is the original bug
        in its next incarnation."""
        traj, dcf = _built(pattern)
        assert traj.shape[0] == 2
        assert traj.shape[1] > 0
        assert dcf.shape[0] == traj.shape[1]
        assert torch.isfinite(traj).all()


class TestTheVocabularyIsOneList:
    def test_cartesian_is_accepted_but_is_not_a_nufft_family(self) -> None:
        """The one asymmetry between the two constants, pinned because it looks like
        an oversight: cartesian is a legal `data.trajectory`, but asking a NUFFT
        trajectory generator for it is a caller bug."""
        assert "cartesian" in TRAJECTORY_TYPES
        assert "cartesian" not in NON_CARTESIAN_TRAJECTORIES
        with pytest.raises(ValueError, match="Unknown trajectory type"):
            get_trajectory(trajectory_type="cartesian", im_size=IM_SIZE)  # type: ignore[arg-type]

    def test_the_two_constants_differ_only_by_cartesian(self) -> None:
        assert set(TRAJECTORY_TYPES) - set(NON_CARTESIAN_TRAJECTORIES) == {"cartesian"}

    def test_the_builders_re_export_is_the_same_object_not_a_copy(self) -> None:
        """``data.builders`` imports the vocabulary from ``data.transforms`` rather
        than from ``infrastructure.physics`` directly, because ``mriforge.data ->
        infrastructure`` is a layer-direction violation and only ``non_cartesian``
        carries the recorded exception for it.

        That indirection is exactly how a "copy" could creep back in and re-open
        #1097, so assert OBJECT IDENTITY: the builder must be validating against the
        same tuple the generator routes on, not a look-alike.
        """
        from mriforge.data.builders import torchio_transform_builder as builder
        from mriforge.infrastructure.physics import trajectories as physics

        assert builder.TRAJECTORY_TYPES is physics.TRAJECTORY_TYPES
        assert builder.NON_CARTESIAN_TRAJECTORIES is physics.NON_CARTESIAN_TRAJECTORIES

    def test_an_unroutable_pattern_raises_instead_of_no_opping(self) -> None:
        tr = NonCartesianSimulationTransform(pattern="not_a_family", im_size=IM_SIZE)
        with pytest.raises(ValueError, match="Unknown pattern"):
            tr._density_kwargs(IM_SIZE)


class TestExplicitKwargsOverrideTheDerivedBudget:
    def test_an_explicit_num_arms_wins_over_the_acceleration_derivation(self) -> None:
        """The old code passed BOTH the derived value and ``**self.kwargs``, so
        supplying ``num_arms`` raised `TypeError: got multiple values for keyword
        argument` — the override branch it appeared to have never worked."""
        tr = NonCartesianSimulationTransform(pattern="spiral", im_size=IM_SIZE)
        tr.kwargs = {"num_arms": 7}
        assert tr._density_kwargs(IM_SIZE) == {"num_arms": 7}

    def test_the_override_reaches_the_generator(self) -> None:
        tr = NonCartesianSimulationTransform(pattern="spiral", im_size=IM_SIZE)
        tr.kwargs = {"num_arms": 3}
        traj, _ = get_trajectory(
            trajectory_type="spiral", im_size=IM_SIZE, **tr._density_kwargs(IM_SIZE)
        )
        baseline, _ = _built("spiral")
        assert traj.shape[1] < baseline.shape[1]
