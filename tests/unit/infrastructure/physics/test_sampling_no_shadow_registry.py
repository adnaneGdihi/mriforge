"""The shadow accelerator hierarchy is gone and its one live name still resolves.

``clinical_sampling.py`` defined its own ``KSpaceAccelerator`` ABC and five
accelerators -- three sharing a class name with a *different* implementation in
``sampling.py`` -- behind a registry no module imported. Nineteen inprogress arms
declared its ``cartesian_vd`` key, so the name resolved nowhere on the live path
(issue #953).
"""

from __future__ import annotations

import importlib

import pytest

from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry


def test_clinical_sampling_module_is_gone() -> None:
    """A second accelerator hierarchy that nothing imports is a trap, not a spare."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("spectramr.infrastructure.physics.clinical_sampling")


def test_cartesian_vd_still_resolves_after_the_deletion() -> None:
    """19 inprogress arms declare it; deleting its only home must not orphan them."""
    assert SamplingPatternRegistry.resolve("cartesian_vd") == "variable_density_1d"


@pytest.mark.parametrize(
    ("retired", "canonical"),
    [
        ("cartesian_equispaced", "uniform_cartesian"),
        ("cartesian_random", "random_cartesian"),
        ("radial_golden", "golden_angle"),
        ("spiral_vds", "spiral"),
    ],
)
def test_every_retired_key_still_resolves(retired: str, canonical: str) -> None:
    """The other four keys had no arms, but a name that once worked should not
    start raising without warning."""
    assert SamplingPatternRegistry.resolve(retired) == canonical


def test_the_trajectory_capability_survives_elsewhere() -> None:
    """The shadow radial/spiral returned (trajectory, dcf, time_vec) rather than a
    mask -- a genuine non-Cartesian capability, not a duplicate. It is not lost:
    ``TrajectoryFactory`` provides trajectory + density compensation, and it has
    live consumers.
    """
    from spectramr.infrastructure.physics.trajectories import TrajectoryFactory

    trajectory, dcf = TrajectoryFactory.get_radial_trajectory((64, 64), num_spokes=16)
    assert trajectory.shape[0] == 2
    assert dcf.numel() == trajectory.shape[1]
