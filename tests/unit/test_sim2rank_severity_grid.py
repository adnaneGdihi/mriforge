"""The canonical severity (theta) grid SSOT and the T!=20 divergence fix.

Before this change two builders produced the ordered severity ramp:
``DegradationSweep.corruption_factors = linspace(1/T, 1, T)`` and
``MetaDegradationSweep.severity_grid = linspace(theta_min=0.05, 1, T)``. They
agreed only because ``0.05 == 1/20`` at the default ``T=20``; at any other T the
meta path silently swept a different grid than its own combined path -- a latent
correctness bug. Both now route through ``severity.severity_grid`` with the one
``1/T`` anchor.

The byte-identity at T=20 (``test_*_byte_identical_to_pre_fix_*``) is the
behavior-preservation proof: an identical grid feeds identical degraded images,
hence an identical leaderboard -- no heavier golden run is needed.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.severity import severity_grid


def test_severity_grid_default_anchor_is_one_over_t() -> None:
    # theta_min=None -> 1/T; byte-identical to the old corruption_factors builder.
    assert severity_grid(20) == torch.linspace(1.0 / 20, 1.0, 20).tolist()
    assert severity_grid(20)[0] == pytest.approx(0.05)
    assert severity_grid(20)[-1] == pytest.approx(1.0)


def test_severity_grid_rejects_degenerate_n() -> None:
    with pytest.raises(ValueError):
        severity_grid(1)


def test_severity_grid_honors_explicit_theta_min() -> None:
    g = severity_grid(10, theta_min=0.2)
    assert math.isclose(g[0], 0.2, abs_tol=1e-6)
    assert math.isclose(g[-1], 1.0, abs_tol=1e-6)


def _sweep_classes():
    pytest.importorskip("scripts.sim2rank.degradation")  # not in the public export
    from scripts.sim2rank.degradation import DegradationSweep, MetaDegradationSweep

    return DegradationSweep, MetaDegradationSweep


@pytest.mark.parametrize("n_steps", [5, 8, 13, 20, 32])
def test_two_sweep_grids_identical_at_every_n(n_steps) -> None:
    """The fix: the ordered ramp is identical by construction, not just at T=20."""
    sweep_cls, meta_cls = _sweep_classes()
    assert (
        sweep_cls(n_timesteps=n_steps).corruption_factors
        == meta_cls(n_timesteps=n_steps).severity_grid
    )


def test_pre_fix_grids_would_have_diverged_off_20() -> None:
    """Guards the regression: the OLD builders differ at T!=20."""
    n_steps = 8
    old_meta = torch.linspace(0.05, 1.0, n_steps).tolist()  # old MetaDegradationSweep
    old_cf = torch.linspace(1.0 / n_steps, 1.0, n_steps).tolist()  # old corruption
    assert old_meta != old_cf


def test_meta_grid_byte_identical_to_pre_fix_at_20() -> None:
    """Behavior preservation: at the production T=20 the meta grid is unchanged."""
    _, meta_cls = _sweep_classes()
    assert meta_cls(n_timesteps=20).severity_grid == torch.linspace(0.05, 1.0, 20).tolist()


def test_corruption_factors_byte_identical_to_pre_fix_at_20() -> None:
    sweep_cls, _ = _sweep_classes()
    assert (
        sweep_cls(n_timesteps=20).corruption_factors == torch.linspace(1.0 / 20, 1.0, 20).tolist()
    )
