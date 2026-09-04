r"""Tests for k-space trajectory generators.

Targets ``spectramr.infrastructure.physics.kspace_trajectory_generator``.

Categories:

- ``archimedean_spiral`` starts at the origin (low frequencies first)
  and reaches the configured ``max_k``
- ``radial_golden_angle``: every spoke passes through DC; angle
  increment matches ``π · (√5 - 1) / 2``
- ``epi_boustrophedon``: alternating-direction property holds
- ``cartesian_sequential``: sample shape and coverage
- ``create_trajectory`` factory dispatches to the right function
- Unknown name raises with the available list
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.kspace_trajectory_generator import (
    TRAJECTORY_REGISTRY,
    archimedean_spiral,
    cartesian_sequential,
    create_trajectory,
    epi_boustrophedon,
    radial_golden_angle,
)


# ---------------------------------------------------------------------------
# Archimedean spiral
# ---------------------------------------------------------------------------


def test_archimedean_starts_at_origin() -> None:
    """First sample of each arm sits at the k-space centre."""
    traj = archimedean_spiral(num_arms=4, points_per_arm=64, max_k=0.5)
    # Arm i occupies indices [i*N, (i+1)*N); first sample of each is at r=0.
    N = 64
    for i in range(4):
        kx, ky = traj[0, i * N], traj[1, i * N]
        assert pytest.approx(kx.item(), abs=1e-6) == 0.0
        assert pytest.approx(ky.item(), abs=1e-6) == 0.0


def test_archimedean_reaches_max_k() -> None:
    """The last sample of each arm has radius ``max_k``."""
    max_k = 0.5
    traj = archimedean_spiral(num_arms=2, points_per_arm=64, max_k=max_k)
    last_radii = []
    N = 64
    for i in range(2):
        kx, ky = traj[0, (i + 1) * N - 1], traj[1, (i + 1) * N - 1]
        last_radii.append(math.hypot(kx.item(), ky.item()))
    for r in last_radii:
        assert pytest.approx(r, abs=1e-5) == max_k


def test_archimedean_total_point_count() -> None:
    """Total sample count is ``num_arms × points_per_arm``."""
    traj = archimedean_spiral(num_arms=3, points_per_arm=64, max_k=0.5)
    assert traj.shape == (2, 192)


# ---------------------------------------------------------------------------
# Radial golden-angle
# ---------------------------------------------------------------------------


def test_radial_every_spoke_passes_through_dc() -> None:
    """Every radial spoke crosses the k-space origin (centre point of t)."""
    num_spokes, pts = 8, 33  # odd → centre is a sample
    traj = radial_golden_angle(num_spokes=num_spokes, points_per_spoke=pts, max_k=0.5)
    centre_index_per_spoke = pts // 2  # 16
    for i in range(num_spokes):
        kx = traj[0, i * pts + centre_index_per_spoke].item()
        ky = traj[1, i * pts + centre_index_per_spoke].item()
        assert abs(kx) < 1e-5 and abs(ky) < 1e-5


def test_radial_golden_angle_increment() -> None:
    """The angle between consecutive spokes equals ``π · (√5 - 1) / 2`` mod π."""
    expected = math.pi * (math.sqrt(5) - 1) / 2
    traj = radial_golden_angle(num_spokes=3, points_per_spoke=11, max_k=0.5)
    # Pick the rightmost (positive-radius) point of each spoke.
    pts = 11
    centre = pts // 2
    rightmost = pts - 1  # index of t = +max_k
    angles = []
    for i in range(3):
        kx = traj[0, i * pts + rightmost].item()
        ky = traj[1, i * pts + rightmost].item()
        angles.append(math.atan2(ky, kx))
    # Differences mod 2π should match the golden angle.
    for prev, nxt in zip(angles, angles[1:]):
        diff = (nxt - prev) % (2 * math.pi)
        # Tolerance: golden angle modulo 2π.
        assert pytest.approx(diff, abs=1e-5) == expected % (2 * math.pi)


def test_radial_total_point_count() -> None:
    """``num_spokes × points_per_spoke`` samples in total."""
    traj = radial_golden_angle(num_spokes=10, points_per_spoke=8, max_k=0.5)
    assert traj.shape == (2, 80)


# ---------------------------------------------------------------------------
# EPI boustrophedon
# ---------------------------------------------------------------------------


def test_epi_alternating_direction() -> None:
    """Even lines: kx ascends. Odd lines: kx descends."""
    num_lines, pts = 4, 8
    max_k = 0.5
    traj = epi_boustrophedon(num_lines=num_lines, points_per_line=pts, max_k=max_k)
    for i in range(num_lines):
        line_kx = traj[0, i * pts : (i + 1) * pts]
        if i % 2 == 0:
            assert (line_kx[1:] - line_kx[:-1] >= 0).all()
        else:
            assert (line_kx[1:] - line_kx[:-1] <= 0).all()


def test_epi_phase_encode_lines_evenly_spaced() -> None:
    """ky values trace the configured ``num_lines`` linearly-spaced offsets."""
    num_lines = 16
    traj = epi_boustrophedon(num_lines=num_lines, points_per_line=4, max_k=0.5)
    # Take the first sample of each line.
    first_ky = traj[1, ::4]  # one per line
    expected = torch.linspace(-0.5, 0.5, num_lines)
    assert torch.allclose(first_ky, expected)


# ---------------------------------------------------------------------------
# Cartesian sequential
# ---------------------------------------------------------------------------


def test_cartesian_total_point_count() -> None:
    """``num_pe_lines × num_fe_samples`` total samples."""
    traj = cartesian_sequential(num_pe_lines=16, num_fe_samples=32, max_k=0.5)
    assert traj.shape == (2, 16 * 32)


def test_cartesian_line_kx_constant_within_line() -> None:
    """Within a single PE line, ky is constant."""
    pts = 16
    traj = cartesian_sequential(num_pe_lines=4, num_fe_samples=pts, max_k=0.5)
    for i in range(4):
        ky_line = traj[1, i * pts : (i + 1) * pts]
        assert (ky_line == ky_line[0]).all()


# ---------------------------------------------------------------------------
# create_trajectory factory
# ---------------------------------------------------------------------------


def test_factory_dispatches_by_name() -> None:
    """Each registered name returns a 2×N trajectory tensor."""
    for name in TRAJECTORY_REGISTRY:
        traj = create_trajectory(name)
        assert traj.shape[0] == 2
        assert traj.shape[1] > 0


def test_factory_kwargs_forwarded() -> None:
    """``**kwargs`` reach the underlying generator."""
    traj = create_trajectory(
        "archimedean_spiral", num_arms=2, points_per_arm=8, max_k=0.5
    )
    assert traj.shape == (2, 16)


def test_factory_unknown_name_raises_with_available() -> None:
    """Unknown name raises with the canonical list."""
    with pytest.raises(ValueError, match="Available"):
        create_trajectory("not_a_trajectory")
