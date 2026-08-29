"""Tests for ``GoldenAngleTrajectory``.

Targets ``mriforge.infrastructure.physics.nufft_ops``. ``NUFFTForwardModel``
requires the optional ``torchkbnufft`` runtime; we cover trajectory
generation here (pure Torch) and defer NUFFT itself to the integration
tier where the runtime is guaranteed.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.nufft_ops import GOLDEN_ANGLE, GoldenAngleTrajectory


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_golden_angle_constant() -> None:
    """``GOLDEN_ANGLE = π·(√5 − 1)/2 ≈ 111.246° in radians`` (radial golden angle).

    Regression: the constant previously held ``π·(3 − √5) ≈ 137.508°`` (the 2-D
    phyllotaxis angle) while every docstring claimed 111.246°.
    """
    expected = math.pi * (math.sqrt(5.0) - 1.0) / 2.0
    assert pytest.approx(GOLDEN_ANGLE) == expected
    assert pytest.approx(math.degrees(GOLDEN_ANGLE), abs=1e-3) == 111.246


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_persists_geometry() -> None:
    """Constructor stores image size, spoke count, samples-per-spoke."""
    traj = GoldenAngleTrajectory(im_size=(128, 128), num_spokes=32, samples_per_spoke=64)
    assert traj.im_size == (128, 128)
    assert traj.num_spokes == 32
    assert traj.samples_per_spoke == 64


# ---------------------------------------------------------------------------
# generate / forward output
# ---------------------------------------------------------------------------


def test_generate_output_shape() -> None:
    """Output has shape ``[T, 2, num_spokes * samples_per_spoke]``."""
    traj = GoldenAngleTrajectory(num_spokes=8, samples_per_spoke=16)
    out = traj.generate(num_frames=3)
    assert out.shape == (3, 2, 8 * 16)


def test_generate_default_single_frame() -> None:
    """Default ``num_frames=1`` produces a single-frame trajectory."""
    traj = GoldenAngleTrajectory(num_spokes=4, samples_per_spoke=8)
    out = traj.generate()
    assert out.shape == (1, 2, 32)


def test_forward_is_alias_for_generate() -> None:
    """``forward`` produces the same output as ``generate``."""
    traj = GoldenAngleTrajectory(num_spokes=4, samples_per_spoke=8)
    out_gen = traj.generate(num_frames=2)
    out_fwd = traj.forward(num_frames=2)
    assert torch.equal(out_gen, out_fwd)


# ---------------------------------------------------------------------------
# Coordinate range
# ---------------------------------------------------------------------------


def test_trajectory_coordinates_in_pi_range() -> None:
    """All k-space coords are in ``[-π, π]`` (torchkbnufft convention)."""
    traj = GoldenAngleTrajectory(num_spokes=8, samples_per_spoke=16)
    out = traj.generate()
    assert out.max().item() <= math.pi + 1e-6
    assert out.min().item() >= -math.pi - 1e-6


def test_radial_spoke_passes_through_origin() -> None:
    """Each radial spoke crosses the k-space origin (centre sample = (0, 0))."""
    traj = GoldenAngleTrajectory(num_spokes=4, samples_per_spoke=15)  # odd → centre
    out = traj.generate()  # [1, 2, 4 * 15]
    # The centre sample of each spoke is at index 7 (out of 15).
    centre_idx = 7
    for spoke in range(4):
        kx = out[0, 0, spoke * 15 + centre_idx].item()
        ky = out[0, 1, spoke * 15 + centre_idx].item()
        assert abs(kx) < 1e-6
        assert abs(ky) < 1e-6


# ---------------------------------------------------------------------------
# Golden-angle property
# ---------------------------------------------------------------------------


def test_first_two_spokes_separated_by_golden_angle() -> None:
    """Angle between spokes 0 and 1 equals ``GOLDEN_ANGLE`` (mod π)."""
    traj = GoldenAngleTrajectory(num_spokes=4, samples_per_spoke=4)
    out = traj.generate()
    # Sample at the end of each spoke (index 3): kx = π·cos(angle), ky = π·sin(angle)
    end0 = out[0, :, 3]   # spoke 0
    end1 = out[0, :, 7]   # spoke 1 (next 4 samples)
    angle0 = math.atan2(end0[1].item(), end0[0].item())
    angle1 = math.atan2(end1[1].item(), end1[0].item())
    diff = (angle1 - angle0) % math.pi  # spokes are signed-symmetric
    expected = GOLDEN_ANGLE % math.pi
    assert pytest.approx(diff, abs=1e-3) == expected


# ---------------------------------------------------------------------------
# spoke_offset & multi-frame indexing
# ---------------------------------------------------------------------------


def test_spoke_offset_shifts_starting_angle() -> None:
    """Non-zero ``spoke_offset`` produces a different starting angle."""
    traj = GoldenAngleTrajectory(num_spokes=4, samples_per_spoke=4)
    out_a = traj.generate(spoke_offset=0)
    out_b = traj.generate(spoke_offset=2)
    # Different starting angles → different trajectories
    assert not torch.allclose(out_a, out_b)


def test_multi_frame_trajectories_are_distinct() -> None:
    """Frame 1 spokes are rotated relative to frame 0 — different coords."""
    traj = GoldenAngleTrajectory(num_spokes=4, samples_per_spoke=4)
    out = traj.generate(num_frames=2)
    assert not torch.allclose(out[0], out[1])


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spokes, samples, frames",
    [(8, 16, 1), (16, 32, 2), (4, 8, 4)],
    ids=["s8_n16_f1", "s16_n32_f2", "s4_n8_f4"],
)
def test_trajectory_shape_matrix(spokes: int, samples: int, frames: int) -> None:
    """Trajectory shape is ``[T, 2, spokes * samples]`` across the matrix."""
    traj = GoldenAngleTrajectory(num_spokes=spokes, samples_per_spoke=samples)
    out = traj.generate(num_frames=frames)
    assert out.shape == (frames, 2, spokes * samples)
    assert torch.isfinite(out).all()
