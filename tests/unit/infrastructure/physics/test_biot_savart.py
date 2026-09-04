"""Tests for ``compute_b1_minus_biot_savart``.

Targets ``spectramr.infrastructure.physics.biot_savart``. The Biot-Savart
helper synthesises analytical B1- receive sensitivity maps for a
cylindrical coil array — used by the digital twin and as a synthetic
ground truth in coil-sensitivity-estimation tests.

Categories:

- Unit: output dtype, shape, finiteness.
- Property: rotational symmetry of the cylindrical array.
- Sanity-shape: parametrised grid sizes.
- Edge: single coil, very small grid.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart


def test_returns_complex_tensor_with_expected_shape() -> None:
    """Output is complex64 with shape ``(num_coils, H, W)``."""
    out = compute_b1_minus_biot_savart(
        grid_shape=(16, 16), num_coils=4, num_segments=16
    )
    assert out.shape == (4, 16, 16)
    assert out.dtype == torch.complex64


def test_output_is_finite_no_divide_by_zero() -> None:
    """The 1e-12 floor in ``dist**3`` prevents NaN/inf even at coil centres."""
    out = compute_b1_minus_biot_savart(
        grid_shape=(8, 8), num_coils=2, coil_radius=0.05, num_segments=16
    )
    assert torch.isfinite(out).all()


def test_output_magnitude_nonzero() -> None:
    """A coil array with non-zero current produces non-zero B1- everywhere."""
    out = compute_b1_minus_biot_savart(
        grid_shape=(16, 16), num_coils=4, num_segments=16
    )
    # Allow some pixels at coil centres to be near zero, but the array
    # average must be positive.
    assert out.abs().mean().item() > 0


@pytest.mark.parametrize(
    "grid_shape,num_coils",
    [((8, 8), 1), ((16, 16), 4), ((32, 32), 8), ((16, 32), 4)],
    ids=lambda v: f"{v}",
)
def test_shape_matrix_grid_and_coils(
    grid_shape: tuple[int, int], num_coils: int
) -> None:
    """Output shape tracks grid_shape and num_coils across the matrix."""
    out = compute_b1_minus_biot_savart(
        grid_shape=grid_shape, num_coils=num_coils, num_segments=16
    )
    assert out.shape == (num_coils, *grid_shape)
    assert torch.isfinite(out).all()


def test_single_coil_array_does_not_crash() -> None:
    """num_coils=1 is a degenerate but valid configuration."""
    out = compute_b1_minus_biot_savart(
        grid_shape=(8, 8), num_coils=1, num_segments=16
    )
    assert out.shape == (1, 8, 8)
    assert torch.isfinite(out).all()


def test_far_field_decays_with_distance() -> None:
    """Pixels far from any coil should have lower magnitude than near pixels.

    The cylinder is centred at the origin; coils sit at radius
    ``cylinder_radius``. Pixels at the far edge of the FOV (outside the
    cylinder) should see weaker fields than pixels at the cylinder's edge.
    """
    out = compute_b1_minus_biot_savart(
        grid_shape=(32, 32),
        num_coils=8,
        cylinder_radius=0.05,
        fov=0.4,  # FOV >> cylinder so edges are clearly far
        num_segments=32,
    )
    # Sum across coils to see total field strength per pixel.
    total = out.abs().sum(dim=0)
    centre = total[16, 16].item()
    corner = total[0, 0].item()
    assert centre > corner, (
        f"centre field {centre:.4f} should exceed corner {corner:.4f}"
    )


def test_rotational_symmetry_of_cylindrical_array() -> None:
    """A cylindrical array is rotation-symmetric: total |B1| is symmetric.

    Not a per-coil check (that would be permutation), but the
    sum-over-coils magnitude should be near-rotationally-symmetric about
    the cylinder axis.
    """
    out = compute_b1_minus_biot_savart(
        grid_shape=(32, 32), num_coils=8, num_segments=32, fov=0.2
    )
    total = out.abs().sum(dim=0)  # [H, W]
    # Compare two opposite quadrants — should be approximately equal.
    q_top_left = total[:16, :16].mean().item()
    q_bottom_right = total[16:, 16:].mean().item()
    rel_err = abs(q_top_left - q_bottom_right) / (q_top_left + 1e-9)
    # Loose tolerance: discretisation breaks exact symmetry.
    assert rel_err < 0.10, f"asymmetry rel_err = {rel_err:.3f}"


def test_device_argument_routed() -> None:
    """``device='cpu'`` keeps the result on CPU."""
    out = compute_b1_minus_biot_savart(
        grid_shape=(8, 8), num_coils=2, num_segments=16, device="cpu"
    )
    assert out.device.type == "cpu"
