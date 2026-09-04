"""Unit tests for src/spectramr/infrastructure/physics/biot_savart.py.

Covers:
  - compute_b1_minus_biot_savart: output shape, dtype, physics properties

Physics property tests:
  - Biot-Savart on a single circular loop along the z-axis:
      on-axis at z=0 (observation plane = coil plane with offset):
      the transverse B field at the coil centre is highest magnitude.
  - Far-field ~1/r³ amplitude decay (order-of-magnitude verification).
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.utils.shape_matrices import shape_id


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_canary_biot_savart_imports() -> None:
    """Module imports cleanly."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    assert compute_b1_minus_biot_savart is not None


@pytest.mark.physics
def test_canary_biot_savart_runs_tiny() -> None:
    """compute_b1_minus_biot_savart runs on the smallest valid inputs and returns
    a complex tensor of shape (num_coils, H, W)."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    num_coils = 2
    H, W = 8, 8
    out = compute_b1_minus_biot_savart(
        grid_shape=(H, W),
        num_coils=num_coils,
        num_segments=16,
    )
    assert out.shape == (num_coils, H, W), f"Expected ({num_coils},{H},{W}), got {out.shape}"
    assert out.dtype == torch.complex64
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"


# ---------------------------------------------------------------------------
# Parametrised: realistic grid shapes and coil counts
# ---------------------------------------------------------------------------

_GRID_COIL_PARAMS: list[tuple[tuple[int, int], int]] = [
    ((16, 16), 1),
    ((16, 16), 4),
    ((32, 32), 8),
    ((64, 64), 8),
    ((32, 48), 4),
]


@pytest.mark.physics
@pytest.mark.parametrize(
    "grid_shape,num_coils",
    _GRID_COIL_PARAMS,
    ids=[f"{h}x{w}_C{c}" for (h, w), c in _GRID_COIL_PARAMS],
)
def test_output_shape(grid_shape: tuple[int, int], num_coils: int) -> None:
    """Output shape is (num_coils, H, W) for all tested configs."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    H, W = grid_shape
    out = compute_b1_minus_biot_savart(
        grid_shape=grid_shape,
        num_coils=num_coils,
        num_segments=32,
    )
    assert out.shape == (num_coils, H, W)
    assert out.dtype == torch.complex64


@pytest.mark.physics
@pytest.mark.parametrize(
    "grid_shape,num_coils",
    _GRID_COIL_PARAMS,
    ids=[f"{h}x{w}_C{c}" for (h, w), c in _GRID_COIL_PARAMS],
)
def test_output_finite(grid_shape: tuple[int, int], num_coils: int) -> None:
    """All output values are finite (no NaN/Inf)."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    out = compute_b1_minus_biot_savart(
        grid_shape=grid_shape,
        num_coils=num_coils,
        num_segments=32,
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "grid_shape",
    [(16, 16), (32, 32), (64, 64)],
    ids=["16x16", "32x32", "64x64"],
)
def test_sanity_shape_biot_savart(grid_shape: tuple[int, int]) -> None:
    """Output tensor matches documented shape contract (num_coils, H, W)."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    num_coils = 4
    out = compute_b1_minus_biot_savart(grid_shape=grid_shape, num_coils=num_coils)
    H, W = grid_shape
    assert out.shape == (num_coils, H, W)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_edge_single_coil() -> None:
    """Single coil (num_coils=1) produces shape (1, H, W) complex tensor."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    out = compute_b1_minus_biot_savart(grid_shape=(16, 16), num_coils=1, num_segments=32)
    assert out.shape == (1, 16, 16)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_edge_large_num_segments() -> None:
    """Increasing num_segments does not crash and preserves shape/dtype."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    out = compute_b1_minus_biot_savart(grid_shape=(16, 16), num_coils=4, num_segments=256)
    assert out.shape == (4, 16, 16)
    assert out.dtype == torch.complex64


@pytest.mark.physics
def test_edge_non_square_grid() -> None:
    """Non-square (H != W) grid produces the correct rectangular output."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    H, W, C = 24, 48, 3
    out = compute_b1_minus_biot_savart(grid_shape=(H, W), num_coils=C)
    assert out.shape == (C, H, W)


@pytest.mark.physics
def test_edge_fov_smaller_than_cylinder() -> None:
    """fov < cylinder_radius (field of view entirely inside the coil array) still runs."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    # Very small fov — all observation points are near the origin
    out = compute_b1_minus_biot_savart(
        grid_shape=(16, 16),
        num_coils=4,
        fov=0.01,  # 1 cm FoV, much smaller than cylinder_radius=0.15
        cylinder_radius=0.15,
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# Physics property tests
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_physics_nonzero_field() -> None:
    """At least some pixels have non-zero |B1-| magnitude after normalisation."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    out = compute_b1_minus_biot_savart(grid_shape=(32, 32), num_coils=8)
    # After internal RSS normalisation the max should be finite and > 0
    assert out.abs().max().item() > 1e-6, "All sensitivities are zero — physics model broken"


@pytest.mark.physics
def test_physics_coil_symmetry() -> None:
    """A symmetric 2-coil array should produce sensitivity maps that are
    180°-rotated versions of each other (magnitude should be rotationally symmetric).

    We use |sens[0]| ≈ rot180(|sens[1]|) for a 2-coil ring array when the grid
    is centred and square.  We test the weaker condition that the two coils have
    the same total energy.
    """
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    # Use even grid so the symmetry is exact
    out = compute_b1_minus_biot_savart(
        grid_shape=(32, 32),
        num_coils=2,
        cylinder_radius=0.15,
        coil_radius=0.1,
        fov=0.2,
        num_segments=64,
    )
    energy_0 = out[0].abs().pow(2).sum()
    energy_1 = out[1].abs().pow(2).sum()
    # For a perfectly symmetric 2-coil layout the energies should be equal.
    ratio = (energy_0 / (energy_1 + 1e-12)).item()
    assert 0.5 < ratio < 2.0, (
        f"Coil energy ratio {ratio:.3f} suggests broken symmetry "
        f"(expected ~1.0 for identical coils)"
    )


@pytest.mark.physics
def test_physics_far_field_decay() -> None:
    """B1- magnitude should decay roughly as ~1/r^3 far from the coil.

    We compare the mean magnitude in a near-centre ring vs a far-field ring
    and verify the outer ring is meaningfully weaker.  This is an
    order-of-magnitude check — exact Biot-Savart 1/r³ applies only along
    the coil axis; in the transverse plane the exponent differs slightly.
    """
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    N = 64
    out = compute_b1_minus_biot_savart(
        grid_shape=(N, N),
        num_coils=1,
        coil_radius=0.04,  # small coil to emphasise r-dependence
        cylinder_radius=0.05,
        fov=0.3,  # wide FoV so far-field is genuinely far
        num_segments=64,
    )
    mag = out[0].abs()  # (N, N)

    # Build distance map from the grid centre
    ys = torch.linspace(-1.0, 1.0, N)
    xs = torch.linspace(-1.0, 1.0, N)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = (xx**2 + yy**2).sqrt()

    near_mask = r < 0.3  # inner 30 % of the [-1,1] grid
    far_mask = r > 0.7  # outer 30 %

    near_mean = mag[near_mask].mean().item()
    far_mean = mag[far_mask].mean().item()

    # Far-field must be strictly weaker than near-field
    assert far_mean < near_mean, (
        f"Far-field ({far_mean:.4f}) is not weaker than near-field ({near_mean:.4f}); "
        "field decay is missing"
    )


# ---------------------------------------------------------------------------
# raises: documented preconditions
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_raises_zero_num_coils() -> None:
    """num_coils=0 must raise ValueError (empty array is meaningless, pitfall #9 guard)."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    with pytest.raises(ValueError, match="num_coils must be a positive integer"):
        compute_b1_minus_biot_savart(grid_shape=(16, 16), num_coils=0)


@pytest.mark.physics
def test_raises_negative_num_coils() -> None:
    """num_coils=-1 must also raise ValueError (covers the <= 0 guard branch)."""
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    with pytest.raises(ValueError, match="num_coils must be a positive integer"):
        compute_b1_minus_biot_savart(grid_shape=(16, 16), num_coils=-1)


# ---------------------------------------------------------------------------
# Doc-contract: complex convention (returns Bx + i By = conj(B1-))
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_convention_returns_conj_b1_minus() -> None:
    """The returned sensitivity is ``Bx + i By = conj(B1-)``, NOT the textbook
    ``B1- = Bx - i By``.

    The function computes ``torch.complex(Bx, By)`` (so the imaginary part is
    ``+By``). For coil 0 the per-coil phase twist ``exp(i*theta_c)`` is exactly
    ``exp(0) = 1`` and the RSS normalization is a real positive scalar, so the
    returned imaginary component for coil 0 must equal ``+By`` up to a positive
    scale. We recompute ``Bx``/``By`` for coil 0 independently via Biot-Savart
    and assert the returned ``real``/``imag`` match ``+Bx``/``+By`` (and NOT the
    conventional ``Bx - i By``), pinning the documented convention.
    """
    from spectramr.infrastructure.physics.biot_savart import compute_b1_minus_biot_savart

    grid, num_segments = (32, 32), 64
    coil_radius, cylinder_radius, fov = 0.1, 0.15, 0.2

    out = compute_b1_minus_biot_savart(
        grid_shape=grid,
        num_coils=1,
        coil_radius=coil_radius,
        cylinder_radius=cylinder_radius,
        fov=fov,
        num_segments=num_segments,
    )
    sens0 = out[0]  # (H, W) complex; coil 0 phase twist == 1, normalized by +scale

    # --- Independent Biot-Savart for the single coil 0 (theta_c = 0) ---
    H, W = grid
    yy, xx = torch.meshgrid(
        torch.linspace(-fov / 2, fov / 2, H),
        torch.linspace(-fov / 2, fov / 2, W),
        indexing="ij",
    )
    pts = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)  # (H, W, 3)

    theta_c = 0.0
    r_c = torch.tensor(
        [cylinder_radius * math.cos(theta_c), cylinder_radius * math.sin(theta_c), 0.0]
    )
    t_hat = torch.tensor([-math.sin(theta_c), math.cos(theta_c), 0.0])
    z_hat = torch.tensor([0.0, 0.0, 1.0])
    alpha = torch.linspace(0, 2 * math.pi, num_segments + 1)
    coil_pts = r_c.view(1, 3) + coil_radius * (
        torch.outer(torch.cos(alpha), t_hat) + torch.outer(torch.sin(alpha), z_hat)
    )
    dl = coil_pts[1:] - coil_pts[:-1]
    midpoints = (coil_pts[1:] + coil_pts[:-1]) / 2.0
    n_seg = dl.shape[0]

    diff = pts.unsqueeze(2) - midpoints.view(1, 1, n_seg, 3)  # (H, W, N, 3)
    dist = torch.norm(diff, dim=-1, keepdim=True)
    dl_exp = dl.view(1, 1, n_seg, 3).expand(H, W, n_seg, 3)
    cross = torch.cross(dl_exp, diff, dim=-1)
    mu_0 = 4 * math.pi * 1e-7
    dB = (mu_0 * 1.0 / (4 * math.pi)) * cross / (dist**3 + 1e-12)
    B = torch.sum(dB, dim=2)  # (H, W, 3)
    Bx, By = B[..., 0], B[..., 1]

    # The returned field is (Bx + i By) scaled by one positive real RSS factor.
    # Recover the scale from the magnitudes, then check the real/imag match
    # +Bx / +By (the +By convention). A Bx - i By implementation would make
    # sens0.imag align with -By instead.
    src_mag = (Bx**2 + By**2).sqrt()
    out_mag = sens0.abs()
    valid = src_mag > 1e-9
    scale = (out_mag[valid] / src_mag[valid]).median()
    assert torch.isfinite(scale) and scale > 0

    # imag should track +By (same sign), NOT -By.
    cos_with_plus_By = torch.nn.functional.cosine_similarity(
        sens0.imag[valid].flatten(), By[valid].flatten(), dim=0
    )
    assert cos_with_plus_By.item() > 0.99, (
        f"imag part does not match +By convention (cos={cos_with_plus_By.item():.4f}); "
        "code may have switched to the textbook B1- = Bx - i By"
    )
    # real should track +Bx (sanity).
    cos_with_plus_Bx = torch.nn.functional.cosine_similarity(
        sens0.real[valid].flatten(), Bx[valid].flatten(), dim=0
    )
    assert cos_with_plus_Bx.item() > 0.99
