"""Tests for ``B0MapSimulator`` and field-strength helpers.

Targets ``mriforge.infrastructure.physics.field_simulation``. Generates
synthetic B0 inhomogeneity maps for multi-field-strength training.

Categories:

- Unit: ``FieldStrength`` enum values, ``get_max_hz_for_field`` lookup
- Property: smooth/anatomical maps stay within ±max_hz×scale
- Property: linear gradient is monotone along the requested axis
- Sanity-shape: every generator returns ``[1, 1, H, W]``
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.field_simulation import (
    FIELD_STRENGTH_HZ_MAP,
    B0MapSimulator,
    FieldStrength,
    get_max_hz_for_field,
)

# ---------------------------------------------------------------------------
# Field strength enum + lookup
# ---------------------------------------------------------------------------


def test_field_strength_enum_covers_documented_strengths() -> None:
    """Enum has 3.0T, 1.5T, 0.55T, 0.35T, 0.064T."""
    values = {f.value for f in FieldStrength}
    assert values == {3.0, 1.5, 0.55, 0.35, 0.064}


def test_get_max_hz_lookup_matches_table() -> None:
    """For tabulated strengths, return the table value verbatim."""
    for k, v in FIELD_STRENGTH_HZ_MAP.items():
        assert get_max_hz_for_field(k) == v


def test_get_max_hz_for_unknown_strength_is_linear() -> None:
    """Unknown strength → linear ~83.3 Hz / Tesla."""
    out = get_max_hz_for_field(7.0)  # 7T not in table
    assert pytest.approx(out, rel=1e-3) == 83.3 * 7.0


# ---------------------------------------------------------------------------
# B0MapSimulator construction
# ---------------------------------------------------------------------------


def test_construction_persists_field_strength() -> None:
    """Constructor stores field strength and corresponding max_hz."""
    sim = B0MapSimulator(field_strength=1.5)
    assert sim.field_strength == 1.5
    assert sim.max_hz == FIELD_STRENGTH_HZ_MAP[1.5]


# ---------------------------------------------------------------------------
# generate_smooth_b0
# ---------------------------------------------------------------------------


def test_smooth_b0_shape_and_range() -> None:
    """Smooth B0 has shape ``[1, 1, H, W]`` and stays roughly within ±max_hz."""
    out = B0MapSimulator.generate_smooth_b0((16, 16), max_hz=100.0, seed=0)
    assert out.shape == (1, 1, 16, 16)
    # After normalisation × max_hz, |value| should be O(max_hz). Allow factor
    # of 5× because z-score normalisation can produce extreme values for tiny
    # 16×16 noise grids.
    assert out.abs().max().item() < 500.0


def test_smooth_b0_seed_reproducibility() -> None:
    """Same seed → same map."""
    out1 = B0MapSimulator.generate_smooth_b0((8, 8), max_hz=50.0, seed=42)
    out2 = B0MapSimulator.generate_smooth_b0((8, 8), max_hz=50.0, seed=42)
    assert torch.equal(out1, out2)


def test_smooth_b0_zero_mean() -> None:
    """The smooth map is normalised to zero mean before scaling."""
    out = B0MapSimulator.generate_smooth_b0((32, 32), max_hz=100.0, seed=0)
    # Mean should be zero modulo float-precision (× max_hz scale)
    assert abs(out.mean().item()) < 1e-3


# ---------------------------------------------------------------------------
# generate_anatomical_b0
# ---------------------------------------------------------------------------


def test_anatomical_b0_shape() -> None:
    """Anatomical B0 has shape ``[1, 1, H, W]``."""
    out = B0MapSimulator.generate_anatomical_b0(
        (16, 16), max_hz=100.0, hotspot_count=2, seed=0
    )
    assert out.shape == (1, 1, 16, 16)


def test_anatomical_b0_clamped_to_safe_range() -> None:
    """Output is clamped to ``±1.5 × max_hz``."""
    out = B0MapSimulator.generate_anatomical_b0(
        (16, 16), max_hz=100.0, hotspot_count=5, seed=0
    )
    assert out.abs().max().item() <= 100.0 * 1.5 + 1e-5


# ---------------------------------------------------------------------------
# generate_linear_gradient
# ---------------------------------------------------------------------------


def test_linear_gradient_y_direction() -> None:
    """Y-direction gradient is monotone along axis 0."""
    out = B0MapSimulator.generate_linear_gradient(
        (16, 16), gradient_hz_per_pixel=1.0, direction="y"
    )
    col = out[0, 0, :, 0]
    # Sorted ascending → monotone non-decreasing
    diffs = col[1:] - col[:-1]
    assert (diffs > 0).all()


def test_linear_gradient_x_direction() -> None:
    """X-direction gradient is monotone along axis 1."""
    out = B0MapSimulator.generate_linear_gradient(
        (16, 16), gradient_hz_per_pixel=1.0, direction="x"
    )
    row = out[0, 0, 0, :]
    diffs = row[1:] - row[:-1]
    assert (diffs > 0).all()


def test_linear_gradient_zero_pixel_value_is_zero_at_centre() -> None:
    """Centre row/col of the gradient is at zero."""
    out = B0MapSimulator.generate_linear_gradient(
        (17, 17), gradient_hz_per_pixel=1.0, direction="y"
    )
    centre = out[0, 0, 8, 0].item()
    assert abs(centre) < 1e-5


def test_linear_gradient_step_is_exactly_per_pixel() -> None:
    """Adjacent-pixel field difference equals gradient_hz_per_pixel exactly.

    Regression: linspace(-N/2, N/2, N) has a per-sample step of N/(N-1),
    so the field changed by gradient_hz_per_pixel·N/(N-1) per pixel — the
    parameter was mislabeled. The centered unit-step ramp makes it exact.
    """
    for size in (16, 17, 33):
        out = B0MapSimulator.generate_linear_gradient(
            (size, size), gradient_hz_per_pixel=2.5, direction="y"
        )
        col = out[0, 0, :, 0]
        steps = col[1:] - col[:-1]
        assert torch.allclose(steps, torch.full_like(steps, 2.5), atol=1e-5)


def test_linear_gradient_unknown_direction_raises() -> None:
    """An unknown direction must raise, not silently fall back to x."""
    with pytest.raises(ValueError, match="direction must be"):
        B0MapSimulator.generate_linear_gradient((16, 16), direction="z")


# ---------------------------------------------------------------------------
# generate_concomitant_field (Maxwell terms)
# ---------------------------------------------------------------------------


def test_concomitant_field_zero_gradients_zero_field() -> None:
    """No gradients → no concomitant field."""
    out = B0MapSimulator.generate_concomitant_field(
        (16, 16), field_strength=0.064, gx=0.0, gy=0.0, gz=0.0
    )
    assert torch.allclose(out, torch.zeros_like(out))


def test_concomitant_field_gz_only_radial() -> None:
    """Pure Gz at isocentre → radially symmetric field (longitudinal term)."""
    out = B0MapSimulator.generate_concomitant_field(
        (16, 16), field_strength=0.064, gz=10.0, z_offset=0.0
    )
    # By symmetry, value at (8, 8) and (8, 0) along same radius should differ;
    # but value at corners (15, 15) ≈ value at (0, 0) (radial).
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# generate_for_field — top-level dispatcher
# ---------------------------------------------------------------------------


def test_generate_for_field_smooth_style() -> None:
    """``style='smooth'`` produces a finite map of correct shape."""
    sim = B0MapSimulator(field_strength=3.0)
    out = sim.generate_for_field((16, 16), style="smooth")
    assert out.shape[-2:] == (16, 16)
    assert torch.isfinite(out).all()


def test_generate_for_field_anatomical_style() -> None:
    """``style='anatomical'`` produces a finite map of correct shape."""
    sim = B0MapSimulator(field_strength=1.5)
    out = sim.generate_for_field((16, 16), style="anatomical")
    assert out.shape[-2:] == (16, 16)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(8, 8), (16, 32), (32, 16), (64, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_smooth_b0_shape_matrix(shape: tuple[int, int]) -> None:
    """Smooth B0 returns ``[1, 1, H, W]`` across the matrix."""
    out = B0MapSimulator.generate_smooth_b0(shape, max_hz=50.0, seed=0)
    assert out.shape == (1, 1, *shape)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# include_maxwell must actually add a non-zero ULF concomitant field
# (2026-06-12 audit: gz=0 made it a silent no-op)
# ---------------------------------------------------------------------------


def test_generate_for_field_maxwell_is_nonzero_at_ulf() -> None:
    """At ULF, include_maxwell=True changes the map vs False (was inert)."""
    sim = B0MapSimulator(field_strength=0.064)
    torch.manual_seed(0)
    with_mx = sim.generate_for_field((32, 32), include_maxwell=True)
    torch.manual_seed(0)
    without_mx = sim.generate_for_field((32, 32), include_maxwell=False)
    assert not torch.allclose(with_mx, without_mx)


def test_generate_batch_maxwell_is_nonzero_at_ulf() -> None:
    sim = B0MapSimulator(field_strength=0.064)
    torch.manual_seed(1)
    with_mx = sim.generate_batch(2, (32, 32), include_maxwell=True)
    torch.manual_seed(1)
    without_mx = sim.generate_batch(2, (32, 32), include_maxwell=False)
    assert not torch.allclose(with_mx, without_mx)


def test_maxwell_is_off_above_ulf_threshold() -> None:
    """Above 0.1T the Maxwell branch is intentionally skipped."""
    sim = B0MapSimulator(field_strength=3.0)
    torch.manual_seed(2)
    with_mx = sim.generate_for_field((32, 32), include_maxwell=True)
    torch.manual_seed(2)
    without_mx = sim.generate_for_field((32, 32), include_maxwell=False)
    assert torch.allclose(with_mx, without_mx)


def test_zero_concomitant_gradient_makes_maxwell_inert() -> None:
    """Documents the knob: gz=0 → the term is identically zero (the old bug)."""
    sim = B0MapSimulator(field_strength=0.064, concomitant_gradient_mt_m=0.0)
    torch.manual_seed(3)
    with_mx = sim.generate_for_field((32, 32), include_maxwell=True)
    torch.manual_seed(3)
    without_mx = sim.generate_for_field((32, 32), include_maxwell=False)
    assert torch.allclose(with_mx, without_mx)


def test_b0_augmentation_transform_maxwell_is_nonzero_at_ulf() -> None:
    """The augmentation transform must actually add a non-zero Maxwell term at
    ULF. It previously called generate_concomitant_field WITHOUT gz, so the
    concomitant field defaulted to zero — an advertised no-op (pitfall #16).

    Seeding identically before each call makes the random smooth-B0 base and the
    prob/smoothness draws identical, so any difference is the Maxwell term.
    """
    from mriforge.infrastructure.physics.field_simulation import B0AugmentationTransform

    def _run(include_maxwell: bool) -> torch.Tensor:
        torch.manual_seed(7)
        tf = B0AugmentationTransform(
            field_strength=0.064, prob=1.0, include_maxwell=include_maxwell
        )
        sample = {"image": torch.zeros(1, 32, 32)}
        return tf(sample)["b0_map"]

    with_mx = _run(True)
    without_mx = _run(False)
    assert not torch.allclose(with_mx, without_mx), (
        "B0AugmentationTransform Maxwell term is inert at ULF (gz not passed)"
    )


# ---------------------------------------------------------------------------
# Regression: concomitant cross-term shares the 1/(2 B0) prefactor
# ---------------------------------------------------------------------------
# Bernstein (1998): Bc = (1/2B0)[(Gx²+Gy²)z² + Gz²(x²+y²)/4 − Gx Gz x z − Gy Gz y z].
# The cross term previously divided by B0 (not 2 B0), doubling it. It is only
# active when z_offset ≠ 0 AND a transverse gradient is set.


def test_concomitant_field_matches_bernstein_closed_form() -> None:
    """Full field (incl. cross term) equals the /(2·B0) Bernstein formula."""
    H = W = 24
    B0, gx, gz, z_off = 0.064, 12.0, 20.0, 0.03  # T, mT/m, mT/m, m
    field = B0MapSimulator.generate_concomitant_field(
        (H, W), field_strength=B0, gx=gx, gy=0.0, gz=gz, z_offset=z_off
    )
    fov = 0.25
    y = torch.linspace(-fov / 2, fov / 2, H)
    x = torch.linspace(-fov / 2, fov / 2, W)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    Gx, Gz = gx * 1e-3, gz * 1e-3
    delta_b = (
        (Gx**2) * z_off**2 / (2 * B0)
        + Gz**2 * (xx**2 + yy**2) / (8 * B0)
        - (Gx * xx) * Gz * z_off / (2 * B0)  # cross term: /(2 B0), not /B0
    )
    expected = (delta_b * 42.576e6).unsqueeze(0).unsqueeze(0)
    assert torch.allclose(field, expected, atol=1e-3)


def test_concomitant_cross_term_uses_half_not_full_prefactor() -> None:
    """The corrected field equals the old /B0 field plus half the cross term.

    Old (buggy) cross term: ``-(Gx·x)·Gz·z / B0``. Correct: ``… / (2·B0)``.
    Correct = buggy + ``(Gx·x)·Gz·z / (2·B0)`` — the fix halves the cross term.
    Uses ``z_offset ≠ 0`` + a transverse gradient so the term is active.
    """
    H = W = 8
    B0, gx, gz, z_off = 0.064, 15.0, 15.0, 0.04
    field = B0MapSimulator.generate_concomitant_field(
        (H, W), field_strength=B0, gx=gx, gy=0.0, gz=gz, z_offset=z_off
    )
    fov = 0.25
    y = torch.linspace(-fov / 2, fov / 2, H)
    x = torch.linspace(-fov / 2, fov / 2, W)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    Gx, Gz = gx * 1e-3, gz * 1e-3
    buggy_delta = (
        (Gx**2) * z_off**2 / (2 * B0)
        + Gz**2 * (xx**2 + yy**2) / (8 * B0)
        - (Gx * xx) * Gz * z_off / B0  # the old, doubled cross term (/B0)
    )
    buggy = (buggy_delta * 42.576e6).unsqueeze(0).unsqueeze(0)
    half_cross = (
        ((Gx * xx) * Gz * z_off / (2 * B0) * 42.576e6).unsqueeze(0).unsqueeze(0)
    )
    assert torch.allclose(field, buggy + half_cross, atol=1e-3)
    # And the cross term genuinely contributes (guard against a trivial pass).
    assert half_cross.abs().max() > 1.0
