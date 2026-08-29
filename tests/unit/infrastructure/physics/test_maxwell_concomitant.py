"""Unit tests for the Maxwell concomitant frequency-offset map.

Pin the physically-meaningful properties:

- Field-strength inverse scaling: `B_c ∝ 1/B_0`. So at 3T the
  concomitant offset is ~10× smaller than at 0.3T (M4Raw) and
  ~50× smaller than at 0.064T (Hyperfine).
- Spatial dependence: the term scales as `z^2` (transverse Maxwell)
  and `x^2 + y^2` (longitudinal Maxwell). At isocenter (centre voxel)
  the offset is exactly zero.
- Output shape and dtype contracts.
"""

from __future__ import annotations

import pytest
import torch

torch = pytest.importorskip("torch")

from mriforge.infrastructure.physics.maxwell_concomitant import (  # noqa: E402
    maxwell_concomitant_field_hz,
)


def _call(B0: float, gradients=(20.0, 20.0, 20.0), shape=(8, 16, 16)) -> torch.Tensor:
    return maxwell_concomitant_field_hz(
        spatial_shape=shape,
        voxel_size_mm=(2.0, 1.0, 1.0),
        gradient_amplitudes_mT_per_m=gradients,
        field_strength_t=B0,
    )


def test_zero_at_isocenter() -> None:
    """Maxwell concomitant offset is minimal at isocentre and exactly
    symmetric around it.

    Bernstein's formula ``B_c = ((Gx²+Gy²)z² + ¼Gz²(x²+y²)) / 2B0`` is
    strictly non-negative — it is zero ONLY at the geometric origin.
    For odd dims the centre voxel coincides with the origin and the
    field is exactly zero there. For even dims the origin falls
    between voxels, so the four (eight) voxels around the centre are
    all equal to each other (by quadratic symmetry) and positive — the
    test verifies that symmetry + the minimum property, not "mean = 0"
    (which would only hold for an odd function, which Bc is not).
    """
    f = _call(B0=0.064)
    D, H, W = f.shape
    cz, cy, cx = (D - 1) // 2, (H - 1) // 2, (W - 1) // 2
    # Odd-dim case: centre voxel coincides with isocentre exactly.
    if D % 2 == 1 and H % 2 == 1 and W % 2 == 1:
        assert f[cz, cy, cx].item() == pytest.approx(0.0, abs=1e-6)
        return
    # Even-dim case: 2×2×2 block around centre — must all be equal
    # (quadratic symmetry) and strictly less than any voxel one step away.
    block = f[cz : cz + 2, cy : cy + 2, cx : cx + 2]
    assert torch.allclose(block, block[0, 0, 0].expand_as(block), atol=1e-6), (
        "voxels around isocentre must be symmetric"
    )
    centre_value = block[0, 0, 0].item()
    assert centre_value >= 0.0, "Bernstein form is non-negative"
    # The voxel one step away in z should have strictly larger B_c.
    if cz + 2 < D:
        assert f[cz + 2, cy, cx].item() > centre_value


def test_field_strength_inverse_scaling() -> None:
    """Doubling B0 must halve the Maxwell offset everywhere."""
    f_low = _call(B0=0.064)
    f_high = _call(B0=0.128)
    ratio = (f_low.abs().mean() / f_high.abs().mean().clamp_min(1e-12)).item()
    assert ratio == pytest.approx(2.0, rel=1e-4)


def test_ulf_offset_dwarfs_high_field() -> None:
    """At ULF (0.064 T) the offset is ~50× larger than at 3 T."""
    f_ulf = _call(B0=0.064)
    f_3t = _call(B0=3.0)
    ratio = (f_ulf.abs().max() / f_3t.abs().max().clamp_min(1e-12)).item()
    assert 40.0 < ratio < 60.0, (
        f"Expected ~50x ratio between 0.064T and 3T offsets, got {ratio:.1f}"
    )


def test_zero_gradients_zero_field() -> None:
    """No imaging gradient ⇒ no concomitant offset anywhere."""
    f = _call(B0=0.064, gradients=(0.0, 0.0, 0.0))
    assert torch.all(f == 0)


def test_output_shape_and_dtype() -> None:
    f = maxwell_concomitant_field_hz(
        spatial_shape=(4, 8, 8),
        voxel_size_mm=(1.0, 1.0, 1.0),
        gradient_amplitudes_mT_per_m=(10.0, 10.0, 10.0),
        field_strength_t=0.3,
        dtype=torch.float64,
    )
    assert f.shape == (4, 8, 8)
    assert f.dtype == torch.float64


def test_rejects_non_positive_field_strength() -> None:
    with pytest.raises(ValueError, match="field_strength_t"):
        maxwell_concomitant_field_hz(
            spatial_shape=(4, 8, 8),
            voxel_size_mm=(1.0, 1.0, 1.0),
            gradient_amplitudes_mT_per_m=(10.0, 10.0, 10.0),
            field_strength_t=0.0,
        )
