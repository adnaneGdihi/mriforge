r"""Tests for ``infrastructure/physics/group_actions.py`` (Phase A step 3 / P0.2).

Group actions :math:`T_g` for Equivariant Imaging (Chen et al., ICCV 2021).
The strategy enforces :math:`f(A T_g f(A^* y)) = T_g f(A^* y)`; these actions
supply :math:`T_g`. Two regimes are covered:

* ``DihedralGroup`` — exact (``rot90``/``flip``), so the round-trip
  :math:`T_g^{-1} T_g = I` is bit-exact even on complex tensors.
* ``SmallRotationGroup`` — continuous small rotation via ``grid_sample`` (the
  brain-appropriate group: the brain is only approximately rotation-invariant),
  so the round-trip is approximate.

``sensing_margin`` is the identifiability gate (Tachella sensing theorems): EI
is identifiable iff the forward operator is NOT equivariant to the group, i.e.
the group rotates signal *out* of the measured subspace ``range(A^* A)``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c  # noqa: E402
from mriforge.infrastructure.physics.group_actions import (  # noqa: E402
    GROUP_REGISTRY,
    DihedralGroup,
    SmallRotationGroup,
    get_group_action,
    sensing_margin,
)


def _complex_image(b: int = 2, c: int = 1, h: int = 16, w: int = 16):
    torch.manual_seed(0)
    return torch.randn(b, c, h, w) + 1j * torch.randn(b, c, h, w)


def _cartesian_mask(h: int, w: int, *, full: bool, keep: int = 4):
    """1-D Cartesian phase-encode mask broadcast to (1, 1, H, W)."""
    mask = torch.zeros(1, 1, h, w)
    if full:
        mask[...] = 1.0
        return mask
    # keep the central ``keep`` columns + every 4th column (R~4 undersampling)
    mask[..., :, ::4] = 1.0
    c0 = w // 2 - keep // 2
    mask[..., :, c0 : c0 + keep] = 1.0
    return mask


def _masked_ops(mask):
    mask_c = mask.to(torch.complex64)

    def forward_op(x):
        return mask_c * fft2c(x)

    def adjoint_op(k):
        return ifft2c(mask_c * k)

    return forward_op, adjoint_op


# ---------------------------------------------------------------------------
# DihedralGroup — exact, complex-safe
# ---------------------------------------------------------------------------
def test_dihedral_inverse_roundtrip_is_exact_on_complex():
    g = DihedralGroup()
    x = _complex_image()
    for elem in range(g.num_elements):
        restored = g.inverse(g.apply(x, elem), elem)
        assert torch.equal(restored, x), f"element {elem} not exactly invertible"


def test_dihedral_has_eight_elements_and_samples_nonidentity():
    g = DihedralGroup()
    assert g.num_elements == 8
    gen = torch.Generator().manual_seed(1)
    draws = {g.sample(generator=gen) for _ in range(50)}
    assert draws, "sample produced nothing"
    assert all(1 <= d < 8 for d in draws), f"sampled identity or out-of-range: {draws}"


def test_dihedral_nonidentity_actually_transforms_input():
    """A non-trivial element must change the tensor (the action is real)."""
    g = DihedralGroup()
    x = _complex_image()
    # element 2 == one 90-degree rotation; on a random image it must differ.
    assert not torch.equal(g.apply(x, 2), x)


# ---------------------------------------------------------------------------
# SmallRotationGroup — continuous, complex-aware, approximate round-trip
# ---------------------------------------------------------------------------
def test_small_rotation_roundtrip_approximate_on_complex():
    g = SmallRotationGroup(max_angle_deg=8.0, num_angles=4)
    # smooth, centred blob so grid_sample interpolation is well-behaved
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij"
    )
    blob = torch.exp(-(xx**2 + yy**2) / 0.3).view(1, 1, 32, 32)
    x = (blob + 1j * 0.5 * blob).to(torch.complex64)
    for elem in range(1, g.num_elements):
        restored = g.inverse(g.apply(x, elem), elem)
        # central crop avoids the FOV edges that grid_sample zero-pads
        err = (restored - x)[..., 8:24, 8:24].abs().mean()
        assert err < 5e-2, f"element {elem} round-trip error {err:.4f} too large"


def test_small_rotation_preserves_complex_dtype():
    g = SmallRotationGroup(max_angle_deg=5.0, num_angles=2)
    xc = _complex_image(h=24, w=24)
    out = g.apply(xc, 1)
    assert out.is_complex() and out.shape == xc.shape
    xr = xc.real.contiguous()
    outr = g.apply(xr, 1)
    assert not outr.is_complex() and outr.shape == xr.shape


# ---------------------------------------------------------------------------
# Registry — no silent fallback (pitfall #9)
# ---------------------------------------------------------------------------
def test_get_group_action_resolves_registered_name():
    assert isinstance(get_group_action("dihedral"), DihedralGroup)
    assert isinstance(
        get_group_action("small_rotation", max_angle_deg=6.0), SmallRotationGroup
    )
    assert "dihedral" in GROUP_REGISTRY and "small_rotation" in GROUP_REGISTRY


def test_get_group_action_unknown_raises():
    with pytest.raises(ValueError, match="Unknown group action"):
        get_group_action("so2_full")


# ---------------------------------------------------------------------------
# sensing_margin — the EI identifiability gate
# ---------------------------------------------------------------------------
def test_sensing_margin_positive_for_partial_cartesian_mask():
    """A 1-D undersampling mask is NOT equivariant to 90-degree rotation =>
    the group rotates signal out of the measured subspace => margin > 0."""
    mask = _cartesian_mask(16, 16, full=False)
    forward_op, adjoint_op = _masked_ops(mask)
    g = DihedralGroup()
    x = _complex_image(b=1, c=1, h=16, w=16)
    margin = sensing_margin(forward_op, adjoint_op, g, x, n_samples=4)
    assert margin > 0.1, f"expected identifiable (margin>0.1), got {margin:.4f}"


def test_sensing_margin_zero_for_full_sampling():
    """Full sampling => A^* A = I => nothing leaves the range => margin ~ 0
    => EI adds no signal (the gate rejects such an arm)."""
    mask = _cartesian_mask(16, 16, full=True)
    forward_op, adjoint_op = _masked_ops(mask)
    g = DihedralGroup()
    x = _complex_image(b=1, c=1, h=16, w=16)
    margin = sensing_margin(forward_op, adjoint_op, g, x, n_samples=4)
    assert margin < 1e-4, f"expected unidentifiable (margin~0), got {margin:.4f}"
