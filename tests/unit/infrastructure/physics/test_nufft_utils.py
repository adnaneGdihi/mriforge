"""Unit tests for nufft_utils.apply_nufft_adjoint (C1, audit 2026-05-28).

``apply_nufft_adjoint`` was previously imported by no test. It is the
K-space -> image adjoint NUFFT used on the non-Cartesian path, and it owns a
fair amount of dimension-wrangling (squeezing collation batch dims, deciding
whether the trajectory needs transposing, broadcasting the density-compensation
function). Those reshaping branches are exactly where silent shape bugs hide,
so the tests below exercise each of them and assert the documented output
contract ``(B, C, H, W)``.

torchkbnufft is an optional dependency; skip cleanly when it is absent.
"""

import pytest
import torch

pytest.importorskip("torchkbnufft")

from mriforge.infrastructure.physics.nufft_utils import apply_nufft_adjoint


def _radial_trajectory(num_spokes: int, samples: int) -> torch.Tensor:
    """A small radial trajectory in (N, 2) layout, scaled to [-pi, pi]."""
    angles = torch.linspace(0, torch.pi, num_spokes + 1)[:-1]
    radius = torch.linspace(-0.5, 0.5, samples)
    pts = []
    for a in angles:
        kx = radius * torch.cos(a)
        ky = radius * torch.sin(a)
        pts.append(torch.stack([kx, ky], dim=-1))
    traj = torch.cat(pts, dim=0)  # (num_spokes*samples, 2)
    return traj * 2 * torch.pi


@pytest.fixture
def radial_data():
    im_size = (24, 24)
    n = 16 * 24  # spokes * samples
    traj = _radial_trajectory(16, 24)  # (N, 2)
    return im_size, n, traj


def test_adjoint_returns_image_shape_bchw(radial_data):
    im_size, n, traj = radial_data
    b, c = 2, 1
    kspace = torch.randn(b, c, n, dtype=torch.complex64)
    ktraj = traj.unsqueeze(0).repeat(b, 1, 1)  # (B, N, 2)
    img = apply_nufft_adjoint(kspace, ktraj, im_size=im_size)
    assert img.shape == (b, c, *im_size)
    assert torch.isfinite(img).all()


def test_collation_batch_dim_is_squeezed(radial_data):
    """kspace (B, 1, ...) extra-dim path and trajectory (B, 1, N, 2) path."""
    im_size, n, traj = radial_data
    b = 1
    # kspace shaped (B, 1, N) -> the function squeezes dim 1.
    kspace = torch.randn(b, 1, n, dtype=torch.complex64)
    ktraj = traj.unsqueeze(0).unsqueeze(0)  # (B, 1, N, 2) -> squeezed to (B, N, 2)
    img = apply_nufft_adjoint(kspace, ktraj, im_size=im_size)
    assert img.shape[0] == b
    assert img.shape[-2:] == im_size


def test_density_compensation_changes_result(radial_data):
    """Passing a non-uniform DCF must re-weight k-space (output differs)."""
    im_size, n, traj = radial_data
    b, c = 1, 1
    kspace = torch.randn(b, c, n, dtype=torch.complex64)
    ktraj = traj.unsqueeze(0).repeat(b, 1, 1)

    img_no_dcf = apply_nufft_adjoint(kspace, ktraj, im_size=im_size)
    dcf = torch.linspace(0.1, 1.0, n).unsqueeze(0)  # (B, N), non-uniform
    img_dcf = apply_nufft_adjoint(kspace, ktraj, dcf=dcf, im_size=im_size)

    assert img_dcf.shape == img_no_dcf.shape
    assert not torch.allclose(img_dcf, img_no_dcf)


def test_trajectory_transpose_branch(radial_data):
    """Trajectory already in (B, 2, N) layout must be accepted as-is.

    The (N, 2) layout is transposed internally; the (2, N) layout must pass
    through unchanged and yield the same image (within fp tolerance).
    """
    im_size, n, traj = radial_data
    b, c = 1, 1
    kspace = torch.randn(b, c, n, dtype=torch.complex64)

    traj_n2 = traj.unsqueeze(0).repeat(b, 1, 1)  # (B, N, 2) -> transposed inside
    traj_2n = traj.t().unsqueeze(0).repeat(b, 1, 1)  # (B, 2, N) -> used as-is

    img_a = apply_nufft_adjoint(kspace, traj_n2, im_size=im_size)
    img_b = apply_nufft_adjoint(kspace, traj_2n, im_size=im_size)
    assert torch.allclose(img_a, img_b, atol=1e-4)
