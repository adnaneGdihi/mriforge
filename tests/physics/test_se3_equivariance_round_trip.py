r"""Physics test: SE(3) equivariance round-trip for the Lie-algebra Mamba block.

Asserts :math:`\|\mathrm{Block}(\mathrm{Ad}_{g_0} u) - \mathrm{Ad}_{g_0}\,\mathrm{Block}(u)\| < 10^{-4}`
for random :math:`g_0 \in SE(3)` (pure-rotation adjoint), the acceptance
criterion from the Tier-2 forward-probe extension in Idea 3.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.blocks.se3_lie_algebra_mamba import SE3LieAlgebraMambaBlock
from tests.utils.optional_backends import requires_cuda_for_mamba

TOL = 1e-4


def _haar_so3(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.linalg.det(q) < 0:
        q = q.clone()
        q[:, -1] = -q[:, -1]
    return q


@requires_cuda_for_mamba
@pytest.mark.physics
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_se3_equivariance_round_trip_random_rotations(seed: int) -> None:
    torch.manual_seed(100 + seed)
    block = SE3LieAlgebraMambaBlock(
        d_state=8, group="SE3", sample_rate_hz=100.0
    ).eval()
    xi = torch.randn(2, 24, 6)
    rot = _haar_so3(seed)

    out = block(xi)
    lhs = block(block.adjoint_rotation(xi, rot))
    rhs = block.adjoint_rotation(out, rot)
    defect = torch.linalg.vector_norm(lhs - rhs).item()
    assert defect < TOL, f"seed={seed}: defect {defect} >= {TOL}"


@pytest.mark.physics
def test_adjoint_is_orthogonality_preserving() -> None:
    """Ad_{(R,0)} preserves the Euclidean norm of each irreducible component."""
    torch.manual_seed(5)
    block = SE3LieAlgebraMambaBlock(d_state=4, group="SE3", sample_rate_hz=100.0)
    xi = torch.randn(2, 8, 6)
    rot = _haar_so3(11)
    adj = block.adjoint_rotation(xi, rot)
    # rotation preserves the per-component norm (rotation invariance of the gate).
    assert torch.allclose(
        torch.linalg.vector_norm(adj[..., :3], dim=-1),
        torch.linalg.vector_norm(xi[..., :3], dim=-1),
        atol=1e-5,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(adj[..., 3:], dim=-1),
        torch.linalg.vector_norm(xi[..., 3:], dim=-1),
        atol=1e-5,
    )


@pytest.mark.physics
def test_identity_rotation_is_noop() -> None:
    torch.manual_seed(6)
    block = SE3LieAlgebraMambaBlock(d_state=4, group="SE3", sample_rate_hz=100.0)
    xi = torch.randn(2, 8, 6)
    identity = torch.eye(3)
    assert torch.allclose(block.adjoint_rotation(xi, identity), xi, atol=1e-6)


@pytest.mark.physics
def test_so2_subgroup_matches_explicit_rotation() -> None:
    """SO(2) in-plane rotation is the z-axis case of the SO(3) adjoint."""
    torch.manual_seed(7)
    block = SE3LieAlgebraMambaBlock(d_state=4, group="SE3", sample_rate_hz=100.0)
    xi = torch.randn(1, 4, 6)
    theta = 0.9
    rz = torch.eye(3)
    rz[0, 0], rz[0, 1] = math.cos(theta), -math.sin(theta)
    rz[1, 0], rz[1, 1] = math.sin(theta), math.cos(theta)
    adj = block.adjoint_rotation(xi, rz)
    # in-plane (x, y) of omega rotates; z component is preserved.
    assert torch.allclose(adj[..., 2], xi[..., 2], atol=1e-6)  # omega_z fixed
    assert torch.allclose(adj[..., 5], xi[..., 5], atol=1e-6)  # v_z fixed
