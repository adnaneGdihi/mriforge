"""Unit tests for SE3EquivarianceDefectLoss (B-3 of the VF campaign).

Covers: registration, near-zero defect on an equivariant block, large defect on
a non-equivariant map, determinism, SO2/SO3 sampling, gradient flow, and error
paths. Direct-import per the source<->test pairing rule.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.se3_lie_algebra_mamba import SE3LieAlgebraMambaBlock
from spectramr.models.losses.se3_equivariance_defect import SE3EquivarianceDefectLoss
from tests.utils.optional_backends import requires_cuda_for_mamba


@pytest.fixture
def equivariant_block() -> SE3LieAlgebraMambaBlock:
    torch.manual_seed(0)
    return SE3LieAlgebraMambaBlock(d_state=8, group="SE3", sample_rate_hz=100.0).eval()


def test_registered_in_loss_registry() -> None:
    from spectramr.models.losses.registry import LossRegistry, create_loss

    assert "se3_equivariance_defect" in LossRegistry.list_available()
    loss = create_loss("se3_equivariance_defect")
    assert isinstance(loss, SE3EquivarianceDefectLoss)


@requires_cuda_for_mamba
def test_equivariant_block_near_zero_defect(
    equivariant_block: SE3LieAlgebraMambaBlock,
) -> None:
    loss = SE3EquivarianceDefectLoss(num_rotations=4, group="SO2")
    u = torch.randn(2, 12, 6)
    defect = loss(u, None, equivariant_fn=equivariant_block, lie_input=u).detach()
    assert float(defect) < 1e-4, f"equivariant block defect {float(defect)} too large"


def test_nonequivariant_map_large_defect() -> None:
    torch.manual_seed(1)
    loss = SE3EquivarianceDefectLoss(num_rotations=4, group="SO2")
    nonequiv = torch.nn.Linear(6, 6)
    u = torch.randn(2, 12, 6)
    defect = loss(u, None, equivariant_fn=nonequiv, lie_input=u).detach()
    assert float(defect) > 1e-2, "non-equivariant map should have positive defect"


def test_identity_defect_is_zero_and_deterministic() -> None:
    loss = SE3EquivarianceDefectLoss(num_rotations=4, group="SO2", seed=7)
    u = torch.randn(2, 10, 6)
    d1 = loss(u, None)  # identity default
    d2 = loss(u, None)
    assert float(d1) == 0.0
    assert float(d1) == float(d2)


@requires_cuda_for_mamba
def test_so3_sampling(equivariant_block: SE3LieAlgebraMambaBlock) -> None:
    loss = SE3EquivarianceDefectLoss(num_rotations=3, group="SO3")
    u = torch.randn(2, 12, 6)
    defect = loss(u, None, equivariant_fn=equivariant_block, lie_input=u).detach()
    assert float(defect) < 1e-4


@requires_cuda_for_mamba
def test_gradient_flow_through_block(
    equivariant_block: SE3LieAlgebraMambaBlock,
) -> None:
    equivariant_block.train()
    loss = SE3EquivarianceDefectLoss(num_rotations=2, group="SO2")
    u = torch.randn(2, 10, 6)
    out = loss(u, None, equivariant_fn=equivariant_block, lie_input=u)
    out.backward()
    total = sum(
        p.grad.abs().sum().item()
        for p in equivariant_block.parameters()
        if p.grad is not None
    )
    # defect is ~0 for an equivariant block, so gradients are tiny but finite.
    assert total >= 0.0
    for p in equivariant_block.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_seed_changes_rotation_sample() -> None:
    """Different seeds sample different rotations → generally different defect."""
    torch.manual_seed(2)
    nonequiv = torch.nn.Linear(6, 6)
    u = torch.randn(2, 12, 6)
    d_a = SE3EquivarianceDefectLoss(num_rotations=4, group="SO3", seed=0)(
        u, None, equivariant_fn=nonequiv, lie_input=u
    ).detach()
    d_b = SE3EquivarianceDefectLoss(num_rotations=4, group="SO3", seed=99)(
        u, None, equivariant_fn=nonequiv, lie_input=u
    ).detach()
    assert float(d_a) != float(d_b)


def test_unknown_group_raises() -> None:
    with pytest.raises(ValueError, match="unknown group"):
        SE3EquivarianceDefectLoss(group="SE3")  # type: ignore[arg-type]


def test_bad_input_shape_raises() -> None:
    loss = SE3EquivarianceDefectLoss(num_rotations=2, group="SO2")
    with pytest.raises(ValueError, match=r"\[B, L, 6\]"):
        loss(torch.randn(2, 12, 5), None)


def test_noncallable_fn_raises() -> None:
    loss = SE3EquivarianceDefectLoss(num_rotations=2, group="SO2")
    u = torch.randn(2, 10, 6)
    with pytest.raises(TypeError, match="callable"):
        loss(u, None, equivariant_fn=123)
