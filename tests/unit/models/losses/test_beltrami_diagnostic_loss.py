"""Unit tests for BeltramiDiagnosticLoss.

Pin the fundamental quasiconformal-geometry properties:

- Identity displacement ⇒ |mu| = 0 (no distortion).
- Pure shear ⇒ |mu| > 0 (smooth distortion).
- Loss is autograd-traceable end-to-end.
- Output shape is correct for both 2D (B,2,H,W) and 3D (B,2,D,H,W).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.losses.beltrami_diagnostic_loss import (  # noqa: E402
    BeltramiDiagnosticLoss,
)


def test_identity_field_has_zero_beltrami() -> None:
    """f(z) = z everywhere ⇒ ∂f/∂z̄ = 0 ⇒ μ ≡ 0."""
    H = W = 16
    # Identity displacement: u(x,y) = x, v(x,y) = y. The constant
    # field (u=0, v=0) is also conformal (just translation), so we
    # use that as the simpler analytic case.
    field = torch.zeros(1, 2, H, W)
    loss = BeltramiDiagnosticLoss(reduction="mean")(field)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_shear_field_has_nonzero_beltrami() -> None:
    """A pure shear deformation has positive |μ|."""
    H = W = 16
    # u(x,y) = a * y (shear in x). Then u_x = 0, u_y = a.
    # v(x,y) = 0 ⇒ v_x = v_y = 0.
    # μ = (u_x - v_y + i(v_x + u_y)) / (u_x + v_y + i(v_x - u_y))
    #   = (0 + i a) / (0 + i (-0))  → undefined denominator.
    # Add a tiny radial component v(x,y)=eps*x to break degeneracy.
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    a = 0.5
    eps = 0.1
    u = a * yy
    v = eps * xx
    field = torch.stack([u, v], dim=0).unsqueeze(0)  # (1, 2, H, W)
    loss = BeltramiDiagnosticLoss(reduction="mean")(field)
    assert loss.item() > 0.0


def test_loss_is_autograd_traceable() -> None:
    """Backprop through Beltrami must produce non-trivial gradients."""
    field = torch.randn(1, 2, 16, 16, requires_grad=True)
    loss = BeltramiDiagnosticLoss(reduction="mean")(field)
    loss.backward()
    assert field.grad is not None
    assert torch.any(field.grad != 0)


def test_3d_input_per_slice_average() -> None:
    """Per-slice 2D Beltrami over a 3D volume must produce a scalar."""
    field = torch.randn(2, 2, 4, 16, 16)  # (B, 2, D, H, W)
    loss = BeltramiDiagnosticLoss(reduction="mean")(field)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_pred_target_fallback_signature() -> None:
    """forward(pred, target) treats residual as a (u, v) field."""
    pred = torch.randn(1, 2, 16, 16)
    target = torch.randn(1, 2, 16, 16)
    loss = BeltramiDiagnosticLoss(reduction="mean")(pred, target)
    assert torch.isfinite(loss)


def test_pred_target_fallback_rejects_non_two_channel() -> None:
    """Fallback signature requires exactly 2 channels."""
    pred = torch.randn(1, 3, 16, 16)
    target = torch.randn(1, 3, 16, 16)
    with pytest.raises(ValueError, match="2-channel"):
        BeltramiDiagnosticLoss(reduction="mean")(pred, target)


def test_penalty_reduction_zero_for_quasiconformal() -> None:
    """`relu(|mu| - 1)^2` is exactly 0 for fields with |mu| <= 1."""
    # Zero field has mu = 0, so penalty must be 0.
    field = torch.zeros(1, 2, 16, 16)
    loss = BeltramiDiagnosticLoss(reduction="penalty")(field)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
