"""Regression test for the autograd bug in PersistentHomologyLoss.

Before the GeoMamba-ULF Phase-1 fix the loss accumulated via
``loss += (...).item()`` which silently detached from the autograd
graph. The loss was therefore a no-op for backprop — gradients
were always zero — even though training metrics looked plausible.

This test pins the fix: gradients with respect to ``pred`` must be
non-zero when ``pred != target``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.losses.topological_loss import PersistentHomologyLoss  # noqa: E402


def test_persistent_homology_loss_is_autograd_traceable() -> None:
    """Gradients on `pred` must flow end-to-end (regression test)."""
    pred = torch.rand(1, 1, 32, 32, requires_grad=True)
    target = torch.rand(1, 1, 32, 32)
    loss = PersistentHomologyLoss(levels=4)(pred, target)

    assert isinstance(loss, torch.Tensor), "loss must be a tensor"
    assert loss.requires_grad, "loss must require grad after the fix"

    loss.backward()

    assert pred.grad is not None, "pred.grad must not be None after backward"
    assert torch.any(pred.grad != 0), (
        "pred.grad is identically zero — the .item() autograd-detach bug is back"
    )


def test_persistent_homology_loss_zero_at_identity() -> None:
    """Loss should be ~0 when pred == target (sanity check)."""
    target = torch.rand(1, 1, 16, 16)
    pred = target.clone().detach().requires_grad_(True)
    loss = PersistentHomologyLoss(levels=4)(pred, target)
    # The soft Euler-characteristic approximation is symmetric around
    # identity, so loss should be exactly zero.
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
