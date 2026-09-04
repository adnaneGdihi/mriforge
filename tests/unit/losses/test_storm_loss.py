"""Tests for the SToRM manifold-smoothness regulariser.

The class lives in ``spectramr.models.losses.storm_loss`` after the Phase 5
canonical-home migration (CLAUDE.md pitfall #12: every
``@register_loss`` decorator must live under ``models/losses/``). Prior
to the migration, ``SToRMRegularizer`` lived in
``spectramr.models.geometric.storm`` and was decorated there, in
violation of the rule. These tests pin both the new canonical location
and the registry entry so a regression on either is caught at CI.
"""
from __future__ import annotations

import pytest
import torch


def test_storm_registered_under_canonical_name() -> None:
    """The ``storm`` key must resolve via the loss registry."""
    import spectramr.models.losses  # trigger registration  # noqa: F401
    from spectramr.models.losses.registry import list_available

    assert "storm" in list_available()


def test_storm_aliases_resolve() -> None:
    """Both advertised aliases must resolve to the same class."""
    import spectramr.models.losses  # noqa: F401
    from spectramr.models.losses.registry import create_loss

    a = create_loss("storm", learn_laplacian=False, num_frames=4, nav_dim=8)
    b = create_loss(
        "manifold_smoothness", learn_laplacian=False, num_frames=4, nav_dim=8
    )
    c = create_loss(
        "storm_regularization", learn_laplacian=False, num_frames=4, nav_dim=8
    )
    assert type(a) is type(b) is type(c)


def test_storm_canonical_home_is_models_losses() -> None:
    """The class must be importable from the canonical losses path.

    Importing from ``models.geometric.storm`` must NOT yield a
    re-decorated copy (else there would be two registrations under the
    same key). The legacy module no longer defines the class.
    """
    from spectramr.models.losses.storm_loss import SToRMRegularizer as Canonical
    from spectramr.models.geometric import storm as legacy_module

    assert not hasattr(legacy_module, "SToRMRegularizer"), (
        "models.geometric.storm must not define SToRMRegularizer after "
        "the Phase 5 migration -- the class lives in "
        "models.losses.storm_loss."
    )
    assert Canonical.__module__ == "spectramr.models.losses.storm_loss"


def test_storm_forward_shape_and_grad() -> None:
    """The penalty must be a scalar and propagate gradients."""
    from spectramr.models.losses.storm_loss import SToRMRegularizer

    reg = SToRMRegularizer(
        lambda_manifold=0.1, learn_laplacian=False, num_frames=4, nav_dim=8
    )
    x = torch.randn(2, 4, 1, 8, 8, requires_grad=True)
    loss = reg(x)
    assert loss.shape == torch.Size([]), "manifold-smoothness loss must be scalar"
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_storm_lambda_zero_produces_zero_loss() -> None:
    """``lambda_manifold=0`` is the no-op identity for the regulariser."""
    from spectramr.models.losses.storm_loss import SToRMRegularizer

    reg = SToRMRegularizer(
        lambda_manifold=0.0, learn_laplacian=False, num_frames=4, nav_dim=8
    )
    x = torch.randn(1, 4, 1, 4, 4)
    assert pytest.approx(0.0, abs=1e-6) == float(reg(x))
