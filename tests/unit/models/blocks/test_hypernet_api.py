"""Tests for the generic hypernetwork API (PR-23)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mriforge.models.blocks.hypernet_api import (
    Hypernet,
    HypernetWrappedModule,
    IdentityHypernet,
    get_hypernet,
    register_hypernet,
)


def test_identity_hypernet_emits_zeros() -> None:
    h = IdentityHypernet(context_dim=4, param_shapes={"gamma": (8,)})
    ctx = torch.randn(2, 4)
    out = h(ctx)
    assert "gamma" in out
    assert out["gamma"].shape == (2, 8)
    assert torch.allclose(out["gamma"], torch.zeros_like(out["gamma"]))


def test_hypernet_emits_correct_shapes() -> None:
    h = Hypernet(
        context_dim=3,
        param_shapes={"gamma": (8,), "beta": (8,)},
        embedding_dim=16,
    )
    ctx = torch.randn(4, 3)
    out = h(ctx)
    assert out["gamma"].shape == (4, 8)
    assert out["beta"].shape == (4, 8)


class _ScalingPrimary(nn.Module):
    """Tiny primary whose forward multiplies ``x`` by ``gain`` per-sample.

    ``gain`` has shape (batch, 1) so it broadcasts against (batch, 4).
    """

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.full((2, 1), 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


def test_hypernet_wrapped_module_replaces_attribute() -> None:
    primary = _ScalingPrimary()
    h = IdentityHypernet(context_dim=2, param_shapes={"gain": (1,)})
    wrapped = HypernetWrappedModule(primary, h, replace_keys=["gain"])
    x = torch.randn(2, 4)
    ctx = torch.randn(2, 2)
    out = wrapped(x, hypernet_context=ctx)
    assert torch.allclose(out, torch.zeros_like(out))
    assert torch.allclose(primary.gain, torch.full((2, 1), 2.0))


def test_hypernet_wrapped_module_passthrough_without_context() -> None:
    primary = _ScalingPrimary()
    h = IdentityHypernet(context_dim=2, param_shapes={"gain": (1,)})
    wrapped = HypernetWrappedModule(primary, h, replace_keys=["gain"])
    x = torch.randn(2, 4)
    out_a = wrapped(x)
    out_b = primary(x)
    assert torch.allclose(out_a, out_b)


def test_register_hypernet_round_trip() -> None:
    @register_hypernet("custom_test_hypernet")
    class _Custom(IdentityHypernet):
        pass

    h = get_hypernet("custom_test_hypernet", context_dim=2, param_shapes={"x": (3,)})
    assert isinstance(h, _Custom)


def test_register_hypernet_rejects_duplicate() -> None:
    @register_hypernet("dup_test_hypernet")
    class _A(IdentityHypernet):
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_hypernet("dup_test_hypernet")
        class _B(IdentityHypernet):
            pass


def test_replace_keys_validation() -> None:
    primary = nn.Linear(4, 4)
    h = IdentityHypernet(context_dim=2, param_shapes={"weight": (4, 4)})
    with pytest.raises(ValueError, match="not in hypernet.param_shapes"):
        HypernetWrappedModule(primary, h, replace_keys=["nonexistent"])


# ---------------------------------------------------------------------------
# Hypernet with 2-D param shapes (covers _prod multi-dim path)
# ---------------------------------------------------------------------------


def test_hypernet_2d_param_shape() -> None:
    """``param_shapes`` with 2-D shapes produce (B, *shape) output."""
    h = Hypernet(
        context_dim=4,
        param_shapes={"weight": (4, 4)},
        embedding_dim=16,
    )
    ctx = torch.randn(2, 4)
    out = h(ctx)
    assert out["weight"].shape == (2, 4, 4)
    assert torch.isfinite(out["weight"]).all()


def test_hypernet_canary_forward_is_finite() -> None:
    """Canary: all outputs are finite for random context."""
    h = Hypernet(
        context_dim=3,
        param_shapes={"a": (8,), "b": (4, 2)},
        embedding_dim=32,
    )
    h.eval()
    ctx = torch.randn(3, 3)
    out = h(ctx)
    for key, val in out.items():
        assert torch.isfinite(val).all(), f"{key} has non-finite values"


def test_hypernet_unknown_name_raises() -> None:
    """``get_hypernet`` with an unknown name raises ``KeyError``."""
    with pytest.raises(KeyError, match="Unknown hypernet"):
        from mriforge.models.blocks.hypernet_api import get_hypernet
        get_hypernet("definitely_not_registered", context_dim=2, param_shapes={"x": (1,)})
