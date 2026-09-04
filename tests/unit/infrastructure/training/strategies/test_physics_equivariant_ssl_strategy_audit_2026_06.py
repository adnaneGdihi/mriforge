"""Regression tests for the 2026-06 audit fix to ``physics_equivariant_ssl``.

The k-space operators (``fft_shift`` / ``kspace_phase_shift``) previously used
image-domain ``torch.roll`` / ``F.grid_sample`` despite advertising Fourier
behaviour; they now round-trip through the physics SSOT (``fft_ops``). Unknown
operators must raise (pitfall #9).
"""
from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies import (
    physics_equivariant_ssl_strategy as mod,
)


def test_default_operator_pool_only_advertises_implemented_ops() -> None:
    pool = mod._make_operator_pool(torch.device("cpu"))
    assert pool == ["fft_shift", "kspace_phase_shift", "rigid_rotation_small"]
    # Every advertised operator is in the dispatchable set.
    assert set(pool) <= mod._SUPPORTED_OPERATORS


@pytest.mark.parametrize("op", ["fft_shift", "kspace_phase_shift"])
def test_kspace_operators_return_real_same_shape_and_change_input(op: str) -> None:
    """The k-space ops round-trip through fft_ops and return a real, finite
    tensor of the same shape (not the identity)."""
    torch.manual_seed(0)
    x = torch.randn(2, 1, 8, 8)
    out = mod._apply_operator(x, op)
    assert out.shape == x.shape
    assert not torch.is_complex(out)
    assert torch.isfinite(out).all()
    # The operator must actually transform the input.
    assert not torch.allclose(out, x)


def test_unknown_operator_raises() -> None:
    with pytest.raises(ValueError, match="Unknown operator"):
        mod._apply_operator(torch.randn(1, 1, 4, 4), "nufft_warp")


def test_init_rejects_unknown_physics_eq_ssl_knob() -> None:
    """A typo'd knob must be rejected (extra='forbid'-equivalent), not silently
    dropped by ``pe.get(..., default)`` — pitfall #15. The guard is in __init__
    (which needs a built env), so assert the construct is present in source."""
    import inspect

    src = inspect.getsource(mod.PhysicsEquivariantSSLStrategy.__init__)
    assert "Unknown physics_eq_ssl knob" in src
    assert "set(pe) - _known" in src
