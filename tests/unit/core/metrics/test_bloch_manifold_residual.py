"""Unit tests for the ``bloch_manifold_residual`` metric (Idea 2 / B-2)."""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.bloch_manifold_residual import BlochManifoldResidual
from mriforge.core.metrics.registry import MetricsRegistry
from mriforge.models.physics.bloch_manifold_projector import BlochManifoldProjector


def _on_manifold(proj: BlochManifoldProjector, b: int = 1, hw: int = 4) -> torch.Tensor:
    k = proj.flip_angles_rad.numel()
    n = b * hw * hw
    bank = proj._bloch_signal_bank(
        torch.full((n, 1), 1.2), torch.full((n, 1), 900.0), torch.full((n, 1), 70.0)
    )
    return bank.reshape(b, hw, hw, k).permute(0, 3, 1, 2).contiguous()


def test_metric_registered_and_alias():
    assert MetricsRegistry.is_registered("bloch_manifold_residual")
    assert MetricsRegistry.is_registered("manifold_residual_l2")


def test_metric_contract_properties():
    m = BlochManifoldResidual()
    assert m.name == "bloch_manifold_residual"
    assert m.higher_is_better is False


def test_returns_float_and_reference_free():
    m = BlochManifoldResidual(n_inner_iterations=8)
    k = m._get_projector().flip_angles_rad.numel()
    pred = torch.rand(1, k, 6, 6) * 0.1
    # target is ignored — passing None and a tensor must agree.
    v_none = m(pred)
    v_tgt = m(pred, torch.rand(1, k, 6, 6))
    assert isinstance(v_none, float)
    assert v_none == pytest.approx(v_tgt, abs=1e-6)


def test_on_manifold_lower_than_off_manifold():
    m = BlochManifoldResidual(n_inner_iterations=20, normalize=False)
    proj = m._get_projector()
    on = m(_on_manifold(proj))
    k = proj.flip_angles_rad.numel()
    torch.manual_seed(0)
    off = m(torch.rand(1, k, 4, 4) * 0.05 + 0.01)
    assert on < off
    assert on < 1e-2


def test_normalize_changes_scale():
    k = BlochManifoldProjector().flip_angles_rad.numel()
    torch.manual_seed(1)
    pred = torch.rand(1, k, 6, 6) * 0.1
    m_norm = BlochManifoldResidual(n_inner_iterations=6, normalize=True)
    m_raw = BlochManifoldResidual(n_inner_iterations=6, normalize=False)
    assert m_norm(pred) != m_raw(pred)
    assert m_norm(pred) >= 0.0


def test_complex_input_supported():
    m = BlochManifoldResidual(n_inner_iterations=6)
    k = m._get_projector().flip_angles_rad.numel()
    pred = torch.complex(torch.rand(2, k, 8, 8) * 0.05, torch.rand(2, k, 8, 8) * 0.05)
    v = m(pred)
    assert isinstance(v, float) and v == v  # finite


def test_wrong_ndim_raises():
    m = BlochManifoldResidual(n_inner_iterations=2)
    with pytest.raises(ValueError):
        m(torch.rand(4, 4))
