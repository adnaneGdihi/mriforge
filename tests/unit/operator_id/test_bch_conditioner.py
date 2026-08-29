"""Unit tests for the BCH operator conditioner (Proposal 1).

Checks the conditioner output shapes and ranges: non-negative severities,
strictly-positive radial PSD, complex gains, and a low-rank factor ``U`` of
the advertised shape.
"""

# ruff: noqa: E402  (pytest.importorskip must precede torch-dependent imports)

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.operator_id.bch_conditioner import BCHOperatorConditioner
from mriforge.models.registry import MODEL_REGISTRY


def _make(num_modes=4, rank=8, bins=16, c=1, h=8, w=8, contrasts=3):
    return BCHOperatorConditioner(
        num_modes=num_modes,
        num_contrasts=contrasts,
        covariance_rank=rank,
        rho_bins=bins,
        channels=c,
        height=h,
        width=w,
    )


def test_registered_in_model_registry():
    assert "bch_operator_conditioner" in MODEL_REGISTRY
    assert MODEL_REGISTRY["bch_operator_conditioner"]["mode"] == "operator_id"


def test_forward_output_shapes_and_ranges():
    model = _make()
    b = 5
    contrast_idx = torch.randint(0, 3, (b,))
    field_coord = torch.full((b,), -0.522)  # log10(0.3)
    out = model(contrast_idx, field_coord)

    assert out["severities"].shape == (b, 4)
    assert out["rho_radial"].shape == (b, 16)
    assert out["gains"].shape == (b, 8)

    # Non-negative severities (softplus head).
    assert torch.all(out["severities"] >= 0)
    # Strictly-positive PSD.
    assert torch.all(out["rho_radial"] > 0)
    # Complex gains.
    assert out["gains"].is_complex()


def test_build_lowrank_shape():
    model = _make(rank=6, c=1, h=8, w=8)
    b = 3
    out = model(torch.zeros(b, dtype=torch.long), torch.zeros(b))
    u = model.build_lowrank(out["gains"])
    n = 1 * 8 * 8
    assert u.shape == (b, n, 6)
    assert u.is_complex()


def test_defaults_to_zeros_when_unconditioned():
    """A bare batch-size hint (no ids) must still produce valid outputs."""
    model = _make()
    out = model(field_coord=torch.zeros(2))
    assert out["severities"].shape[0] == 2


def test_gradients_flow_to_conditioner():
    model = _make()
    out = model(torch.zeros(2, dtype=torch.long), torch.zeros(2))
    loss = out["severities"].sum() + out["rho_radial"].sum() + out["gains"].abs().sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
