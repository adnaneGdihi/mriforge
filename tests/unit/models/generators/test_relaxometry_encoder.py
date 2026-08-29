"""Tests for RelaxometryEncoder (MICCAI MRIxFields2026, idea 2.1)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.relaxometry_encoder import RelaxometryEncoder


def test_forward_multicontrast_to_single_image() -> None:
    m = RelaxometryEncoder(in_channels=3)
    x = torch.rand(2, 3, 16, 16)
    y = m(x, field_strength=torch.tensor([7.0, 7.0]))
    assert y.shape == (2, 1, 16, 16)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_exposes_param_maps_and_beta() -> None:
    m = RelaxometryEncoder(in_channels=3)
    m.predict_parameters(torch.rand(2, 3, 16, 16))
    assert set(m.last_param_maps) == {"rho", "T1", "T2"}
    assert m.last_dispersion_beta is not None
    assert m.estimated_params == ("rho", "T1", "T2")


def test_beta_zero_is_field_invariant() -> None:
    """Consistency limit (Prop 3): beta=0 -> T1 does not change with field, so the
    render is identical at any field. The 'no dispersion' baseline arm."""
    m = RelaxometryEncoder(
        in_channels=3, learn_beta_per_tissue=False, dispersion_beta=0.0
    )
    x = torch.rand(2, 3, 16, 16)
    p = m.predict_parameters(x)
    assert torch.allclose(
        m.render(p, torch.tensor([0.1, 0.1])), m.render(p, torch.tensor([7.0, 7.0]))
    )


def test_render_backpropagates_through_signal_op() -> None:
    m = RelaxometryEncoder(in_channels=3)
    x = torch.rand(2, 3, 16, 16, requires_grad=False)
    m(x, field_strength=torch.tensor([7.0, 7.0])).mean().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_opaque_residual_highpass_removes_dc() -> None:
    """The opaque projector is a high-pass: a constant (pure-DC) field is exactly
    removed in the interior, so the residual cannot install a measurement-independent
    blob (pitfall #20). Tested on the operator directly (independent of the CNN)."""
    m = RelaxometryEncoder(in_channels=3)
    assert m.refiner is not None
    hp = m.refiner._highpass(torch.full((1, 1, 32, 32), 0.7))
    # interior (away from the padded borders) of a constant is exactly cancelled
    assert abs(float(hp[:, :, 8:-8, 8:-8].mean())) < 1e-5


def test_opaque_residual_shape() -> None:
    m = RelaxometryEncoder(in_channels=3)
    y_det = torch.rand(2, 1, 32, 32)
    r = m.opaque_residual(y_det, torch.rand(2, 3, 32, 32))
    assert r.shape == y_det.shape


def test_disabled_residual_returns_zero() -> None:
    m = RelaxometryEncoder(in_channels=3, use_opaque_residual=False)
    y_det = torch.rand(2, 1, 16, 16)
    r = m.opaque_residual(y_det, torch.rand(2, 3, 16, 16))
    assert torch.count_nonzero(r) == 0


def test_registered() -> None:
    from mriforge.models.registry import get_model_class

    assert get_model_class("relaxometry_encoder") is RelaxometryEncoder


def test_rejects_multichannel_output() -> None:
    with pytest.raises(ValueError):
        RelaxometryEncoder(in_channels=3, out_channels=2)
