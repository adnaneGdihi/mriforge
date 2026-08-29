"""Tests for FieldConditionedINR (B-2.8 field-conditioned SIREN INR model)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.field_conditioned_inr import FieldConditionedINR


def _net() -> FieldConditionedINR:
    return FieldConditionedINR(feat_dim=8, hidden_features=64, hidden_layers=3, style_dim=32)


def test_forward_shape_native() -> None:
    out = _net()(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 7.0]))
    assert out.shape == (2, 1, 16, 16)


def test_resolution_free_super_resolution_seam() -> None:
    # The INR is resolution-free: querying out_hw renders at an arbitrary grid (the
    # encoder features are grid_sampled to the query lattice). 16x16 input -> 32x32.
    m = _net().eval()
    out = m(torch.rand(1, 1, 16, 16), field_strength=torch.tensor([7.0]), out_hw=(32, 32))
    assert out.shape == (1, 1, 32, 32)
    # ...and a non-square denser grid.
    out2 = m(torch.rand(1, 1, 16, 16), field_strength=torch.tensor([7.0]), out_hw=(24, 40))
    assert out2.shape == (1, 1, 24, 40)


def test_rejects_complex() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.0]))


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        FieldConditionedINR(in_channels=2, out_channels=1)


def test_field_is_load_bearing_after_perturbation() -> None:
    # Once FiLM/field_mlp move off identity init, the continuous field changes the
    # render (the field is genuine conditioning, not inert).
    m = _net()
    with torch.no_grad():
        for fl in m.siren.film_layers:
            if fl is not None:
                for p in fl.parameters():
                    p.normal_(std=0.3)
        for p in m.field_mlp.parameters():
            p.normal_(std=0.3)
    x = torch.rand(2, 1, 16, 16)
    lo = m(x, field_strength=torch.tensor([0.1, 0.1]))
    hi = m(x, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(lo, hi)


def test_field_inert_at_identity_init() -> None:
    # With the (now correct) FiLM identity init, the field has no effect at step 0 —
    # documents the init contract the load-bearing test perturbs past.
    m = _net().eval()
    x = torch.rand(1, 1, 16, 16)
    a = m(x, field_strength=torch.tensor([0.1]))
    b = m(x, field_strength=torch.tensor([7.0]))
    assert torch.allclose(a, b, atol=1e-6)


def test_registered() -> None:
    from mriforge.models.registry import MODEL_REGISTRY

    assert "field_conditioned_inr" in MODEL_REGISTRY


# --- Multi-contrast conditioning (mrixfields2026 broad rollout) ---


def test_contrast_widens_field_mlp_input() -> None:
    m = FieldConditionedINR(
        feat_dim=8, hidden_features=64, hidden_layers=3, style_dim=32,
        use_contrast_conditioning=True, num_contrasts=3,
    )
    assert m.field_mlp[0].in_features == 1 + 3  # field + num_contrasts


def test_contrast_requires_id_when_enabled() -> None:
    m = FieldConditionedINR(
        feat_dim=8, hidden_features=64, hidden_layers=3, style_dim=32,
        use_contrast_conditioning=True, num_contrasts=3,
    )
    with pytest.raises(ValueError, match="contrast_id"):
        m(torch.rand(1, 1, 16, 16), field_strength=torch.tensor([7.0]))


def test_contrast_changes_output_after_perturbation() -> None:
    torch.manual_seed(0)
    m = FieldConditionedINR(
        feat_dim=8, hidden_features=64, hidden_layers=3, style_dim=32,
        use_contrast_conditioning=True, num_contrasts=3,
    ).eval()
    with torch.no_grad():
        for fl in m.siren.film_layers:
            if fl is not None:
                for p in fl.parameters():
                    p.normal_(std=0.3)
        for p in m.field_mlp.parameters():
            p.normal_(std=0.3)
    x = torch.rand(2, 1, 16, 16)
    fs = torch.tensor([7.0, 7.0])
    a = m(x, field_strength=fs, contrast_id=torch.tensor([0, 0]))
    b = m(x, field_strength=fs, contrast_id=torch.tensor([2, 2]))
    assert not torch.allclose(a, b)
