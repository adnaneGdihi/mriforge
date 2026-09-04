"""Tests for FieldPhysicsEmbeddedScoreUNet (A-5.2 standalone field-conditioned score prior)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.field_physics_score_unet import FieldPhysicsEmbeddedScoreUNet


def _net(**kw) -> FieldPhysicsEmbeddedScoreUNet:
    return FieldPhysicsEmbeddedScoreUNet(width=24, time_dim=16, **kw)


def _args(n: int = 2):
    return {"timesteps": torch.tensor([10] * n), "field_strength": torch.full((n,), 1.5)}


def test_forward_shape() -> None:
    out = _net()(torch.rand(2, 1, 16, 16), **_args())
    assert out.shape == (2, 1, 16, 16)


def test_field_blind_at_init_then_field_dependent() -> None:
    # FieldFiLM is zero-init (identity) -> field-blind at init (by design); after escaping the
    # identity the PHYSICS field drives the score.
    torch.manual_seed(0)
    m = _net().eval()
    x = torch.rand(2, 1, 16, 16)
    t = torch.tensor([10, 10])
    o1 = m(x, timesteps=t, field_strength=torch.tensor([0.1, 0.1]))
    o2 = m(x, timesteps=t, field_strength=torch.tensor([7.0, 7.0]))
    assert torch.allclose(o1, o2)  # identity FiLM at init -> field-blind
    with torch.no_grad():
        m.film.mlp[-1].weight.normal_(0.0, 0.3)
    o3 = m(x, timesteps=t, field_strength=torch.tensor([0.1, 0.1]))
    o4 = m(x, timesteps=t, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(o3, o4)  # physics field now drives the score


def test_cfg_null_path_differs_after_escape() -> None:
    torch.manual_seed(0)
    m = _net().eval()
    with torch.no_grad():
        m.film.mlp[-1].weight.normal_(0.0, 0.3)
        m.null_gamma.normal_(1.0, 0.3)
        m.null_beta.normal_(0.0, 0.3)
    x = torch.rand(2, 1, 16, 16)
    t = torch.tensor([10, 10])
    b = torch.tensor([7.0, 7.0])
    o_field = m(x, timesteps=t, field_strength=b)
    o_null = m(x, timesteps=t, field_strength=b, field_cond=torch.tensor([0.0, 0.0]))
    assert not torch.allclose(o_field, o_null)


def test_physics_ablation_is_param_matched() -> None:
    on = FieldPhysicsEmbeddedScoreUNet(width=24, time_dim=16, enable_physics=True)
    off = FieldPhysicsEmbeddedScoreUNet(width=24, time_dim=16, enable_physics=False)
    assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())


def test_consumable_as_posterior_sampling_prior() -> None:
    # The standalone prior exposes eps(x_t, t, b) -> can be wrapped as the score_model(x_t, y, t)
    # interface PosteriorLangevinEnsemble expects (a reusable cross-field prior).
    m = _net().eval()
    x = torch.rand(2, 1, 16, 16)
    t = torch.tensor([5, 5])

    def score_model(xt, y, tt):  # the PosteriorLangevinEnsemble score_model(x_t, y, t) interface
        return m(xt, timesteps=tt, field_strength=torch.full((xt.shape[0],), 3.0))

    out = score_model(x, None, t)
    assert out.shape == x.shape


def test_trainable() -> None:
    m = FieldPhysicsEmbeddedScoreUNet(width=16, time_dim=8)
    loss = m(torch.rand(2, 1, 16, 16), timesteps=torch.tensor([3, 7]),
             field_strength=torch.tensor([1.5, 3.0])).pow(2).mean()
    loss.backward()
    g = m.film.mlp[-1].weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0


def test_rejects_complex_and_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), timesteps=torch.tensor([1]),
               field_strength=torch.tensor([1.5]))
    with pytest.raises(ValueError, match="magnitude-only"):
        FieldPhysicsEmbeddedScoreUNet(in_channels=2)


def test_cond_image_explicitly_ignored() -> None:
    # The shared field_guided_diffusion strategy passes cond_image; this UNCONDITIONAL prior
    # declares it and ignores it (explicit contract, not a silent **_ drop) -> output unchanged.
    m = _net().eval()
    x = torch.rand(2, 1, 16, 16)
    a = _args()
    o_no = m(x, **a)
    o_cond = m(x, cond_image=torch.rand(2, 1, 16, 16), **a)
    assert torch.allclose(o_no, o_cond)  # source image has no effect (prior, not translator)


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "field_physics_score_unet" in MODEL_REGISTRY
