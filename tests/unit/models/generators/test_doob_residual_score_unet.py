"""Tests for DoobResidualScoreUNet (B-1.10 source-conditioned 7T residual score net)."""

from __future__ import annotations

import inspect

import pytest
import torch

from spectramr.models.generators.doob_residual_score_unet import DoobResidualScoreUNet


def _net() -> DoobResidualScoreUNet:
    return DoobResidualScoreUNet(width=16, n_blocks=2)


def test_forward_shape() -> None:
    out = _net()(
        torch.rand(2, 1, 16, 16),
        timesteps=torch.tensor([3, 7]),
        cond_image=torch.rand(2, 1, 16, 16),
    )
    assert out.shape == (2, 1, 16, 16)


def test_source_conditioning_fires() -> None:
    # ANTI-FACADE (#16): the source condition must actually change the output, else the arm
    # silently collapses to an unconditional net and the residual mechanism is inert. The
    # concat channel feeds body_in directly (non-zero weights), so even at init the source
    # changes the output — no training step needed.
    torch.manual_seed(0)
    m = _net().eval()
    x = torch.rand(1, 1, 16, 16)
    t = torch.tensor([5])
    out_a = m(x, timesteps=t, cond_image=torch.zeros(1, 1, 16, 16))
    out_b = m(x, timesteps=t, cond_image=torch.ones(1, 1, 16, 16))
    assert not torch.allclose(
        out_a, out_b
    ), "output ignores the source condition (facade)"


def test_timestep_is_load_bearing() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(2, 1, 16, 16)
    src = torch.rand(2, 1, 16, 16)
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        loss = (
            (
                m(x, timesteps=torch.tensor([1, 30]), cond_image=src)
                - torch.randn_like(x)
            )
            .pow(2)
            .mean()
        )
        loss.backward()
        opt.step()
    m.eval()
    a = m(x, timesteps=torch.tensor([1, 1]), cond_image=src)
    b = m(x, timesteps=torch.tensor([39, 39]), cond_image=src)
    assert not torch.allclose(a, b)


def test_conditions_on_source_signature() -> None:
    # Unlike the marginal net, this net's forward REQUIRES cond_image (the source) — that is the
    # residual-conditioning seam. Both timesteps and cond_image are keyword-only with no default
    # so the Tier-2 probe injects them.
    params = inspect.signature(DoobResidualScoreUNet.forward).parameters
    assert "cond_image" in params
    assert params["cond_image"].default is inspect.Parameter.empty
    assert params["timesteps"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        _net()(
            torch.rand(1, 1, 8, 8), timesteps=torch.tensor([1])
        )  # missing cond_image


def test_rejects_complex() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(
            torch.rand(1, 1, 8, 8, dtype=torch.cfloat),
            timesteps=torch.tensor([1]),
            cond_image=torch.rand(1, 1, 8, 8),
        )


def test_rejects_mismatched_cond_shape() -> None:
    with pytest.raises(ValueError, match="cond_image"):
        _net()(
            torch.rand(1, 1, 16, 16),
            timesteps=torch.tensor([1]),
            cond_image=torch.rand(1, 1, 8, 8),
        )


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        DoobResidualScoreUNet(in_channels=2)


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "doob_residual_score_unet" in MODEL_REGISTRY
    assert MODEL_REGISTRY["doob_residual_score_unet"]["mode"] == "doob_bridge"
