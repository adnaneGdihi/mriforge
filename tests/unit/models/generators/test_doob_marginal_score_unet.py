"""Tests for DoobMarginalScoreUNet (B-1.10 unconditional 7T marginal score net)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.doob_marginal_score_unet import DoobMarginalScoreUNet


def _net() -> DoobMarginalScoreUNet:
    return DoobMarginalScoreUNet(width=16, n_blocks=2)


def test_forward_shape() -> None:
    out = _net()(torch.rand(2, 1, 16, 16), timesteps=torch.tensor([3, 7]))
    assert out.shape == (2, 1, 16, 16)


def test_timestep_is_load_bearing() -> None:
    # The score depends on the diffusion time; after a step of training the time embedding
    # is non-trivial, so different timesteps give different eps. Train one step to escape
    # the zero-init FiLM identity, then assert time-dependence.
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(2, 1, 16, 16)
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        loss = (m(x, timesteps=torch.tensor([1, 30])) - torch.randn_like(x)).pow(2).mean()
        loss.backward()
        opt.step()
    m.eval()
    a = m(x, timesteps=torch.tensor([1, 1]))
    b = m(x, timesteps=torch.tensor([39, 39]))
    assert not torch.allclose(a, b)


def test_is_unconditional_no_field_arg() -> None:
    # The marginal score (Doob h) is UNCONDITIONAL: forward takes ONLY x + timesteps, no
    # field_strength / cond_image (that is what distinguishes it from FieldGuidedScoreUNet).
    import inspect

    params = set(inspect.signature(DoobMarginalScoreUNet.forward).parameters)
    assert "field_strength" not in params
    assert "cond_image" not in params


def test_timesteps_is_required_for_probe() -> None:
    # No default on timesteps -> the Tier-2 synthetic forward probe fills it (a defaulted
    # arg would be skipped, failing a healthy model at the --probe gate).
    import inspect

    p = inspect.signature(DoobMarginalScoreUNet.forward).parameters["timesteps"]
    assert p.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        _net()(torch.rand(1, 1, 8, 8))  # calling without timesteps must fail


def test_rejects_complex() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), timesteps=torch.tensor([1]))


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        DoobMarginalScoreUNet(in_channels=2)


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "doob_marginal_score_unet" in MODEL_REGISTRY
