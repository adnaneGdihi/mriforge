"""Tests for RecoverabilityVIBNet (B-2.3 deep-VIB restorer)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.recoverability_vib_net import RecoverabilityVIBNet


def _net(**kw) -> RecoverabilityVIBNet:
    return RecoverabilityVIBNet(width=24, latent_channels=8, n_downsample=2, **kw)


def test_encode_decode_shapes() -> None:
    m = _net()
    x = torch.rand(2, 1, 64, 64)
    mu, logvar = m.encode(x)
    assert mu.shape == logvar.shape and mu.shape[1] == 8
    assert m.decode(mu, x.shape[-2:]).shape == (2, 1, 64, 64)


def test_forward_is_deterministic_posterior_mean() -> None:
    # forward() uses mu (no sampling) -> repeatable, probe/validation-stable.
    m = _net().eval()
    x = torch.rand(2, 1, 64, 64)
    assert torch.allclose(m(x), m(x))


def test_reparameterize_is_stochastic() -> None:
    torch.manual_seed(0)
    m = _net()
    mu, logvar = m.encode(torch.rand(2, 1, 32, 32))
    assert not torch.allclose(m.reparameterize(mu, logvar), m.reparameterize(mu, logvar))


def test_odd_size_recon_matches_input() -> None:
    # bilinear up/down can mis-round odd sizes; decode interpolates back to the input HxW.
    m = _net()
    x = torch.rand(1, 1, 50, 54)
    assert m(x).shape == (1, 1, 50, 54)


def test_rejects_complex_and_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net().encode(torch.rand(1, 1, 16, 16, dtype=torch.cfloat))
    with pytest.raises(ValueError, match="magnitude-only"):
        RecoverabilityVIBNet(in_channels=2)


def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="n_downsample"):
        RecoverabilityVIBNet(n_downsample=0)
    with pytest.raises(ValueError, match="latent_channels"):
        RecoverabilityVIBNet(latent_channels=0)


def test_registered() -> None:
    from mriforge.models.init_registry import populate_model_registry
    from mriforge.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "recoverability_vib_net" in MODEL_REGISTRY
