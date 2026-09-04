"""Tests for the FiLM-conditioned SIREN INR primitives (models/inr/siren.py)."""

from __future__ import annotations

import torch

from spectramr.models.inr.siren import FiLMLayer, SIRENWithFiLM


def test_film_layer_is_identity_at_init() -> None:
    # Regression: FiLMLayer must start as the IDENTITY (gamma=1, beta=0) for any style
    # so it does not perturb the SIREN's variance-preserving init. The previous init
    # wrote ones into a temporary (weight.data.mean(...)), leaving gamma_proj random,
    # so the layer scaled features by ~0.08 at init.
    torch.manual_seed(0)
    film = FiLMLayer(feature_dim=8, style_dim=4)
    feats = torch.randn(2, 5, 8)
    for style in (torch.zeros(2, 4), torch.randn(2, 4) * 3.0):
        out = film(feats, style)
        assert torch.allclose(out, feats, atol=1e-6)  # passthrough regardless of style


def test_film_layer_modulates_after_params_move() -> None:
    film = FiLMLayer(feature_dim=8, style_dim=4)
    with torch.no_grad():
        for p in film.parameters():
            p.normal_(std=0.3)
    feats = torch.randn(2, 5, 8)
    a = film(feats, torch.zeros(2, 4))
    b = film(feats, torch.ones(2, 4))
    assert not torch.allclose(a, b)  # style genuinely modulates once params move


def test_siren_with_film_style_inert_at_init() -> None:
    # With FiLM identity at init, the SIREN output must be INVARIANT to the style
    # (the conditioning only becomes load-bearing once FiLM moves off identity).
    torch.manual_seed(0)
    net = SIRENWithFiLM(in_features=2, hidden_features=32, hidden_layers=3, style_dim=8)
    coords = torch.rand(2, 16, 2) * 2 - 1
    a = net(coords, torch.zeros(2, 8))
    b = net(coords, torch.randn(2, 8) * 5.0)
    assert torch.allclose(a, b, atol=1e-6)
