"""Tests for FieldGuidedScoreUNet (B-3.5 field-CFG score net)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.field_guided_score_unet import FieldGuidedScoreUNet


def _net() -> FieldGuidedScoreUNet:
    return FieldGuidedScoreUNet(width=16, time_dim=16)


def test_forward_shape_eps_prediction() -> None:
    net = _net()
    x = torch.rand(2, 1, 16, 16)
    src = torch.rand(2, 1, 16, 16)
    out = net(
        x,
        timesteps=torch.tensor([3, 7]),
        field_strength=torch.tensor([0.1, 7.0]),
        cond_image=src,
    )
    assert out.shape == (2, 1, 16, 16)


def test_requires_cond_image() -> None:
    net = _net()
    with pytest.raises(ValueError, match="cond_image"):
        net(
            torch.rand(1, 1, 16, 16),
            timesteps=torch.tensor([0]),
            field_strength=torch.tensor([3.0]),
        )


def test_rejects_complex_input() -> None:
    net = _net()
    with pytest.raises(ValueError, match="magnitude"):
        net(
            torch.rand(1, 1, 8, 8, dtype=torch.cfloat),
            timesteps=torch.tensor([0]),
            field_strength=torch.tensor([3.0]),
            cond_image=torch.rand(1, 1, 8, 8),
        )


def test_identity_init_is_field_agnostic_at_start() -> None:
    # FieldFiLMBlock + null FiLM are identity-init (no warm-up shock): at init the
    # field has no effect yet. This documents the init contract (the field becomes
    # load-bearing only after the FiLM weights move — see the next test).
    net = _net()
    x = torch.rand(1, 1, 16, 16)
    src = torch.rand(1, 1, 16, 16)
    t = torch.tensor([4])
    a = net(x, timesteps=t, field_strength=torch.tensor([0.1]), cond_image=src)
    b = net(x, timesteps=t, field_strength=torch.tensor([7.0]), cond_image=src)
    assert torch.allclose(a, b)


def test_field_and_null_are_load_bearing_after_perturbation() -> None:
    # ANTI-FACADE (#16): once the FiLM/null params move off identity, BOTH the
    # continuous field AND the field_cond (CFG null) mask must change the output —
    # proving the field path is a genuine conditioning channel, not inert.
    net = _net()
    with torch.no_grad():
        net.film.mlp[-1].weight.normal_(std=0.5)
        net.film.mlp[-1].bias.normal_(std=0.5)
        net.null_gamma.normal_()
        net.null_beta.normal_()
    x = torch.rand(2, 1, 16, 16)
    src = torch.rand(2, 1, 16, 16)
    t = torch.tensor([5, 5])
    lo = net(x, timesteps=t, field_strength=torch.tensor([0.1, 0.1]), cond_image=src)
    hi = net(x, timesteps=t, field_strength=torch.tensor([7.0, 7.0]), cond_image=src)
    assert not torch.allclose(lo, hi)  # continuous field changes the output
    cond = net(
        x, timesteps=t, field_strength=torch.tensor([7.0, 7.0]), cond_image=src,
        field_cond=torch.ones(2),
    )
    null = net(
        x, timesteps=t, field_strength=torch.tensor([7.0, 7.0]), cond_image=src,
        field_cond=torch.zeros(2),
    )
    assert not torch.allclose(cond, null)  # CFG null path is distinct from field path


def test_registered_in_model_registry() -> None:
    # The top-level import above runs the @register_model decorator.
    from mriforge.models.registry import MODEL_REGISTRY

    assert "field_guided_score_unet" in MODEL_REGISTRY


# --- Multi-contrast conditioning (mrixfields2026 broad rollout) ---


def test_contrast_widens_film_sequence_dim() -> None:
    net = FieldGuidedScoreUNet(
        width=16, time_dim=16, use_contrast_conditioning=True, num_contrasts=3
    )
    assert net.film.sequence_dim == 16 + 3  # time_dim + num_contrasts


def test_contrast_requires_id_when_enabled() -> None:
    # #15: enabled knob with no contrast_id must raise, not silently mode-average.
    net = FieldGuidedScoreUNet(
        width=16, time_dim=16, use_contrast_conditioning=True, num_contrasts=3
    )
    with pytest.raises(ValueError, match="contrast_id"):
        net(
            torch.rand(1, 1, 16, 16),
            timesteps=torch.tensor([3]),
            field_strength=torch.tensor([3.0]),
            cond_image=torch.rand(1, 1, 16, 16),
        )


def test_contrast_changes_output_after_perturbation() -> None:
    # #16 mechanism-fires: a different contrast id changes the score once FiLM moves
    # off its identity init.
    net = FieldGuidedScoreUNet(
        width=16, time_dim=16, use_contrast_conditioning=True, num_contrasts=3
    )
    with torch.no_grad():
        net.film.mlp[-1].weight.normal_(std=0.5)
        net.film.mlp[-1].bias.normal_(std=0.5)
    x = torch.rand(2, 1, 16, 16)
    src = torch.rand(2, 1, 16, 16)
    t = torch.tensor([5, 5])
    fs = torch.tensor([3.0, 3.0])
    a = net(x, timesteps=t, field_strength=fs, cond_image=src, contrast_id=torch.tensor([0, 0]))
    b = net(x, timesteps=t, field_strength=fs, cond_image=src, contrast_id=torch.tensor([2, 2]))
    assert not torch.allclose(a, b)
