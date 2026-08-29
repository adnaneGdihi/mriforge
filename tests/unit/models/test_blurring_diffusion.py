"""Tests for the blurring-diffusion model and its DCT heat schedule.

Asserts registry resolution, schedule monotonicity (alpha strictly
decreasing in t for high-frequency modes), forward shape, v-target
shape, AMP survival, and DCT round-trip consistency of the forward
process at t=0.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.diffusion.blurring_diffusion import BlurringDiffusion
from mriforge.models.diffusion.schedules.blurring_schedule import BlurringSchedule
from mriforge.models.registry import MODEL_REGISTRY
from tests.unit.models._diffusion_base import (
    assert_amp_survives,
    assert_schedule_monotone_decreasing,
)

# Small image size keeps the DCT grid + U-Net cheap for CI.
IMG = 16
T = 50


def _model() -> BlurringDiffusion:
    return BlurringDiffusion(
        in_channels=1,
        out_channels=1,
        num_timesteps=T,
        image_size=IMG,
        base_channels=8,
        time_embedding_dim=32,
    )


class TestRegistration:
    def test_registered_name(self) -> None:
        assert "blurring_diffusion" in MODEL_REGISTRY
        entry = MODEL_REGISTRY["blurring_diffusion"]
        assert entry["training_mode"] == "diffusion"

    def test_constructible_with_no_args(self) -> None:
        # Factory path constructs with filtered kwargs; defaults must work.
        model = BlurringDiffusion()
        assert isinstance(model, torch.nn.Module)


class TestBlurringSchedule:
    def test_buffers_present(self) -> None:
        sched = BlurringSchedule(num_timesteps=T, image_size=IMG)
        names = dict(sched.named_buffers())
        assert "lambda_k" in names
        assert "tau" in names
        assert "sigma" in names
        assert names["lambda_k"].shape == (IMG, IMG)

    def test_alpha_monotone_decreasing(self) -> None:
        sched = BlurringSchedule(num_timesteps=T, image_size=IMG)
        assert_schedule_monotone_decreasing(sched.alpha, T)

    def test_alpha_dc_is_one(self) -> None:
        sched = BlurringSchedule(num_timesteps=T, image_size=IMG)
        t = torch.tensor([T - 1])
        alpha = sched.alpha(t)
        # DC mode (eigenvalue 0) never decays.
        assert torch.allclose(alpha[..., 0, 0], torch.ones(1), atol=1e-6)

    def test_sigma_monotone_increasing(self) -> None:
        sched = BlurringSchedule(num_timesteps=T, image_size=IMG)
        sig = sched.sigma
        assert torch.all(sig[1:] >= sig[:-1])

    def test_q_sample_shape(self) -> None:
        sched = BlurringSchedule(num_timesteps=T, image_size=IMG)
        x0 = torch.randn(2, 1, IMG, IMG)
        t = torch.randint(0, T, (2,))
        out = sched.q_sample(x0, t, torch.randn_like(x0))
        assert out.shape == x0.shape

    def test_q_sample_t0_near_identity(self) -> None:
        # At t=0, blur time tau≈0 so alpha≈1 -> q_sample ≈ x0 + sigma_0*noise.
        sched = BlurringSchedule(
            num_timesteps=T, image_size=IMG, sigma_min=1e-6, sigma_max=0.1
        )
        x0 = torch.randn(1, 1, IMG, IMG)
        t = torch.zeros(1, dtype=torch.long)
        out = sched.q_sample(x0, t, torch.zeros_like(x0))
        assert torch.allclose(out, x0, atol=1e-3)

    def test_invalid_sigma_order_raises(self) -> None:
        with pytest.raises(ValueError):
            BlurringSchedule(num_timesteps=T, image_size=IMG, sigma_min=0.5, sigma_max=0.1)


class TestBlurringDiffusionModel:
    def test_forward_shape(self) -> None:
        model = _model()
        x = torch.randn(2, 1, IMG, IMG)
        t = torch.randint(0, T, (2,))
        out = model(x, t)
        assert out.shape == x.shape

    def test_v_target_shape(self) -> None:
        model = _model()
        x0 = torch.randn(2, 1, IMG, IMG)
        noise = torch.randn_like(x0)
        t = torch.randint(0, T, (2,))
        v = model.schedule.v_target(x0, noise, t)
        assert v.shape == x0.shape

    def test_training_loss_finite(self) -> None:
        model = _model()
        x0 = torch.randn(2, 1, IMG, IMG)
        loss = model.training_loss(x0)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_training_loss_backprop(self) -> None:
        model = _model()
        x0 = torch.randn(2, 1, IMG, IMG)
        loss = model.training_loss(x0)
        loss.backward()
        grads = [p.grad for p in model.unet.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_amp_survives(self) -> None:
        model = _model()
        x0 = torch.randn(2, 1, IMG, IMG)
        assert_amp_survives(model.training_loss, x0)
