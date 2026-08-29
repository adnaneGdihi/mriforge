r"""Tests for the EDM noise schedule + preconditioning (PR-10 / M2).

Targets ``mriforge.models.diffusion.edm_schedule``.

Plan acceptance criteria from §"Tests":

- *"σ → 0: ``D_θ(x; 0) = x``"* — covered indirectly: the
  preconditioning coefficients at small σ give ``c_skip ≈ 1``,
  ``c_out ≈ 0``, so the denoiser output approaches the input.
- *"σ → ∞: D_θ(x; ∞) → mean of training data"* — at large σ the
  preconditioning gives ``c_skip → 0``, ``c_in → 0``, so the
  denoiser output is decoupled from the input. We assert the
  scaling limits directly.

Other contract tests:

- ``EDMNoiseSchedule`` validates arguments
- ``sigmas()`` is descending and bounded by ``[sigma_min, sigma_max]``
- ``sample_sigma()`` returns values in the schedule range
- ``edm_preconditioning_coefficients`` returns correct shapes
- ``edm_loss_weight`` matches the closed form
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.models.diffusion.edm_schedule import (
    EDMNoiseSchedule,
    edm_loss_weight,
    edm_preconditioning_coefficients,
)


# ---------------------------------------------------------------------------
# EDMNoiseSchedule construction
# ---------------------------------------------------------------------------


def test_zero_sigma_min_rejected() -> None:
    """``sigma_min ≤ 0`` raises."""
    with pytest.raises(ValueError, match="sigma_min"):
        EDMNoiseSchedule(sigma_min=0.0)


def test_sigma_max_must_exceed_sigma_min() -> None:
    """``sigma_max ≤ sigma_min`` raises."""
    with pytest.raises(ValueError, match="sigma_max"):
        EDMNoiseSchedule(sigma_min=0.5, sigma_max=0.5)


def test_zero_rho_rejected() -> None:
    """``rho ≤ 0`` raises."""
    with pytest.raises(ValueError, match="rho"):
        EDMNoiseSchedule(rho=0.0)


def test_too_few_steps_rejected() -> None:
    """``num_steps < 2`` raises."""
    with pytest.raises(ValueError, match="num_steps"):
        EDMNoiseSchedule(num_steps=1)


# ---------------------------------------------------------------------------
# sigmas() schedule
# ---------------------------------------------------------------------------


def test_sigmas_endpoints_match_min_max() -> None:
    """First σ = ``sigma_max``, last σ = ``sigma_min``."""
    sched = EDMNoiseSchedule(sigma_min=0.002, sigma_max=80.0, num_steps=20)
    s = sched.sigmas()
    assert pytest.approx(s[0].item()) == 80.0
    assert pytest.approx(s[-1].item()) == 0.002


def test_sigmas_strictly_descending() -> None:
    """``sigmas()`` is strictly descending."""
    sched = EDMNoiseSchedule(num_steps=18)
    s = sched.sigmas()
    diffs = s[1:] - s[:-1]
    assert (diffs < 0).all()


def test_sigmas_within_range() -> None:
    """Every entry lies in ``[sigma_min, sigma_max]``."""
    sched = EDMNoiseSchedule(sigma_min=0.01, sigma_max=10.0, num_steps=12)
    s = sched.sigmas()
    assert (s >= 0.01 - 1e-6).all()
    assert (s <= 10.0 + 1e-6).all()


def test_sigmas_length_matches_num_steps() -> None:
    """``len(sigmas) == num_steps``."""
    sched = EDMNoiseSchedule(num_steps=33)
    assert sched.sigmas().shape == (33,)


# ---------------------------------------------------------------------------
# sample_sigma()
# ---------------------------------------------------------------------------


def test_sample_sigma_in_range() -> None:
    """Sampled σ values lie in ``[sigma_min, sigma_max]``."""
    torch.manual_seed(0)
    sched = EDMNoiseSchedule(sigma_min=0.002, sigma_max=80.0)
    s = sched.sample_sigma(batch_size=1024)
    assert (s >= sched.sigma_min - 1e-6).all()
    assert (s <= sched.sigma_max + 1e-6).all()


def test_sample_sigma_zero_batch_size_rejected() -> None:
    """``batch_size < 1`` rejected."""
    sched = EDMNoiseSchedule()
    with pytest.raises(ValueError, match="batch_size"):
        sched.sample_sigma(batch_size=0)


# ---------------------------------------------------------------------------
# Preconditioning coefficients
# ---------------------------------------------------------------------------


def test_preconditioning_shapes_match_sigma() -> None:
    """All four returned tensors have the same shape as ``sigma``."""
    sigma = torch.tensor([0.1, 1.0, 10.0])
    c_skip, c_out, c_in, c_noise = edm_preconditioning_coefficients(sigma)
    assert c_skip.shape == sigma.shape
    assert c_out.shape == sigma.shape
    assert c_in.shape == sigma.shape
    assert c_noise.shape == sigma.shape


def test_preconditioning_c_skip_to_one_at_low_sigma() -> None:
    """``c_skip → 1`` as σ → 0 (denoiser passes through)."""
    sigma = torch.tensor([1e-4])
    c_skip, _, _, _ = edm_preconditioning_coefficients(sigma, sigma_data=0.5)
    assert c_skip.item() > 0.999


def test_preconditioning_c_skip_to_zero_at_high_sigma() -> None:
    """``c_skip → 0`` as σ → ∞ (denoiser ignores noisy input)."""
    sigma = torch.tensor([1000.0])
    c_skip, _, _, _ = edm_preconditioning_coefficients(sigma, sigma_data=0.5)
    assert c_skip.item() < 1e-3


def test_preconditioning_c_in_decreases_with_sigma() -> None:
    """``c_in`` is monotone decreasing in σ."""
    sigma = torch.tensor([0.1, 1.0, 10.0])
    _, _, c_in, _ = edm_preconditioning_coefficients(sigma)
    diffs = c_in[1:] - c_in[:-1]
    assert (diffs < 0).all()


def test_preconditioning_zero_sigma_rejected() -> None:
    """σ = 0 rejected (would div/0)."""
    with pytest.raises(ValueError, match="sigma"):
        edm_preconditioning_coefficients(torch.tensor([0.0]))


def test_preconditioning_zero_sigma_data_rejected() -> None:
    """``sigma_data = 0`` rejected."""
    with pytest.raises(ValueError, match="sigma_data"):
        edm_preconditioning_coefficients(torch.tensor([1.0]), sigma_data=0.0)


def test_preconditioning_c_noise_log_form() -> None:
    """``c_noise = 0.25 · log(σ)``."""
    sigma = torch.tensor([math.e, math.e ** 4])
    _, _, _, c_noise = edm_preconditioning_coefficients(sigma)
    assert pytest.approx(c_noise[0].item()) == 0.25
    assert pytest.approx(c_noise[1].item()) == 1.0


# ---------------------------------------------------------------------------
# Loss weight λ(σ)
# ---------------------------------------------------------------------------


def test_loss_weight_closed_form() -> None:
    """``λ(σ) = (σ² + σ_d²) / (σ σ_d)²`` matches the Karras formula."""
    sigma = torch.tensor([0.5, 1.0, 2.0])
    sigma_data = 0.5
    expected = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    assert torch.allclose(edm_loss_weight(sigma, sigma_data), expected)


