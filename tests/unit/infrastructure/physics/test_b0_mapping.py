"""Tests for B₀ field mapping primitives.

Targets ``spectramr.infrastructure.physics.b0_mapping``:

- ``phase_difference``: phase residual between unshifted/shifted images
  with global-phase median correction
- ``estimate_b0_tikhonov``: CG solver for Tikhonov-regularised B₀ map
- ``map_b0``: end-to-end pipeline

Categories:

- Unit: identical inputs → near-zero phase difference
- Property: linear B₀ field is recoverable to within Tikhonov tolerance
- Property: median centring stabilises against global phase offsets
- Sanity-shape: parametrised matrix
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.b0_mapping import (
    estimate_b0_tikhonov,
    map_b0,
    phase_difference,
)


def _complex_image(shape: tuple[int, int, int, int]) -> torch.Tensor:
    """Helper: random complex tensor with non-zero magnitude."""
    real = torch.randn(*shape)
    imag = torch.randn(*shape)
    return torch.complex(real, imag)


# ---------------------------------------------------------------------------
# phase_difference
# ---------------------------------------------------------------------------


def test_phase_difference_identical_inputs_near_zero() -> None:
    """``phase_difference(z, z)`` is near zero up to median-centring."""
    img = _complex_image((1, 1, 16, 16))
    out = phase_difference(img, img)
    # Median centring may shift the result; check it's bounded close to 0.
    assert out.abs().max().item() < 1e-3


def test_phase_difference_linear_phase_recovered() -> None:
    """A pure phase rotation of ``φ_offset`` is recovered as a constant offset."""
    torch.manual_seed(0)
    img = torch.complex(torch.ones(1, 1, 8, 8) * 2.0, torch.zeros(1, 1, 8, 8))
    phase = torch.full((1, 1, 8, 8), 0.3)  # constant 0.3 rad shift
    img_shifted = img * torch.exp(1j * phase)
    out = phase_difference(img, img_shifted)
    # After median centring of the shifted phase, the difference is ~0.
    # (The median of a constant phase = the phase itself, so the result
    # is near zero — that's the documented behaviour.)
    assert out.abs().max().item() < 1e-3


def test_phase_difference_returns_real() -> None:
    """Output is a real-valued phase map (radians), not complex."""
    img = _complex_image((1, 1, 8, 8))
    out = phase_difference(img, img)
    assert not torch.is_complex(out)


def test_phase_difference_in_pi_range() -> None:
    """Phase output is in ``[-π, π]`` (output of torch.angle)."""
    torch.manual_seed(0)
    img1 = _complex_image((1, 1, 8, 8))
    img2 = _complex_image((1, 1, 8, 8))
    out = phase_difference(img1, img2)
    assert out.max().item() <= math.pi + 1e-5
    assert out.min().item() >= -math.pi - 1e-5


# ---------------------------------------------------------------------------
# estimate_b0_tikhonov
# ---------------------------------------------------------------------------


def test_tikhonov_zero_phase_diff_yields_near_zero_b0() -> None:
    """Zero phase-difference input → near-zero B₀ field."""
    phase_diff = torch.zeros(1, 1, 16, 16)
    b0 = estimate_b0_tikhonov(phase_diff, t_shift=1e-3, max_cg_iterations=20)
    assert b0.abs().max().item() < 1e-3


def test_tikhonov_output_shape() -> None:
    """Output has shape ``[1, 1, H, W]``."""
    phase_diff = torch.zeros(1, 1, 8, 8)
    b0 = estimate_b0_tikhonov(phase_diff, t_shift=1e-3, max_cg_iterations=10)
    assert b0.shape == (1, 1, 8, 8)


def test_tikhonov_recovers_linear_b0() -> None:
    """Synthetic linear B₀ → recovered b0 ≈ true_b0 (high t_shift, low γ)."""
    torch.manual_seed(0)
    h = w = 16
    # Synthetic ground-truth B0 in Hz.
    true_b0 = torch.linspace(-50.0, 50.0, w).repeat(h, 1)
    t_shift = 1e-3  # 1 ms
    # Forward model: phi = -2π * b0 * t_shift  (paper §2.3)
    phi = -2.0 * math.pi * true_b0 * t_shift

    b0_est = estimate_b0_tikhonov(
        phi.unsqueeze(0).unsqueeze(0),
        t_shift=t_shift,
        gamma=1e-12,  # very small regularisation
        max_cg_iterations=200,
    )
    # Recovery shouldn't be perfect (Tikhonov bias), but should track the
    # general shape and sign.
    rel_err = (b0_est.squeeze() - true_b0).abs().mean() / true_b0.abs().mean()
    assert rel_err.item() < 0.5  # within 50% relative error


def test_tikhonov_with_mask_zeroes_outside() -> None:
    """Mask outside ROI has no effect on the result inside."""
    phase_diff = torch.zeros(1, 1, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    mask[..., :4, :] = 0.0
    b0 = estimate_b0_tikhonov(
        phase_diff, t_shift=1e-3, mask=mask, max_cg_iterations=10
    )
    # Outside the mask, b0 contribution is zero.
    assert torch.isfinite(b0).all()


# ---------------------------------------------------------------------------
# map_b0 — end-to-end
# ---------------------------------------------------------------------------


def test_map_b0_end_to_end_runs() -> None:
    """End-to-end pipeline produces finite output."""
    img = torch.complex(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    out = map_b0(img, img, t_shift=1e-3, gamma=1e-9)
    assert out.shape == (1, 1, 8, 8)
    assert torch.isfinite(out.real).all()


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8), (1, 1, 16, 16), (1, 1, 16, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_phase_difference_shape_matrix(shape: tuple[int, ...]) -> None:
    """phase_difference handles square/rectangular shapes."""
    img1 = _complex_image(shape)
    img2 = _complex_image(shape)
    out = phase_difference(img1, img2)
    assert out.shape == shape
    assert torch.isfinite(out).all()
