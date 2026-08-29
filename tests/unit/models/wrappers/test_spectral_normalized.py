r"""Tests for ``SpectralNormalizedDenoiser`` (paradigm-expansion PR-9 / M1).

Targets ``mriforge.models.wrappers.spectral_normalized``.

Categories:

- Construction: rejects non-``nn.Module`` inputs, rejects bad
  ``n_power_iterations``
- Wrapping a Conv-only network reports the Conv layer in
  ``normalised_layers``
- ``is_lipschitz_constrained`` is True for a wrapped Conv net,
  False for an empty container
- Forward pass produces the same output shape as the inner module
- Spectral norm of every wrapped layer's weight is ≤ 1 + ε after
  enough power iterations
- The wrapper is *non-expansive*: ``||f(a) - f(b)|| ≤ ||a - b||``
  for two random inputs, given a depth-1 conv network
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mriforge.models.wrappers.spectral_normalized import (
    SpectralNormalizedDenoiser,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_non_module_inner_rejected() -> None:
    """``inner`` must be an ``nn.Module``."""
    with pytest.raises(TypeError, match="inner"):
        SpectralNormalizedDenoiser(inner=lambda x: x)  # type: ignore[arg-type]


def test_zero_power_iterations_rejected() -> None:
    """``n_power_iterations < 1`` rejected."""
    inner = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    with pytest.raises(ValueError, match="n_power_iterations"):
        SpectralNormalizedDenoiser(inner, n_power_iterations=0)


# ---------------------------------------------------------------------------
# Wrapping detection
# ---------------------------------------------------------------------------


def test_wrap_single_conv_layer() -> None:
    """A single Conv2d gets spectral-norm reparameterised."""
    inner = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    wrapped = SpectralNormalizedDenoiser(inner)
    assert wrapped.is_lipschitz_constrained is True
    # Single Conv2d → just one entry.
    assert len(wrapped.normalised_layers) == 1


def test_wrap_sequential_finds_all_supported_layers() -> None:
    """Every Conv / Linear layer in a Sequential is wrapped."""
    inner = nn.Sequential(
        nn.Conv2d(4, 8, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(8, 4, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    wrapped = SpectralNormalizedDenoiser(inner)
    # Two convs + one linear = 3 wrapped.
    assert len(wrapped.normalised_layers) == 3


def test_empty_container_not_lipschitz_constrained() -> None:
    """A wrapper around a container with no Conv / Linear is not constrained."""
    inner = nn.Sequential(nn.SiLU(), nn.Tanh())
    wrapped = SpectralNormalizedDenoiser(inner)
    assert wrapped.is_lipschitz_constrained is False
    assert wrapped.normalised_layers == ()


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_forward_preserves_shape() -> None:
    """Wrapping doesn't change the output shape of the inner module."""
    inner = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    wrapped = SpectralNormalizedDenoiser(inner)
    x = torch.randn(2, 4, 8, 8)
    out = wrapped(x)
    assert out.shape == x.shape


def test_forward_is_differentiable() -> None:
    """Backward through the wrapper reaches the input."""
    inner = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    wrapped = SpectralNormalizedDenoiser(inner)
    x = torch.randn(1, 4, 8, 8, requires_grad=True)
    out = wrapped(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Spectral-norm contract
# ---------------------------------------------------------------------------


def test_wrapped_weight_spectral_norm_close_to_one() -> None:
    """After warm-up, every wrapped weight matrix has spectral norm ≈ 1."""
    inner = nn.Linear(8, 8)
    wrapped = SpectralNormalizedDenoiser(inner, n_power_iterations=20)
    # Run a few forward passes to drive power iterations to convergence.
    for _ in range(5):
        wrapped(torch.randn(4, 8))
    # The reparameterised weight is exposed via ``inner.weight`` after
    # spectral_norm is applied.
    weight = inner.weight
    sigma = torch.linalg.svdvals(weight)[0].item()
    assert sigma <= 1.0 + 1e-3


def test_non_expansive_property() -> None:
    """``||f(a) - f(b)|| ≤ ||a - b||`` for the spectral-norm-wrapped layer.

    A depth-1 Conv2d after spectral-norm warm-up should be 1-Lipschitz.
    """
    torch.manual_seed(0)
    inner = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
    wrapped = SpectralNormalizedDenoiser(inner, n_power_iterations=30)
    # Drive the power iteration to near-convergence with random inputs.
    for _ in range(8):
        wrapped(torch.randn(2, 4, 16, 16))

    a = torch.randn(1, 4, 16, 16)
    b = a + torch.randn(1, 4, 16, 16) * 0.1
    diff_in = (a - b).norm().item()
    diff_out = (wrapped(a) - wrapped(b)).norm().item()
    # Allow a small slack because Conv2d's padding inflates the
    # effective spectral norm slightly above the matrix-form bound.
    assert diff_out <= 1.05 * diff_in


# ---------------------------------------------------------------------------
# Audit-gate
# ---------------------------------------------------------------------------


def test_normalised_layers_includes_root_for_bare_conv() -> None:
    """For a bare Conv2d (no submodules), the layer is listed under '<root>'."""
    inner = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    wrapped = SpectralNormalizedDenoiser(inner)
    assert "<root>" in wrapped.normalised_layers
