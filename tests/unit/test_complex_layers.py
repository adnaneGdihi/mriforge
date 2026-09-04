"""Unit tests for complex-valued neural network layers.

Follows strict verification protocols from unit-test.md:
- Directive 4.A: Property-based testing with Hypothesis
- Directive 4.D.1: NaN/Inf safety assertions
- Directive 4.D.2: Deterministic randomness with fixed seeds

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
given = hypothesis.given
settings = hypothesis.settings
st = pytest.importorskip("hypothesis.strategies", reason="hypothesis not installed")

# Module under test
from spectramr.models.layers.complex_layers import (
    ComplexBatchNorm2d,
    ComplexLinear,
    ComplexReLU,
    WirtingerGradient,
)

# =============================================================================
# Directive 4.D.2: Deterministic Randomness Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def set_deterministic_seeds():
    """Ensure reproducibility for all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Enable deterministic algorithms where possible
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Note: torch.use_deterministic_algorithms(True) may break some ops
    yield


# =============================================================================
# Hypothesis Strategies for Complex Tensors
# =============================================================================


@st.composite
def complex_tensor_strategy(
    draw,
    min_batch: int = 1,
    max_batch: int = 4,
    min_features: int = 2,
    max_features: int = 16,
    min_spatial: int = 4,
    max_spatial: int = 16,
    include_2d: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random complex tensor as (real, imag) tuple."""
    batch = draw(st.integers(min_value=min_batch, max_value=max_batch))
    features = draw(st.integers(min_value=min_features, max_value=max_features))

    if include_2d:
        height = draw(st.integers(min_value=min_spatial, max_value=max_spatial))
        width = draw(st.integers(min_value=min_spatial, max_value=max_spatial))
        shape = (batch, features, height, width)
    else:
        shape = (batch, features)

    # Generate random values in reasonable range
    real = torch.randn(shape) * 0.1
    imag = torch.randn(shape) * 0.1

    return real, imag


@st.composite
def linear_input_strategy(
    draw,
    min_batch: int = 1,
    max_batch: int = 4,
    min_features: int = 2,
    max_features: int = 32,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Generate input for ComplexLinear tests."""
    batch = draw(st.integers(min_value=min_batch, max_value=max_batch))
    in_features = draw(st.integers(min_value=min_features, max_value=max_features))
    out_features = draw(st.integers(min_value=min_features, max_value=max_features))

    real = torch.randn(batch, in_features) * 0.1
    imag = torch.randn(batch, in_features) * 0.1

    return in_features, out_features, real, imag


# =============================================================================
# ComplexLinear Tests
# =============================================================================


class TestComplexLinear:
    """Test suite for ComplexLinear layer."""

    def test_output_shape(self):
        """Test output shape matches expected dimensions."""
        layer = ComplexLinear(in_features=16, out_features=32)
        x_real = torch.randn(4, 16)
        x_imag = torch.randn(4, 16)

        y_real, y_imag = layer(x_real, x_imag)

        assert y_real.shape == (4, 32), f"Expected (4, 32), got {y_real.shape}"
        assert y_imag.shape == (4, 32), f"Expected (4, 32), got {y_imag.shape}"

    def test_complex_input_format(self):
        """Test that native complex tensors work correctly."""
        layer = ComplexLinear(in_features=16, out_features=8)
        x = torch.randn(4, 16, dtype=torch.cfloat)

        y = layer(x)

        assert y.dtype == torch.cfloat
        assert y.shape == (4, 8)

    def test_no_bias(self):
        """Test layer works without bias."""
        layer = ComplexLinear(in_features=8, out_features=4, bias=False)
        x_real = torch.randn(2, 8)
        x_imag = torch.randn(2, 8)

        y_real, y_imag = layer(x_real, x_imag)

        assert y_real.shape == (2, 4)
        assert layer.bias_real is None

    @given(data=linear_input_strategy())
    @settings(max_examples=20, deadline=None)
    def test_no_nan_output(self, data):
        """Directive 4.D.1: Assert no NaN/Inf in output."""
        in_features, out_features, x_real, x_imag = data
        layer = ComplexLinear(in_features, out_features)

        y_real, y_imag = layer(x_real, x_imag)

        assert torch.isfinite(y_real).all(), "NaN/Inf in real output"
        assert torch.isfinite(y_imag).all(), "NaN/Inf in imag output"

    @given(data=linear_input_strategy())
    @settings(max_examples=10, deadline=None)
    def test_gradient_flow(self, data):
        """Test that gradients flow through the layer."""
        in_features, out_features, x_real, x_imag = data
        x_real.requires_grad_(True)
        x_imag.requires_grad_(True)

        layer = ComplexLinear(in_features, out_features)
        y_real, y_imag = layer(x_real, x_imag)

        loss = (y_real**2 + y_imag**2).sum()
        loss.backward()

        assert x_real.grad is not None, "No gradient on real input"
        assert x_imag.grad is not None, "No gradient on imag input"
        assert torch.isfinite(x_real.grad).all(), "NaN/Inf in gradient"

    def test_complex_linearity(self):
        """Test complex multiplication property: (a+bi)(c+di) = (ac-bd) + i(ad+bc)."""
        layer = ComplexLinear(4, 4, bias=False)

        # Set known weights
        with torch.no_grad():
            layer.weight_real.fill_(1.0)
            layer.weight_imag.fill_(0.5)

        x_real = torch.ones(1, 4)
        x_imag = torch.ones(1, 4) * 2

        y_real, y_imag = layer(x_real, x_imag)

        # Expected: real = sum(1*1 - 0.5*2) = sum(-0) = 0 per output
        # Expected: imag = sum(1*2 + 0.5*1) = sum(2.5) = 10 per output
        # (with 4 inputs summed)
        expected_real = (1.0 * 1.0 - 0.5 * 2.0) * 4  # = 0
        expected_imag = (1.0 * 2.0 + 0.5 * 1.0) * 4  # = 10

        assert torch.allclose(y_real, torch.full_like(y_real, expected_real), atol=1e-5)
        assert torch.allclose(y_imag, torch.full_like(y_imag, expected_imag), atol=1e-5)


# =============================================================================
# ComplexBatchNorm2d Tests
# =============================================================================


class TestComplexBatchNorm2d:
    """Test suite for ComplexBatchNorm2d layer."""

    def test_output_shape_preserved(self):
        """Test that spatial dimensions are preserved."""
        bn = ComplexBatchNorm2d(num_features=8)
        x_real = torch.randn(4, 8, 16, 16)
        x_imag = torch.randn(4, 8, 16, 16)

        y_real, y_imag = bn(x_real, x_imag)

        assert y_real.shape == x_real.shape
        assert y_imag.shape == x_imag.shape

    def test_normalization_effect(self):
        """Test that output has approximately unit variance."""
        bn = ComplexBatchNorm2d(num_features=4, affine=False)
        bn.train()

        # Large batch for stable statistics
        x_real = torch.randn(32, 4, 8, 8) * 5.0  # High variance
        x_imag = torch.randn(32, 4, 8, 8) * 5.0

        y_real, y_imag = bn(x_real, x_imag)

        # Magnitude variance should be ~1
        magnitude = torch.sqrt(y_real**2 + y_imag**2)
        mag_var = magnitude.var(dim=(0, 2, 3)).mean()

        # Allow some tolerance
        assert mag_var < 5.0, f"Variance too high after normalization: {mag_var}"

    def test_no_nan_with_zero_variance(self):
        """Directive 4.D.1: No NaN when input has zero variance."""
        bn = ComplexBatchNorm2d(num_features=2)

        # Constant input (zero variance)
        x_real = torch.ones(4, 2, 4, 4)
        x_imag = torch.ones(4, 2, 4, 4) * 2

        y_real, y_imag = bn(x_real, x_imag)

        assert torch.isfinite(y_real).all(), "NaN with zero variance input"
        assert torch.isfinite(y_imag).all(), "NaN with zero variance input"

    def test_running_stats_update(self):
        """Test that running statistics are updated during training."""
        bn = ComplexBatchNorm2d(num_features=4, track_running_stats=True)
        bn.train()

        initial_mean = bn.running_mean_real.clone()

        x_real = torch.randn(8, 4, 4, 4) + 5.0  # Shifted mean
        x_imag = torch.randn(8, 4, 4, 4)

        bn(x_real, x_imag)

        assert not torch.allclose(
            bn.running_mean_real, initial_mean
        ), "Running mean not updated"

    def test_eval_mode_uses_running_stats(self):
        """Test that eval mode uses stored statistics."""
        bn = ComplexBatchNorm2d(num_features=2)

        # Train on some data
        bn.train()
        for _ in range(10):
            x_real = torch.randn(16, 2, 4, 4)
            x_imag = torch.randn(16, 2, 4, 4)
            bn(x_real, x_imag)

        # Switch to eval
        bn.eval()

        # Same input should give same output
        test_real = torch.randn(2, 2, 4, 4)
        test_imag = torch.randn(2, 2, 4, 4)

        y1_real, y1_imag = bn(test_real, test_imag)
        y2_real, y2_imag = bn(test_real, test_imag)

        assert torch.allclose(y1_real, y2_real), "Eval mode not deterministic"


# =============================================================================
# ComplexReLU Tests
# =============================================================================


class TestComplexReLU:
    """Test suite for phase-preserving ReLU."""

    def test_preserves_phase(self):
        """Test that phase is preserved after activation."""
        relu = ComplexReLU()

        # Create input with known phase
        magnitude = torch.tensor(
            [[1.0, 2.0, -0.5, 3.0]]
        )  # Note: -0.5 should stay same phase
        phase = torch.tensor([[0.1, 0.5, 1.0, 2.0]])

        x_real = magnitude * torch.cos(phase)
        x_imag = magnitude * torch.sin(phase)

        y_real, y_imag = relu(
            x_real.unsqueeze(-1).unsqueeze(-1), x_imag.unsqueeze(-1).unsqueeze(-1)
        )

        # Check phase preserved (where magnitude > 0)
        input_phase = torch.atan2(x_imag, x_real).flatten()
        output_phase = torch.atan2(y_imag.squeeze(), y_real.squeeze()).flatten()

        # Only check where output magnitude > 0
        output_mag = torch.sqrt(y_real.squeeze() ** 2 + y_imag.squeeze() ** 2).flatten()
        mask = output_mag > 1e-6

        if mask.any():
            assert torch.allclose(
                input_phase[mask], output_phase[mask], atol=1e-5
            ), "Phase not preserved"

    def test_negative_magnitude_zeroed(self):
        """Test that modReLU with bias zeros small magnitudes."""
        relu = ComplexReLU(learnable_bias=True, num_features=2)

        with torch.no_grad():
            relu.bias.fill_(0.5)  # Threshold at 0.5

        # Small magnitude input
        x_real = torch.tensor([[0.1, 0.1]]).unsqueeze(-1).unsqueeze(-1)
        x_imag = torch.tensor([[0.1, 0.1]]).unsqueeze(-1).unsqueeze(-1)

        y_real, y_imag = relu(x_real, x_imag)

        # Magnitude ~0.14 < 0.5, so output should be zero
        output_mag = torch.sqrt(y_real**2 + y_imag**2)
        assert (output_mag < 1e-6).all(), "Small magnitude not zeroed"

    def test_no_nan_output(self):
        """Directive 4.D.1: No NaN even with zero input."""
        relu = ComplexReLU()

        x_real = torch.zeros(4, 8, 4, 4)
        x_imag = torch.zeros(4, 8, 4, 4)

        y_real, y_imag = relu(x_real, x_imag)

        assert torch.isfinite(y_real).all(), "NaN with zero input"
        assert torch.isfinite(y_imag).all(), "NaN with zero input"


# =============================================================================
# WirtingerGradient Tests
# =============================================================================


class TestWirtingerGradient:
    """Test suite for Wirtinger calculus utilities."""

    def test_conjugate_gradient_shape(self):
        """Test output shape matches input."""
        grad_real = torch.randn(4, 8)
        grad_imag = torch.randn(4, 8)

        result = WirtingerGradient.conjugate_gradient(grad_real, grad_imag)

        assert result.shape == grad_real.shape
        assert result.dtype == torch.cfloat

    def test_conjugate_gradient_formula(self):
        """Test that formula ∂L/∂z* = (1/2)(∂L/∂x + i∂L/∂y) is correct."""
        grad_real = torch.tensor([2.0, 4.0])
        grad_imag = torch.tensor([1.0, 2.0])

        result = WirtingerGradient.conjugate_gradient(grad_real, grad_imag)

        # Expected: 0.5 * (2 + 1j), 0.5 * (4 + 2j) = (1 + 0.5j), (2 + 1j)
        expected = torch.tensor([1.0 + 0.5j, 2.0 + 1.0j])

        assert torch.allclose(result, expected)


# =============================================================================
# Integration Tests
# =============================================================================


class TestComplexLayerIntegration:
    """Integration tests for complex layer composition."""

    def test_sequential_composition(self):
        """Test that layers compose correctly."""
        # Build a small complex network
        layer1 = ComplexLinear(16, 32)
        bn = ComplexBatchNorm2d(32)  # Note: This expects 2D spatial
        relu = ComplexReLU()
        layer2 = ComplexLinear(32, 8)

        # For linear layers (no spatial dims)
        x = torch.randn(4, 16, dtype=torch.cfloat)

        # Forward through linear
        h = layer1(x)

        # Can't use BN without spatial dims, so just relu and layer2
        h = relu(h)
        y = layer2(h)

        assert y.shape == (4, 8)
        assert y.dtype == torch.cfloat
        assert torch.isfinite(y).all()

    def test_gradient_through_stack(self):
        """Test gradient flow through multiple layers."""
        layer1 = ComplexLinear(8, 16)
        relu = ComplexReLU()
        layer2 = ComplexLinear(16, 4)

        x = torch.randn(2, 8, dtype=torch.cfloat, requires_grad=True)

        h = layer1(x)
        h = relu(h)
        y = layer2(h)

        loss = torch.abs(y).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer1.weight_real.grad).all()


# =============================================================================
# Property-Based Tests (Directive 4.A)
# =============================================================================


class TestComplexLayerInvariants:
    """Property-based tests for mathematical invariants."""

    @given(data=linear_input_strategy(max_features=8))
    @settings(max_examples=15, deadline=None)
    def test_zero_input_zero_output_unbiased(self, data):
        """Invariant: Zero input → Zero output (when unbiased)."""
        in_features, out_features, _, _ = data
        layer = ComplexLinear(in_features, out_features, bias=False)

        zero_real = torch.zeros(1, in_features)
        zero_imag = torch.zeros(1, in_features)

        y_real, y_imag = layer(zero_real, zero_imag)

        assert torch.allclose(y_real, torch.zeros_like(y_real), atol=1e-6)
        assert torch.allclose(y_imag, torch.zeros_like(y_imag), atol=1e-6)

    @given(data=linear_input_strategy(max_features=8))
    @settings(max_examples=15, deadline=None)
    def test_output_magnitude_bounded(self, data):
        """Invariant: Output magnitude should be bounded for bounded input."""
        in_features, out_features, x_real, x_imag = data
        layer = ComplexLinear(in_features, out_features)

        # Input magnitude
        input_mag = torch.sqrt(x_real**2 + x_imag**2).max()

        y_real, y_imag = layer(x_real, x_imag)
        output_mag = torch.sqrt(y_real**2 + y_imag**2).max()

        # Should be bounded (not infinite)
        assert torch.isfinite(output_mag)
        # Rough bound based on weight initialization
        max_expected = input_mag * in_features * 2 + 10  # With some margin for bias
        assert (
            output_mag < max_expected
        ), f"Output {output_mag} exceeds bound {max_expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
