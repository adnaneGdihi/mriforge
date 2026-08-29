"""Comprehensive unit tests for TRELLIS utility helpers.

Tests the TRELLIS utility functions for datatype management, parameter manipulation,
and modulation operations.
"""

from unittest.mock import patch

import torch
from torch import nn

from mriforge.models.blocks.trellis_utils import (
    convert_module_to_f16,
    convert_module_to_f32,
    modulate,
    scale_module,
    zero_module,
)


# @pytest.fixture(autouse=True)
def clean_mocks():
    """Ensure mocks are cleaned up before/after each test."""
    patch.stopall()
    yield
    patch.stopall()


class TestConvertModuleToF16:
    """Test convert_module_to_f16 function."""

    def test_convert_linear_to_f16(self):
        """Test converting Linear layer to float16."""
        layer = nn.Linear(10, 5)
        original_dtype = layer.weight.dtype

        # Ensure it's not already float16
        assert original_dtype != torch.float16

        convert_module_to_f16(layer)

        assert layer.weight.dtype == torch.float16
        assert layer.bias.dtype == torch.float16

    def test_convert_conv2d_to_f16(self):
        """Test converting Conv2d layer to float16."""
        layer = nn.Conv2d(3, 64, 3, padding=1)
        original_dtype = layer.weight.dtype

        assert original_dtype != torch.float16

        convert_module_to_f16(layer)

        assert layer.weight.dtype == torch.float16
        assert layer.bias.dtype == torch.float16

    def test_convert_unsupported_module_to_f16(self):
        """Test that unsupported modules are not converted."""
        layer = nn.LayerNorm(64)  # LayerNorm is not in _FP16_MODULES
        original_dtype = layer.weight.dtype

        convert_module_to_f16(layer)

        # Should remain unchanged
        assert layer.weight.dtype == original_dtype
        assert layer.bias.dtype == original_dtype

    def test_convert_nested_module_to_f16(self):
        """Test converting nested modules to float16."""
        # The function only converts modules that are directly instances of supported types
        # It doesn't recursively convert children of containers like Sequential
        linear1 = nn.Linear(10, 20)
        linear2 = nn.Linear(20, 5)

        original_dtype1 = linear1.weight.dtype
        original_dtype2 = linear2.weight.dtype

        # Convert individual layers
        convert_module_to_f16(linear1)
        convert_module_to_f16(linear2)

        assert linear1.weight.dtype == torch.float16
        assert linear1.bias.dtype == torch.float16
        assert linear2.weight.dtype == torch.float16
        assert linear2.bias.dtype == torch.float16

        # Ensure they were actually changed
        assert linear1.weight.dtype != original_dtype1
        assert linear2.weight.dtype != original_dtype2

    def test_gradient_preservation_f16(self):
        """Test that gradients are preserved during f16 conversion."""
        layer = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad is not None
        original_grad_dtype = layer.weight.grad.dtype

        convert_module_to_f16(layer)

        assert layer.weight.grad is not None
        assert layer.weight.grad.dtype == torch.float16


class TestConvertModuleToF32:
    """Test convert_module_to_f32 function."""

    def test_convert_linear_to_f32(self):
        """Test converting Linear layer back to float32."""
        layer = nn.Linear(10, 5).half()  # Start as float16
        assert layer.weight.dtype == torch.float16

        convert_module_to_f32(layer)

        assert layer.weight.dtype == torch.float32
        assert layer.bias.dtype == torch.float32

    def test_convert_conv2d_to_f32(self):
        """Test converting Conv2d layer back to float32."""
        layer = nn.Conv2d(3, 64, 3, padding=1).half()
        assert layer.weight.dtype == torch.float16

        convert_module_to_f32(layer)

        assert layer.weight.dtype == torch.float32
        assert layer.bias.dtype == torch.float32

    def test_convert_unsupported_module_to_f32(self):
        """Test that unsupported modules are not converted to f32."""
        layer = nn.LayerNorm(64).half()
        original_dtype = layer.weight.dtype

        convert_module_to_f32(layer)

        # Should remain unchanged
        assert layer.weight.dtype == original_dtype

    def test_gradient_preservation_f32(self):
        """Test that gradients are preserved during f32 conversion."""
        layer = nn.Linear(10, 5).half()
        x = torch.randn(2, 10).half()  # Match input dtype to layer dtype
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.weight.grad.dtype == torch.float16

        convert_module_to_f32(layer)

        assert layer.weight.grad is not None
        assert layer.weight.grad.dtype == torch.float32


class TestZeroModule:
    """Test zero_module function."""

    def test_zero_linear_module(self):
        """Test zeroing Linear module parameters."""
        layer = nn.Linear(10, 5)
        # Set non-zero weights and biases
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.fill_(2.0)

        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert not torch.allclose(layer.bias, torch.zeros_like(layer.bias))

        result = zero_module(layer)

        assert result is layer  # Returns the same module
        assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_zero_conv2d_module(self):
        """Test zeroing Conv2d module parameters."""
        layer = nn.Conv2d(3, 64, 3, padding=1)
        with torch.no_grad():
            layer.weight.fill_(1.5)
            layer.bias.fill_(-1.0)

        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert not torch.allclose(layer.bias, torch.zeros_like(layer.bias))

        zero_module(layer)

        assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_zero_nested_module(self):
        """Test zeroing nested module parameters."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # Set non-zero parameters
        with torch.no_grad():
            for layer in model:
                if hasattr(layer, "weight"):
                    layer.weight.fill_(1.0)
                    layer.bias.fill_(1.0)

        zero_module(model)

        for layer in model:
            if hasattr(layer, "weight"):
                assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
                assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_zero_module_with_gradients(self):
        """Test that zero_module works with gradients present."""
        layer = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad is not None

        zero_module(layer)

        # Parameters should be zeroed, but gradients remain
        assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert layer.weight.grad is not None


class TestScaleModule:
    """Test scale_module function."""

    def test_scale_linear_module(self):
        """Test scaling Linear module parameters."""
        layer = nn.Linear(10, 5)
        scale_factor = 2.0

        # Set known weights and biases
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.fill_(3.0)

        original_weight = layer.weight.clone()
        original_bias = layer.bias.clone()

        result = scale_module(layer, scale_factor)

        assert result is layer  # Returns the same module
        assert torch.allclose(layer.weight, original_weight * scale_factor)
        assert torch.allclose(layer.bias, original_bias * scale_factor)

    def test_scale_conv2d_module(self):
        """Test scaling Conv2d module parameters."""
        layer = nn.Conv2d(3, 64, 3, padding=1)
        scale_factor = 0.5

        with torch.no_grad():
            layer.weight.fill_(4.0)
            layer.bias.fill_(2.0)

        original_weight = layer.weight.clone()
        original_bias = layer.bias.clone()

        scale_module(layer, scale_factor)

        assert torch.allclose(layer.weight, original_weight * scale_factor)
        assert torch.allclose(layer.bias, original_bias * scale_factor)

    def test_scale_nested_module(self):
        """Test scaling nested module parameters."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        scale_factor = 3.0

        # Store original values
        originals = []
        for layer in model:
            if hasattr(layer, "weight"):
                originals.append((layer.weight.clone(), layer.bias.clone()))

        scale_module(model, scale_factor)

        idx = 0
        for layer in model:
            if hasattr(layer, "weight"):
                orig_weight, orig_bias = originals[idx]
                assert torch.allclose(layer.weight, orig_weight * scale_factor)
                assert torch.allclose(layer.bias, orig_bias * scale_factor)
                idx += 1

    def test_scale_with_zero(self):
        """Test scaling with zero factor."""
        layer = nn.Linear(10, 5)
        with torch.no_grad():
            layer.weight.fill_(5.0)
            layer.bias.fill_(1.0)

        scale_module(layer, 0.0)

        assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_scale_with_negative(self):
        """Test scaling with negative factor."""
        layer = nn.Linear(10, 5)
        scale_factor = -1.5

        with torch.no_grad():
            layer.weight.fill_(2.0)
            layer.bias.fill_(4.0)

        scale_module(layer, scale_factor)

        assert torch.allclose(layer.weight, torch.full_like(layer.weight, -3.0))
        assert torch.allclose(layer.bias, torch.full_like(layer.bias, -6.0))


class TestModulate:
    """Test modulate function."""

    def test_modulate_basic(self):
        """Test basic modulation operation."""
        batch_size, seq_len, channels = 2, 4, 64

        x = torch.randn(batch_size, seq_len, channels)
        shift = torch.randn(batch_size, channels)
        scale = torch.randn(batch_size, channels)

        result = modulate(x, shift, scale)

        # Check shape
        assert result.shape == x.shape

        # Check the modulation formula: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        expected = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        assert torch.allclose(result, expected)

    def test_modulate_with_zeros(self):
        """Test modulation with zero shift and scale."""
        x = torch.randn(2, 8, 32)
        shift = torch.zeros(2, 32)
        scale = torch.zeros(2, 32)

        result = modulate(x, shift, scale)

        # Should be identity operation
        assert torch.allclose(result, x)

    def test_modulate_with_ones(self):
        """Test modulation with shift=1 and scale=1."""
        x = torch.randn(1, 4, 16)
        shift = torch.ones(1, 16)
        scale = torch.ones(1, 16)

        result = modulate(x, shift, scale)

        expected = x * 2 + 1  # x * (1+1) + 1
        assert torch.allclose(result, expected)

    def test_modulate_shape_consistency(self):
        """Test modulation with different batch sizes."""
        # Different batch sizes
        x1 = torch.randn(1, 6, 128)
        shift1 = torch.randn(1, 128)
        scale1 = torch.randn(1, 128)

        x2 = torch.randn(3, 10, 128)
        shift2 = torch.randn(3, 128)
        scale2 = torch.randn(3, 128)

        result1 = modulate(x1, shift1, scale1)
        result2 = modulate(x2, shift2, scale2)

        assert result1.shape == x1.shape
        assert result2.shape == x2.shape

    def test_modulate_gradient_flow(self):
        """Test that gradients flow through modulation."""
        x = torch.randn(2, 4, 32, requires_grad=True)
        shift = torch.randn(2, 32, requires_grad=True)
        scale = torch.randn(2, 32, requires_grad=True)

        result = modulate(x, shift, scale)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert shift.grad is not None
        assert scale.grad is not None

        assert x.grad.shape == x.shape
        assert shift.grad.shape == shift.shape
        assert scale.grad.shape == scale.shape

    def test_modulate_dtype_preservation(self):
        """Test that modulation preserves dtype."""
        x = torch.randn(2, 4, 16).half()
        shift = torch.randn(2, 16).half()
        scale = torch.randn(2, 16).half()

        result = modulate(x, shift, scale)

        assert result.dtype == x.dtype

    def test_modulate_device_consistency(self):
        """Test device consistency in modulation."""
        if torch.cuda.is_available():
            x = torch.randn(2, 4, 32).cuda()
            shift = torch.randn(2, 32).cuda()
            scale = torch.randn(2, 32).cuda()

            result = modulate(x, shift, scale)

            assert result.device == x.device


class TestUtilityIntegration:
    """Test integration between utility functions."""

    def test_f16_f32_round_trip(self):
        """Test round-trip conversion between f16 and f32."""
        layer = nn.Linear(10, 5)
        original_weight = layer.weight.clone()
        original_bias = layer.bias.clone()

        # Convert to f16
        convert_module_to_f16(layer)
        assert layer.weight.dtype == torch.float16

        # Convert back to f32
        convert_module_to_f32(layer)
        assert layer.weight.dtype == torch.float32

        # Values should be preserved (within precision limits)
        assert torch.allclose(layer.weight, original_weight, rtol=1e-3)
        assert torch.allclose(layer.bias, original_bias, rtol=1e-3)

    def test_zero_then_scale(self):
        """Test zeroing then scaling a module."""
        layer = nn.Linear(8, 4)
        with torch.no_grad():
            layer.weight.fill_(2.0)
            layer.bias.fill_(3.0)

        # Zero the module
        zero_module(layer)
        assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))

        # Scale it (should still be zero)
        scale_module(layer, 5.0)
        assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_scale_then_zero(self):
        """Test scaling then zeroing a module."""
        layer = nn.Linear(8, 4)
        with torch.no_grad():
            layer.weight.fill_(2.0)
            layer.bias.fill_(1.0)

        # Scale the module
        scale_module(layer, 3.0)
        assert torch.allclose(layer.weight, torch.full_like(layer.weight, 6.0))

        # Zero it
        zero_module(layer)
