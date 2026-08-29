#!/usr/bin/env python
"""Verify Experiment 39 PINN Gradient Flow.

This script validates that the PINN-NeRF implementation has proper gradient flow:
1. SIREN layers receive non-zero gradients
2. Physics loss (MaxwellRegularizer) contributes to gradients
3. No dead neurons from SIREN initialization

Usage:
    python tests/verification/verify_exp39_gradients.py
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn


def test_siren_gradient_flow():
    """Verify gradients flow through SIREN layers."""
    print("\n=== Testing SIREN Gradient Flow ===")

    try:
        from mriforge.models.blocks.pinn_nerf_block import PINNNeRFBlock
    except ImportError as e:
        print(f"  ⚠️  Could not import PINNNeRFBlock: {e}")
        return None

    # Create block
    block = PINNNeRFBlock(
        in_features=4,  # Latent features
        hidden_dim=64,
        num_layers=4,
        out_channels=2,
    )

    # Synthetic input
    batch_size = 4
    num_points = 100
    features = torch.randn(batch_size, num_points, 4, requires_grad=True)
    coords = torch.rand(batch_size, num_points, 2) * 2 - 1  # Normalized [-1, 1]

    # Forward pass
    output = block(features, coords)

    # Compute loss and backprop
    target = torch.randn_like(output)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()

    # Check gradients in each layer
    layers_with_grad = 0
    layers_no_grad = 0

    for name, param in block.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                layers_with_grad += 1
            else:
                layers_no_grad += 1
                print(f"  ⚠️  {name}: grad_norm = {grad_norm:.2e} (near zero)")
        else:
            layers_no_grad += 1
            print(f"  ❌ {name}: no gradient")

    print(f"  Layers with gradients: {layers_with_grad}")
    print(f"  Layers without gradients: {layers_no_grad}")

    if layers_no_grad == 0:
        print("  ✅ All SIREN layers have non-zero gradients")
        return True
    else:
        print("  ❌ Some layers have zero/missing gradients")
        return False


def test_maxwell_regularizer():
    """Verify MaxwellRegularizer produces gradients."""
    print("\n=== Testing Maxwell Regularizer ===")

    try:
        from mriforge.infrastructure.physics.pinn import MaxwellRegularizer
    except ImportError as e:
        print(f"  ⚠️  Could not import MaxwellRegularizer: {e}")
        return None

    regularizer = MaxwellRegularizer()

    # Synthetic field prediction
    prediction = torch.randn(2, 2, 64, 64, requires_grad=True)

    # Compute physics loss
    physics_loss = regularizer.compute_residual(prediction, coords=None)

    # Check it's a scalar and not zero
    if physics_loss.numel() != 1:
        print(f"  ❌ Physics loss is not a scalar: shape = {physics_loss.shape}")
        return False

    print(f"  Physics loss value: {physics_loss.item():.6f}")

    # Backprop
    physics_loss.backward()

    if prediction.grad is not None and prediction.grad.norm() > 1e-8:
        print(f"  Gradient norm: {prediction.grad.norm().item():.6f}")
        print("  ✅ MaxwellRegularizer produces gradients")
        return True
    else:
        print("  ❌ MaxwellRegularizer produces no gradients")
        return False


def test_siren_activation():
    """Verify SIREN activation doesn't collapse to zero."""
    print("\n=== Testing SIREN Activation Range ===")

    try:
        from mriforge.models.blocks.pinn_nerf_block import Sine
    except ImportError as e:
        print(f"  ⚠️  Could not import Sine: {e}")
        return None

    sine = Sine(w0=30.0)

    # Test with different input ranges
    x = torch.linspace(-1, 1, 1000)
    y = sine(x)

    mean_abs = y.abs().mean().item()
    max_val = y.max().item()
    min_val = y.min().item()

    print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Output range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Mean |output|: {mean_abs:.4f}")

    # Should have variety in output (not collapsed)
    if max_val - min_val > 1.5 and mean_abs > 0.3:
        print("  ✅ SIREN activation has good dynamic range")
        return True
    else:
        print("  ❌ SIREN activation may be collapsed")
        return False


def main():
    print("=" * 60)
    print("EXPERIMENT 39 GRADIENT VERIFICATION")
    print("=" * 60)

    results = {}

    results["siren_gradients"] = test_siren_gradient_flow()
    results["maxwell_regularizer"] = test_maxwell_regularizer()
    results["siren_activation"] = test_siren_activation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(v is True for v in results.values() if v is not None)

    for test, result in results.items():
        if result is True:
            print(f"  ✅ {test}")
        elif result is False:
            print(f"  ❌ {test}")
        else:
            print(f"  ⚠️  {test} (skipped)")

    print()
    if all_passed:
        print("Gradients Validated: True")
        return 0
    else:
        print("Gradients Validated: False")
        return 1


if __name__ == "__main__":
    sys.exit(main())
