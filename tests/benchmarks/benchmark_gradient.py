"""Benchmarks for Gradient Computation (Phase 5)."""

import pytest
import torch
import torch.nn as nn
from spectramr.models.generators.unet_2d import UNet2D


@pytest.fixture
def model_and_data(benchmark_config):
    """Setup model and data for gradient benchmarking."""
    device = benchmark_config["device"]
    B, C, H, W = benchmark_config["batch_size"], 1, 128, 128

    model = UNet2D(in_channels=C, out_channels=C, features=[32, 64, 128]).to(device)
    input_tensor = torch.randn(B, C, H, W, device=device)
    target_tensor = torch.randn(B, C, H, W, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    return model, input_tensor, target_tensor, optimizer, criterion


def benchmark_gradient_computation(benchmark, model_and_data):
    """Measure backward pass time."""
    model, input_tensor, target_tensor, optimizer, criterion = model_and_data

    def _step():
        optimizer.zero_grad(set_to_none=True)
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        return loss

    # Warmup
    _step()

    benchmark(_step)


def benchmark_optimizer_step(benchmark, model_and_data):
    """Measure optimizer step time."""
    model, input_tensor, target_tensor, optimizer, criterion = model_and_data

    # Pre-compute gradients once
    optimizer.zero_grad(set_to_none=True)
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()

    def _step():
        optimizer.step()
        # Note: In real training, we'd zero_grad after, but for benchmark we want to measure step() overhead.
        # However, step() modifies params. We might need to keep gradients?
        # Repeating step() on same gradients is valid for measuring checking overheads.

    benchmark(_step)
