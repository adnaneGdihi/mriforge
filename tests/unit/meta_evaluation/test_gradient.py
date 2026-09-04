"""Regression tests for metric-gradient device handling.

``spsa_gradient`` used to allocate its perturbation directions on the CPU
unconditionally (``torch.randn(...)`` with no ``device=``), so a ``hat``
tensor on CUDA crashed with::

    RuntimeError: Expected all tensors to be on the same device, but found
    at least two devices, cuda:0 and cpu!

at ``plus = (hat.float() + delta * u)``. Directions are now drawn from a
CPU generator and moved onto ``hat.device``, so the estimator runs on
whatever device ``hat`` lives on and yields the *same* directions on CPU
and GPU.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.gradient import (
    metric_gradient,
    spsa_gradient,
)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred.float() - target.float()) ** 2).mean().item())


def test_spsa_gradient_cpu_is_deterministic() -> None:
    clean = torch.zeros(1, 8, 8)
    hat = torch.ones(1, 8, 8)
    g1 = spsa_gradient(_mse, clean, hat, seed=123)
    g2 = spsa_gradient(_mse, clean, hat, seed=123)
    assert torch.equal(g1, g2)
    assert g1.device == hat.device
    assert g1.shape == hat.shape
    assert torch.isfinite(g1).all()


def test_spsa_gradient_seed_changes_estimate() -> None:
    clean = torch.zeros(1, 8, 8)
    hat = torch.ones(1, 8, 8)
    g0 = spsa_gradient(_mse, clean, hat, seed=0)
    g1 = spsa_gradient(_mse, clean, hat, seed=1)
    assert not torch.equal(g0, g1)


def test_metric_gradient_non_differentiable_routes_to_spsa() -> None:
    clean = torch.zeros(1, 8, 8)
    hat = torch.ones(1, 8, 8)
    g = metric_gradient(_mse, clean, hat, differentiable_hint=False, seed=5)
    assert g.shape == hat.shape
    assert torch.isfinite(g).all()


@pytest.mark.gpu
def test_spsa_gradient_runs_on_cuda() -> None:
    """The reported crash: hat on CUDA, perturbation on CPU."""
    device = torch.device("cuda:0")
    clean = torch.zeros(1, 8, 8, device=device)
    hat = torch.ones(1, 8, 8, device=device)
    g = spsa_gradient(_mse, clean, hat, seed=123)
    assert g.device.type == "cuda"
    assert g.shape == hat.shape
    assert torch.isfinite(g).all()


@pytest.mark.gpu
def test_spsa_gradient_directions_are_device_independent() -> None:
    """Same seed -> same directions on CPU and CUDA (up to float32 noise)."""
    clean = torch.zeros(1, 8, 8)
    hat = torch.ones(1, 8, 8)
    g_cpu = spsa_gradient(_mse, clean, hat, seed=7)
    g_cuda = spsa_gradient(_mse, clean.cuda(), hat.cuda(), seed=7)
    assert torch.allclose(g_cpu, g_cuda.cpu(), atol=1e-4, rtol=1e-4)
