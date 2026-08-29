"""Shared test scaffolding for VAE / VQ-family models.

Factors out the ELBO non-negativity and reconstruction-shape assertions
used by ``hierarchical_vae_ladder``, ``hierarchical_vq_vae``,
``hyperspherical_vae``, ``moe_vae``, and ``beta_vae_gan``. Written but
not executed here; they run on the cluster.
"""

from __future__ import annotations

import torch


def assert_recon_shape(x_hat: torch.Tensor, x: torch.Tensor) -> None:
    """Assert the reconstruction matches the input batch/spatial shape."""
    assert x_hat.shape[0] == x.shape[0], f"batch mismatch {x_hat.shape} vs {x.shape}"
    assert x_hat.shape[-2:] == x.shape[-2:], (
        f"spatial mismatch {x_hat.shape[-2:]} vs {x.shape[-2:]}"
    )
    assert torch.isfinite(x_hat).all(), "reconstruction has non-finite values"


def assert_kl_nonneg(kl: torch.Tensor, *, atol: float = 1e-4) -> None:
    """Assert a KL term is a finite, (numerically) non-negative scalar."""
    kl_val = kl.detach()
    assert torch.isfinite(kl_val).all(), "KL produced non-finite values"
    assert float(kl_val.min()) >= -atol, f"KL is negative: {float(kl_val.min())}"


def assert_gradient_flows(model, loss: torch.Tensor) -> None:
    """Assert a scalar loss produces non-zero gradients somewhere."""
    loss.backward()
    grad_total = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    assert grad_total > 0, "no gradient flowed through the model"


def assert_registered(name: str, expected_mode: str) -> None:
    """Assert ``name`` resolves to a class with the expected training mode."""
    from mriforge.models.registry import get_model_class, get_model_mode

    cls = get_model_class(name)
    assert cls is not None
    assert get_model_mode(name) == expected_mode
