r"""DP-SGD primitives: per-sample gradient clipping + Gaussian noise.

Implements the two ingredients of the DP-SGD update from
Abadi et al. 2016 [Algorithm 1]:

1. *Per-sample gradient clipping.* Each example's gradient is
   independently rescaled so its L2 norm is at most ``max_grad_norm``.
   This bounds the *sensitivity* of the aggregate gradient.

2. *Gaussian noise injection.* The mean of the clipped per-sample
   gradients is corrupted with isotropic Gaussian noise of standard
   deviation ``noise_multiplier · max_grad_norm / batch_size``.

Composing those two operations gives a Gaussian mechanism whose RDP
the :class:`RDPAccountant` can track.

Note
----
For production-grade DP-SGD use ``opacus.PrivacyEngine``; that module
hooks into PyTorch's autograd to compute true per-sample gradients
without re-running the backward pass. The helpers here operate on
*pre-computed* per-sample gradients (e.g. as returned by
``torch.func.vmap`` or by manually looping the backward) and are
useful for testing, custom training loops, or environments without
the optional ``[dp]`` extra installed.
"""

from __future__ import annotations

import torch


def clip_per_sample_grad(
    per_sample_grads: torch.Tensor,
    max_grad_norm: float,
) -> torch.Tensor:
    """Rescale each per-sample gradient so its L2 norm ≤ ``max_grad_norm``.

    Args:
        per_sample_grads: Tensor of shape ``[B, *param_shape]`` where
            ``B`` is the batch size and ``per_sample_grads[i]`` is the
            gradient for example ``i``.
        max_grad_norm: L2-norm cap (must be > 0).

    Returns:
        Tensor of the same shape, with each row's L2 norm ≤
        ``max_grad_norm``.

    Raises:
        ValueError: when ``max_grad_norm <= 0`` or ``B == 0``.
    """
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be > 0; got {max_grad_norm}")
    if per_sample_grads.shape[0] == 0:
        raise ValueError("per_sample_grads has empty batch dim")

    flat = per_sample_grads.reshape(per_sample_grads.shape[0], -1)
    per_sample_norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (max_grad_norm / per_sample_norm).clamp(max=1.0)  # only shrink
    return per_sample_grads * scale.view(-1, *([1] * (per_sample_grads.dim() - 1)))


def add_gaussian_noise(
    aggregate_grad: torch.Tensor,
    noise_multiplier: float,
    max_grad_norm: float,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add isotropic Gaussian noise to the *averaged* clipped gradient.

    Noise std = ``noise_multiplier * max_grad_norm / batch_size``.

    Args:
        aggregate_grad: Mean (or sum) of clipped per-sample gradients.
        noise_multiplier: ``σ`` in ``N(0, (σ · max_grad_norm)²)``. Must be ≥ 0.
        max_grad_norm: The clip threshold used upstream (must be > 0).
        batch_size: Number of examples in the aggregate (must be ≥ 1).
        generator: Optional ``torch.Generator`` for deterministic tests.

    Returns:
        Noised gradient — same shape as ``aggregate_grad``.

    Raises:
        ValueError: invalid arguments.
    """
    if noise_multiplier < 0:
        raise ValueError(f"noise_multiplier must be ≥ 0; got {noise_multiplier}")
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be > 0; got {max_grad_norm}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1; got {batch_size}")

    std = noise_multiplier * max_grad_norm / batch_size
    if std == 0:
        return aggregate_grad
    noise = torch.randn(
        aggregate_grad.shape,
        device=aggregate_grad.device,
        dtype=aggregate_grad.dtype,
        generator=generator,
    )
    return aggregate_grad + std * noise


def compose_dp_sgd_update(
    per_sample_grads: torch.Tensor,
    max_grad_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One DP-SGD update step: clip per-sample, average, add noise.

    Returns the noised mean gradient — what the optimiser then steps with.
    """
    clipped = clip_per_sample_grad(per_sample_grads, max_grad_norm)
    batch_size = clipped.shape[0]
    averaged = clipped.mean(dim=0)
    return add_gaussian_noise(
        averaged,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        generator=generator,
    )


__all__ = [
    "add_gaussian_noise",
    "clip_per_sample_grad",
    "compose_dp_sgd_update",
]
