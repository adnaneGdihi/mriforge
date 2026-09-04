"""Sinusoidal timestep embedding — canonical, parameter-free.

Why this module exists
----------------------
A diffusion backbone has to tell neighbouring timesteps apart. Two mistakes
recur in this tree and both are silent:

1. **Encoding ``t`` linearly.** ``nn.Linear(1, d)`` applied to the raw scalar is
   a rank-1 map of ``t``: every timestep code lies on a single line in embedding
   space, so the network can represent "large t" but not "which t". A sinusoidal
   basis spreads the codes over ``d`` dimensions at geometrically spaced
   frequencies, which is what lets a model condition on a specific step.

2. **Not scaling ``t`` before the sinusoid.** With ``t`` in ``[0, T)`` and a
   fixed 10000-base frequency ladder, a large ``T`` pushes every code into the
   high-frequency regime and adjacent steps alias onto each other; a small ``T``
   collapses them all near zero. Dividing by ``max_timesteps`` puts ``t`` in
   ``[0, 1]`` where the ladder is well conditioned. This is the failure the
   ``experiment_11_kspace_cold_diffusion`` YAML documents for ``timesteps: 28``:
   "if this is 1000 and the diffusion schedule only uses [0, 27] the sinusoidal
   embedding collapses all 28 codes into a tiny region near zero and the model
   can't distinguish neighbouring timesteps."

Scope note
----------
Ten near-identical private copies of this function exist across
``models/generators/``. This module is deliberately introduced for NEW consumers
only — migrating the existing copies would need each one proven bit-identical
first, or it silently changes trained-model behaviour. Consolidating them is
tracked separately.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["sinusoidal_timestep_embedding"]


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    *,
    max_timesteps: float | None = None,
) -> torch.Tensor:
    """Map a batch of scalar timesteps onto a sinusoidal basis.

    Args:
        timesteps: ``[B]`` (or ``[B, 1]``) scalar step indices. Floats are fine —
            a continuous acceleration factor is a legitimate conditioning signal
            here, not only an integer step.
        dim: Width of the returned embedding. Odd widths are right-padded.
        max_timesteps: Divide by this before encoding, so the input lands in
            ``[0, 1]``. Pass the schedule's horizon (``T``). ``None`` leaves the
            values untouched — only correct when they are already normalised.

    Returns:
        ``[B, dim]``, ``[sin(...) ‖ cos(...)]``.

    Raises:
        ValueError: If ``dim < 2`` (``half_dim - 1`` would divide by zero) or
            ``max_timesteps`` is non-positive. Both are silent NaN sources
            otherwise.
    """
    if dim < 2:
        raise ValueError(f"dim must be >= 2 for a sin/cos split, got {dim}")

    t = timesteps.float()
    if t.ndim > 1:
        t = t.reshape(t.shape[0], -1)[:, 0]

    if max_timesteps is not None:
        if max_timesteps <= 0:
            raise ValueError(f"max_timesteps must be > 0, got {max_timesteps}")
        t = t / float(max_timesteps)

    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=t.device, dtype=t.dtype)
        / max(half_dim - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
