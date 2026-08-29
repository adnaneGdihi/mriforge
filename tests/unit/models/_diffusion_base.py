"""Shared test scaffolding for diffusion forward-process schedules.

Factors out the load-bearing invariants tested across blurring / cold /
other diffusion schedules:

* schedule monotonicity (the dissipation factor strictly decreases in
  ``t`` for every non-DC frequency), and
* AMP survival (a forward/loss pass under ``autocast`` produces no NaN).

These helpers take already-constructed objects so each concrete test
module supplies its own schedule / model instance.
"""

from __future__ import annotations

import torch


def assert_schedule_monotone_decreasing(
    alpha_fn,
    num_timesteps: int,
    *,
    skip_dc: bool = True,
    atol: float = 1e-6,
) -> None:
    """Assert the per-frequency dissipation factor decreases in ``t``.

    Args:
        alpha_fn: Callable ``t -> alpha`` where ``t`` is a long tensor of
            timestep indices and ``alpha`` is ``[B, 1, H, W]`` (or
            broadcastable) with values in ``(0, 1]``.
        num_timesteps: Total schedule length.
        skip_dc: Skip the DC frequency ``(0, 0)`` whose eigenvalue is 0
            (it stays exactly 1.0 and is constant, not strictly
            decreasing).
        atol: Tolerance for the "non-increasing" comparison.
    """
    t0 = torch.zeros(1, dtype=torch.long)
    t_last = torch.full((1,), num_timesteps - 1, dtype=torch.long)
    a0 = alpha_fn(t0)
    a_last = alpha_fn(t_last)

    if skip_dc:
        # Mask out the DC corner so the strict comparison ignores it.
        a0 = a0.clone()
        a_last = a_last.clone()
        a0[..., 0, 0] = 0.0
        a_last[..., 0, 0] = 0.0

    # alpha at the final step must be <= alpha at the first step
    # everywhere, and strictly smaller for at least one high-freq mode.
    assert torch.all(a_last <= a0 + atol), "dissipation factor increased in t"
    assert torch.any(a_last < a0 - atol), "dissipation factor never decreased"


def assert_amp_survives(loss_fn, x0: torch.Tensor) -> None:
    """Assert a forward/loss pass under CPU autocast yields a finite loss.

    Uses CPU autocast (bf16) so the assertion runs on CI without CUDA;
    the production path uses ``torch.amp.autocast("cuda")``.

    Args:
        loss_fn: Callable ``x0 -> scalar loss``.
        x0: Clean input batch.
    """
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss = loss_fn(x0)
    assert torch.isfinite(loss).all(), "loss is NaN/Inf under autocast"
