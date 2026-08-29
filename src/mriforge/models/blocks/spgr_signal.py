"""SPGR / spin-echo signal models — small differentiable Bloch decoders.

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §3 (bloch_bottleneck) /
       §4 (cycle_bloch_digital_twin).

Provides closed-form magnitude expressions for the two pulse families
used by the v6.3 ``bloch_bottleneck`` and
``cycle_bloch_digital_twin`` strategies:

- :func:`spgr_signal` — SPGR steady state ``M0 (1−E1) sin α / (1 − E1 cos α)``
  with ``E1 = exp(-TR/T1)`` and an optional ``T2*`` modulation
  ``exp(-TE/T2*)``.
- :func:`spin_echo_signal` — ``M0 (1 − exp(-TR/T1)) exp(-TE/T2)``.

Both functions are autograd-traced w.r.t. the parameter maps so the
encoder bottleneck strategy can backprop physical-parameter losses
through them.
"""

from __future__ import annotations

import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return num / (den + eps)


def spgr_signal(
    M0: torch.Tensor,
    T1: torch.Tensor,
    T2_star: torch.Tensor | None,
    TR: float | torch.Tensor,
    TE: float | torch.Tensor,
    alpha_rad: float | torch.Tensor,
) -> torch.Tensor:
    """SPGR steady-state magnitude.

    ``S = M0 · (1 − E1) · sin α / (1 − E1 cos α) · exp(−TE/T2*)``
    """
    if not torch.is_tensor(TR):
        TR = M0.new_tensor(TR)
    if not torch.is_tensor(TE):
        TE = M0.new_tensor(TE)
    if not torch.is_tensor(alpha_rad):
        alpha_rad = M0.new_tensor(alpha_rad)

    E1 = torch.exp(-_safe_div(TR, T1.clamp_min(1e-3)))
    sin_a = torch.sin(alpha_rad)
    cos_a = torch.cos(alpha_rad)
    base = M0 * (1.0 - E1) * sin_a / ((1.0 - E1 * cos_a).clamp_min(1e-6))
    if T2_star is not None:
        base = base * torch.exp(-_safe_div(TE, T2_star.clamp_min(1e-3)))
    return base


def spin_echo_signal(
    M0: torch.Tensor,
    T1: torch.Tensor,
    T2: torch.Tensor,
    TR: float | torch.Tensor,
    TE: float | torch.Tensor,
) -> torch.Tensor:
    """Standard spin-echo magnitude: ``M0 (1 − exp(-TR/T1)) exp(-TE/T2)``."""
    if not torch.is_tensor(TR):
        TR = M0.new_tensor(TR)
    if not torch.is_tensor(TE):
        TE = M0.new_tensor(TE)
    return (
        M0
        * (1.0 - torch.exp(-_safe_div(TR, T1.clamp_min(1e-3))))
        * torch.exp(-_safe_div(TE, T2.clamp_min(1e-3)))
    )


__all__ = ["spgr_signal", "spin_echo_signal"]
