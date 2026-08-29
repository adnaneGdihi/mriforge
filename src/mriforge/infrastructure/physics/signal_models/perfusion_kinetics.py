"""Perfusion kinetics forward models (DCE / DSC / ASL).

Pure, differentiable tensor implementations of the tracer-kinetic forward maps
that turn perfusion parameters into a measured concentration-time curve:

- **Extended Tofts** (DCE): ``C_t(t) = Ktrans int Cp(tau) e^{-(Ktrans/ve)(t-tau)} dtau + vp Cp(t)``.
- **Parker population AIF**: the standard analytic arterial input function.
- **Gamma-variate** (DSC): first-pass bolus shape.
- **SPGR signal -> concentration**: spoiled-gradient-echo signal linearisation.

These are the physics behind the ``mri_perfusion`` regime. They are analytic and
CPU-testable (known ``Ktrans/ve/vp`` -> curve -> residual ~ 0 at the true
parameters), which the physics tests pin. Time is a discrete axis in seconds.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mriforge.config.schemas.enums import Regime

from .registry import register_signal_model


def parker_population_aif(t_s: Tensor) -> Tensor:
    """Parker population arterial input function ``Cp(t)`` (mM), ``t`` in seconds.

    Two Gaussian boluses plus a sigmoid-modulated exponential washout, using
    the published Parker et al. (2006) population parameters.
    """
    a = torch.tensor([0.809, 0.330], dtype=t_s.dtype, device=t_s.device)  # mM*min
    t0 = torch.tensor([0.17046, 0.365], dtype=t_s.dtype, device=t_s.device)  # min
    sigma = torch.tensor([0.0563, 0.132], dtype=t_s.dtype, device=t_s.device)  # min
    alpha, beta, s, tau = 1.050, 0.1685, 38.078, 0.483  # mM, /min, /min, min

    t_min = t_s / 60.0
    tt = t_min.unsqueeze(-1)  # [..., 1]
    gaussians: Tensor = (
        a / (sigma * (2 * torch.pi) ** 0.5) * torch.exp(-((tt - t0) ** 2) / (2 * sigma**2))
    ).sum(dim=-1)
    washout: Tensor = alpha * torch.exp(-beta * t_min) / (1.0 + torch.exp(-s * (t_min - tau)))
    return gaussians + washout


@register_signal_model(
    name="extended_tofts",
    regime=Regime.PERFUSION,
    parameters=("ktrans", "ve", "vp"),
    signal="concentration_time_curve",
    reference="Tofts et al., J Magn Reson Imaging 10(3):223-232, 1999",
)
def extended_tofts_forward(
    t_s: Tensor,
    aif: Tensor,
    ktrans: Tensor | float,
    ve: Tensor | float,
    vp: Tensor | float = 0.0,
) -> Tensor:
    """Extended-Tofts tissue concentration ``C_t(t)`` from an AIF.

    Evaluated in closed form (no Python loop over ``T``) so it is safe to call
    every training step: ``ToftsResidualLoss`` is the primary objective of the
    perfusion strategy, and the previous per-timestep loop issued a host sync
    (``dt.item()``) and a kernel launch for each of the ``T`` steps.

    Args:
        t_s: ``[T]`` time axis (seconds), assumed uniformly sampled.
        aif: ``[T]`` plasma concentration ``Cp(t)``. Must be rank-1 — an AIF is
            arterial, one curve per acquisition.
        ktrans, ve, vp: kinetic parameters (scalars or tensors broadcastable
            against ``[..., T]``; may be per-voxel maps).

    Returns:
        ``[T]`` (or broadcast) tissue concentration curve.

    Raises:
        ValueError: if ``t_s`` is not 1-D, ``aif`` is not 1-D, or their lengths
            disagree.
    """
    if t_s.ndim != 1 or aif.shape[-1] != t_s.shape[0]:
        raise ValueError("t_s must be 1-D and aif's last dim must match len(t_s)")
    if aif.ndim != 1:
        # The AIF Toeplitz below needs a 1-D [T] curve. An AIF is arterial —
        # one curve per subject — so a per-voxel AIF is not physical. Raise
        # rather than silently broadcast into a wrong answer.
        raise ValueError(
            f"aif must be a 1-D [T] curve, got shape {tuple(aif.shape)}. "
            "An arterial input function is one curve per acquisition."
        )
    dt = (t_s[1] - t_s[0]).clamp_min(1e-6)  # a TENSOR — never .item() (CLAUDE.md #9)
    ktrans_t = torch.as_tensor(ktrans, dtype=t_s.dtype, device=t_s.device)
    ve_t = torch.as_tensor(ve, dtype=t_s.dtype, device=t_s.device).clamp_min(1e-6)
    vp_t = torch.as_tensor(vp, dtype=t_s.dtype, device=t_s.device)

    kep = ktrans_t / ve_t  # rate constant
    tau = t_s - t_s[0]  # [T], lags
    kernel = torch.exp(-kep.unsqueeze(-1) * tau)  # [..., T]

    # The convolution is evaluated in closed form rather than by looping over T.
    # Causal AIF matrix A[i, d] = Cp(t_{i-d}) for d <= i, else 0 — it is [T, T]
    # and VOXEL-INDEPENDENT, which is the whole point: materialising the kep
    # matrix exp(-kep(t_i - t_j)) instead would be [..., T, T] (~3.8 GB at
    # B=4, 256^2, T=60).
    n = t_s.shape[0]
    idx = torch.arange(n, device=t_s.device)
    lag = idx.unsqueeze(1) - idx.unsqueeze(0)  # [i, d] = i - d
    a_mat = torch.where(
        lag >= 0,
        aif[lag.clamp_min(0)],
        torch.zeros((), dtype=aif.dtype, device=aif.device),
    )  # [T, T]
    full = kernel @ a_mat.transpose(0, 1)  # [..., T] causal convolution

    # Trapezoid endpoint correction. The reference used torch.trapz, whose
    # weights are 1/2 at j=0 and j=i and 1 between, so a plain causal sum is NOT
    # equivalent. Since kernel[..., 0] == exp(0) == 1:
    #     conv[i] = dt * (sum_{j<=i} k[i-j] Cp[j] - k[i] Cp[0]/2 - Cp[i]/2)
    # which is exact at i == 0 too (both correction terms cancel the single
    # sample), so no special case is needed. This is an algebraic identity of the
    # trapezoid rule, not an approximation — the equivalence test pins it to
    # ~1e-15 in float64 against the loop it replaced.
    conv = dt * (full - 0.5 * aif[..., 0:1] * kernel - 0.5 * aif)
    return ktrans_t.unsqueeze(-1) * conv + vp_t.unsqueeze(-1) * aif


@register_signal_model(
    name="gamma_variate",
    regime=Regime.PERFUSION,
    parameters=("amplitude", "t0_s", "alpha", "beta_s"),
    signal="concentration_time_curve",
    reference="Madsen, Phys Med Biol 37(7):1597-1600, 1992",
)
def gamma_variate(
    t_s: Tensor,
    amplitude: float | Tensor = 1.0,
    t0_s: float | Tensor = 0.0,
    alpha: float | Tensor = 3.0,
    beta_s: float | Tensor = 1.5,
) -> Tensor:
    """Gamma-variate first-pass curve ``C(t) = A (t-t0)^alpha e^{-(t-t0)/beta}`` for ``t>t0``."""
    dt = t_s - torch.as_tensor(t0_s, dtype=t_s.dtype, device=t_s.device)
    dt_pos = dt.clamp_min(0.0)
    amp = torch.as_tensor(amplitude, dtype=t_s.dtype, device=t_s.device)
    a = torch.as_tensor(alpha, dtype=t_s.dtype, device=t_s.device)
    b = torch.as_tensor(beta_s, dtype=t_s.dtype, device=t_s.device).clamp_min(1e-6)
    curve = amp * dt_pos.pow(a) * torch.exp(-dt_pos / b)
    return torch.where(dt > 0, curve, torch.zeros_like(curve))


def spgr_signal_to_concentration(
    signal: Tensor,
    s0: Tensor,
    t10_s: Tensor | float,
    r1: float = 4.5,
    tr_s: float = 0.005,
    flip_angle_deg: float = 15.0,
) -> Tensor:
    """Convert SPGR signal to Gd concentration via the linear-with-R1 model.

    Uses the small-signal linearisation ``DR1 = (S - S0) / (S0 * TR)`` and
    ``C = DR1 / r1`` — adequate for the moderate-dose DCE regime and exact
    enough for the round-trip test. ``r1`` is the contrast relaxivity (/mM/s).
    """
    t10 = torch.as_tensor(t10_s, dtype=signal.dtype, device=signal.device)
    _ = (
        t10,
        tr_s,
        flip_angle_deg,
    )  # retained for signature parity / future SPGR inversion
    delta_r1 = (signal - s0) / (s0.clamp_min(1e-6) * tr_s)
    return (delta_r1 / r1).clamp_min(0.0)


__all__ = [
    "extended_tofts_forward",
    "gamma_variate",
    "parker_population_aif",
    "spgr_signal_to_concentration",
]
