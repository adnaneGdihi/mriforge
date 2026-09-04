r"""Beltrami faithfulness threshold (A-6.2 *BFT*).

A geometric operation on an image — registration, distortion correction,
cross-field synthesis — induces a deformation :math:`\varphi(z)=z+d(z)` (complex
coordinate :math:`z=x+iy`, displacement :math:`d=d_x+i d_y`). Its **Beltrami
coefficient**

.. math::

    \mu = \frac{\partial_{\bar z} d}{1 + \partial_z d},\qquad
    \partial_z=\tfrac12(\partial_x-i\partial_y),\ \partial_{\bar z}=\tfrac12(\partial_x+i\partial_y),

measures the local quasiconformal distortion: :math:`\mu=0` is conformal
(angle-preserving — only local scale/rotation), and the map is a **homeomorphism
(bijective, no folding) iff** :math:`|\mu|<1` everywhere.

BFT turns this into a **faithfulness certificate** against a reader-defined
tolerance :math:`\delta\in(0,1)`: a deformation is *faithful* when its distortion
stays within the reader's acceptable bound, :math:`\max|\mu|\le\delta`. Because
:math:`\delta<1`, faithful :math:`\Rightarrow` bijective; and a synthesis that
*folds* anatomy (:math:`|\mu|\ge 1` somewhere) is flagged exactly. The reports are
the max / mean distortion, the faithful pixel fraction
:math:`\operatorname{mean}(|\mu|\le\delta)`, and the boolean faithful / bijective
verdicts.

This generalises :func:`spectramr.infrastructure.physics.epi_forward.beltrami_from_b0_and_gnl`
(which is B0/GNL-specific and clamps :math:`|\mu|<k_{\max}`): BFT takes an
**arbitrary** displacement field and does **not** clamp, so it can see folds. It is
an analytical certificate primitive (operating on any displacement field), not a
training paradigm — the threshold :math:`\delta` is the one human-calibrated knob.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _as_dx_dy(displacement: Tensor) -> tuple[Tensor, Tensor]:
    """Coerce a displacement to real ``(d_x, d_y)`` each ``[B, 1, H, W]``.

    Accepts ``[B, 2, H, W]`` (stacked d_x, d_y) or complex ``[B, H, W]`` /
    ``[B, 1, H, W]`` (``d = d_x + i d_y``).
    """
    d = displacement
    if d.is_complex():
        if d.dim() == 3:
            d = d[:, None]
        if d.dim() != 4 or d.shape[1] != 1:
            raise ValueError(
                f"complex displacement must be [B,H,W] or [B,1,H,W], got {tuple(d.shape)}"
            )
        return d.real, d.imag
    if d.dim() != 4 or d.shape[1] != 2:
        raise ValueError(f"real displacement must be [B,2,H,W] (d_x,d_y), got {tuple(d.shape)}")
    return d[:, 0:1], d[:, 1:2]


def beltrami_from_displacement(displacement: Tensor) -> Tensor:
    r"""Beltrami coefficient :math:`\mu` of a general displacement field (no clamp).

    Args:
        displacement: ``[B, 2, H, W]`` (d_x, d_y) real, or complex ``[B, H, W]`` /
            ``[B, 1, H, W]`` (``d = d_x + i d_y``), in pixel units.

    Returns:
        complex ``[B, H, W]`` Beltrami coefficient (un-clamped, so :math:`|\mu|`
        can reach / exceed 1 where the deformation folds).
    """
    d_x, d_y = _as_dx_dy(displacement)

    def _dpx(f: Tensor) -> Tensor:  # central difference along W (x)
        g = (f[..., :, 2:] - f[..., :, :-2]) * 0.5
        return torch.nn.functional.pad(g, (1, 1, 0, 0), mode="replicate")

    def _dpy(f: Tensor) -> Tensor:  # central difference along H (y)
        g = (f[..., 2:, :] - f[..., :-2, :]) * 0.5
        return torch.nn.functional.pad(g, (0, 0, 1, 1), mode="replicate")

    dxd = torch.complex(_dpx(d_x), _dpx(d_y)).squeeze(1)  # ∂_x d, [B,H,W]
    dyd = torch.complex(_dpy(d_x), _dpy(d_y)).squeeze(1)  # ∂_y d
    dz = 0.5 * (dxd - 1j * dyd)
    dzbar = 0.5 * (dxd + 1j * dyd)
    # Denominator 1 + ∂_z d = ∂_z φ vanishes exactly at fold points; guard it so a
    # fold yields a large-but-finite |μ| (the module's intended signal) instead of
    # Inf/NaN that then poisons .abs().max()/.mean().
    denom = 1.0 + dz
    denom = torch.where(denom.abs() < 1e-8, denom + 1e-8, denom)
    return dzbar / denom


def beltrami_faithfulness(displacement: Tensor, delta: float) -> float:
    r"""Faithful pixel fraction :math:`\operatorname{mean}(|\mu|\le\delta)` in ``[0, 1]``."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1) (a faithfulness tolerance), got {delta}")
    mu = beltrami_from_displacement(displacement)
    return float((mu.abs() <= delta).float().mean())


def is_faithful(displacement: Tensor, delta: float) -> bool:
    r"""True iff the deformation is faithful everywhere: :math:`\max|\mu|\le\delta`."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return float(beltrami_from_displacement(displacement).abs().max()) <= delta


def is_bijective(displacement: Tensor, eps: float = 1e-6) -> bool:
    r"""True iff the deformation is a homeomorphism (no folding): :math:`\max|\mu|<1`."""
    return float(beltrami_from_displacement(displacement).abs().max()) < 1.0 - eps


class BeltramiFaithfulnessThreshold:
    """Beltrami faithfulness certificate against a reader-defined threshold (A-6.2 BFT).

    Args:
        delta: faithfulness tolerance in ``(0, 1)`` — the one human-calibrated knob
            (the maximum quasiconformal distortion a reader accepts).
    """

    def __init__(self, delta: float = 0.5) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.delta = float(delta)

    def certify(self, displacement: Tensor) -> dict[str, Any]:
        """Return ``{max_distortion, mean_distortion, faithful_fraction, faithful, bijective}``."""
        mu_abs = beltrami_from_displacement(displacement).abs()
        max_d = float(mu_abs.max())
        return {
            "max_distortion": max_d,
            "mean_distortion": float(mu_abs.mean()),
            "faithful_fraction": float((mu_abs <= self.delta).float().mean()),
            "faithful": max_d <= self.delta,
            "bijective": max_d < 1.0,
        }


__all__ = [
    "BeltramiFaithfulnessThreshold",
    "beltrami_faithfulness",
    "beltrami_from_displacement",
    "is_bijective",
    "is_faithful",
]
