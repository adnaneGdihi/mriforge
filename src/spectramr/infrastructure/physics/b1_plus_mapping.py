"""B1+ (transmit field) mapping utilities.

Whereas ``coil_sensitivity.py`` estimates B1- (receive sensitivity), this
module provides estimators for B1+ — the transmit RF field — which is
needed for flip-angle correction at higher field strengths and for
quantitative T1/T2 mapping.

Two estimators:

- ``DoubleAngleB1Plus``: classic Stollberger & Wach (1996) two-flip-angle
  formula α = arccos(|S2| / 2|S1|), expects a pair of acquired magnitudes
  at flip angles α and 2α.
- ``BlochSiegertB1Plus`` (light variant): phase-difference flip angle
  estimate from a Bloch-Siegert pulse. Implementation follows the linear
  approximation only — not a substitute for full BSI mapping but useful
  as a differentiable component in physics-informed networks.

Both are differentiable; both return a per-pixel B1+ map (relative to the
nominal flip angle) of shape ``[B, 1, H, W]``.

References:

- Stollberger, R. and Wach, P. (1996). "Imaging of the active B1 field
  in vivo." MRM 35:246-251.
- Sacolick, L. et al. (2010). "B1 mapping by Bloch-Siegert shift." MRM
  63:1315-1322.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DoubleAngleB1Plus(nn.Module):
    """Two-flip-angle B1+ estimator.

    Args:
        nominal_flip_deg: nominal flip angle of the first acquisition (degrees).
        eps: numerical stabiliser for the arccos argument.

    Forward inputs:
        s_alpha: magnitude image at flip angle α  ``[B, 1, H, W]``.
        s_2alpha: magnitude image at flip angle 2α  ``[B, 1, H, W]``.

    Returns:
        Relative B1+ map ``[B, 1, H, W]`` (1.0 means correct flip).
    """

    def __init__(self, nominal_flip_deg: float = 60.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.nominal_flip_rad = math.radians(float(nominal_flip_deg))
        self.eps = float(eps)

    def forward(self, s_alpha: torch.Tensor, s_2alpha: torch.Tensor) -> torch.Tensor:
        if s_alpha.shape != s_2alpha.shape:
            raise ValueError(
                f"shape mismatch: s_alpha={tuple(s_alpha.shape)}, s_2alpha={tuple(s_2alpha.shape)}"
            )
        if torch.is_complex(s_alpha):
            s_alpha = s_alpha.abs()
        if torch.is_complex(s_2alpha):
            s_2alpha = s_2alpha.abs()
        # arccos argument; clamp to guard against noise > 1
        ratio = (s_2alpha / (2.0 * s_alpha + self.eps)).clamp(-1.0 + self.eps, 1.0 - self.eps)
        alpha_actual = torch.arccos(ratio)
        # Relative B1+ = actual / nominal flip
        return alpha_actual / self.nominal_flip_rad


class BlochSiegertB1Plus(nn.Module):
    """Linearised Bloch-Siegert B1+ estimator from a phase-difference image.

    Args:
        kbs: Bloch-Siegert constant K_BS in rad/G^2 (sequence-dependent).
            Defaults to 74.0 rad/G^2 (typical 4 ms Fermi pulse).
        eps: stabiliser for the absolute-value taken under the square root.

    Forward inputs:
        phase_diff: phase-difference map (rad)  ``[B, 1, H, W]`` real-valued.

    Returns:
        B1+ amplitude in Gauss  ``[B, 1, H, W]``.
    """

    def __init__(self, kbs: float = 74.0, eps: float = 1e-9) -> None:
        super().__init__()
        if kbs <= 0:
            raise ValueError(f"kbs must be > 0, got {kbs}")
        self.kbs = float(kbs)
        self.eps = float(eps)

    def forward(self, phase_diff: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.abs(phase_diff) / self.kbs + self.eps)


def apply_b1_correction(
    image: torch.Tensor,
    b1_plus_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Inverse-correct a magnitude image for B1+ inhomogeneity.

    For a small-flip-angle approximation, signal scales linearly with
    sin(α_actual), so dividing by the relative B1+ map removes the
    transmit-field bias. Larger flip angles require a sequence-specific
    Bloch lookup; this helper is the linear-regime baseline.

    Args:
        image: magnitude image ``[B, C, H, W]``.
        b1_plus_map: relative B1+ map ``[B, 1, H, W]``.
        eps: stabiliser for the division.
    """
    if torch.is_complex(image):
        image = image.abs()
    if b1_plus_map.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"spatial shape mismatch: image={tuple(image.shape)}, b1+={tuple(b1_plus_map.shape)}"
        )
    return image / (b1_plus_map + eps)


__all__ = [
    "BlochSiegertB1Plus",
    "DoubleAngleB1Plus",
    "apply_b1_correction",
]
