r"""Concomitant-phase forward-operator wrapper (idea 1: CAUR).

Implements the operator :math:`x \mapsto e^{i\phi_c(\mathbf r, t)} x`
that pre-rotates the image into the *concomitant frame* before the
standard ``M F S`` chain. At ultra-low field the concomitant phase
:math:`\phi_c = \gamma \int_0^t B_z^{(2)} \mathrm d\tau` is non-negligible
(tens of Hz at FOV edges for 64 mT systems), so coupling it into the
forward operator is required for physically-faithful reconstruction.

Mathematics
-----------
Bernstein 1998 [1] gives the leading concomitant term

.. math::

    B_z^{(2)}(\mathbf r) =
        \tfrac{1}{2 B_0} \bigl[(G_x^2 + G_y^2) z^2
        + \tfrac{1}{4} G_z^2 (x^2 + y^2)\bigr]

which produces an effective frequency offset
:math:`\Delta f_c(\mathbf r) = \gamma B_z^{(2)} / (2\pi)`. Integrating
over the readout time :math:`t` gives a per-voxel phase
:math:`\phi_c(\mathbf r, t) = 2\pi \Delta f_c(\mathbf r) \cdot t`.

This operator multiplies the input image by :math:`e^{i\phi_c}` for a
single chosen readout time (typically :math:`t = T_{ESP}/2`, the
echo-spacing midpoint) and is the building block consumed by the
``caur`` strategy.

Reuse
-----
Wraps :func:`spectramr.infrastructure.physics.maxwell_concomitant.maxwell_concomitant_field_hz`
— do *not* re-derive Bernstein's formula here.

References
----------
[1] Bernstein et al., *Magn. Reson. Med.* 39:300-308 (1998).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.maxwell_concomitant import (
    maxwell_concomitant_field_hz,
)


class ConcomitantPhaseOperator(nn.Module):
    r"""Apply :math:`e^{i\phi_c(\mathbf r, t)}` to a complex image.

    Args:
        spatial_shape: ``(D, H, W)`` of the volume.
        voxel_size_mm: per-axis voxel size in millimetres.
        gradient_amplitudes_mT_per_m: ``(G_x, G_y, G_z)`` peak amplitudes
            of the imaging gradients during the readout.
        field_strength_t: static field :math:`B_0` in Tesla.
        readout_time_s: time at which to evaluate the accumulated phase
            (typically half the echo-spacing).
        enabled: master switch — when ``False``, the operator is the
            identity. Used for ablation experiments and the v6.1
            opt-in default.

    The Maxwell field map is computed once at construction time and
    registered as a buffer so it migrates with ``.to(device)``.
    """

    def __init__(
        self,
        spatial_shape: tuple[int, int, int],
        voxel_size_mm: tuple[float, float, float],
        gradient_amplitudes_mT_per_m: tuple[float, float, float],
        field_strength_t: float,
        readout_time_s: float = 1e-3,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if readout_time_s < 0:
            raise ValueError(f"readout_time_s must be ≥ 0; got {readout_time_s}")
        if field_strength_t > 0.5:
            # Concomitant correction is principally an ULF concern.
            # We don't *forbid* HF use (it just adds a small bias),
            # but warn the caller — Tier-1 audit will surface it loudly.
            import warnings

            warnings.warn(
                f"ConcomitantPhaseOperator activated at B0={field_strength_t} T; "
                "the concomitant correction is most relevant below 0.5 T.",
                UserWarning,
                stacklevel=2,
            )
        self.enabled = enabled
        self.readout_time_s = float(readout_time_s)

        # Compute the Hz-frequency-offset map up front — independent of
        # the runtime image so it's a pure model buffer.
        freq_hz = maxwell_concomitant_field_hz(
            spatial_shape=spatial_shape,
            voxel_size_mm=voxel_size_mm,
            gradient_amplitudes_mT_per_m=gradient_amplitudes_mT_per_m,
            field_strength_t=field_strength_t,
        )
        # Per-voxel accumulated phase φ = 2π · f · t  (radians).
        phase = 2.0 * math.pi * freq_hz * self.readout_time_s
        self.register_buffer("phase", phase)
        self.register_buffer("phasor", torch.polar(torch.ones_like(phase), phase))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`e^{i\phi_c}` to ``image``.

        Args:
            image: Complex tensor with the last three dims ``(D, H, W)``.
                Real-valued images are first cast to complex.

        Returns:
            Same-shape complex tensor.
        """
        if not self.enabled:
            return image
        if not torch.is_complex(image):
            image = torch.complex(image, torch.zeros_like(image))
        # Broadcast the phasor over leading batch / channel dims.
        return image * self.phasor

    def get_frequency_map_hz(self) -> torch.Tensor:
        r"""Return the underlying Hz-domain frequency-offset map (read-only)."""
        return self.phase / (2.0 * math.pi * max(self.readout_time_s, 1e-12))


__all__ = ["ConcomitantPhaseOperator"]
