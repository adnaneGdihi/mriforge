r"""Phase-residual producer transform (inverse-Bloch phase strategy, idea).

Emits a ``phase_residual`` image: the *non-smooth* component of the measured
phase, i.e. the measured phase with its slowly-varying background removed. This
is the quantity the ``InverseBlochPhaseStrategy`` phase-smoothness regulariser
operates on (a physical prior that the residual phase, once the smooth
B0 / concomitant background is removed, should itself be spatially smooth).

Method (phasor-based, wrap-safe)
--------------------------------
Working on the unit phasor :math:`u = e^{j\,\angle z}` avoids phase-wrap
artefacts that a direct low-pass of :math:`\angle z \in (-\pi, \pi]` would
introduce:

.. math::

    \bar u = \mathrm{LowPass}(u), \quad
    \phi_\text{bg} = \angle \bar u, \quad
    r = \angle\!\big(e^{j(\angle z - \phi_\text{bg})}\big) \in (-\pi, \pi].

``LowPass`` is a box filter of side ``kernel_size`` over the trailing two
(spatial) axes.

NOTE: this is a *background-removed phase residual*, not a full Bloch-SPGR
forward-model residual. The latter (forward-simulate the SPGR phase from tissue
+ sequence parameters and subtract it) needs tissue maps + sequence metadata and
remains roadmap work; see ``docs/strategy_audit_design_triage_2026_06.rst``.
The transform is deliberately named for what it computes.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torchio as tio
from torch.nn.functional import avg_pool2d

from mriforge.data.transforms.registry import register_transform

__all__ = ["PhaseResidualTransform"]


@register_transform("phase_residual", produces=("phase_residual",))
class PhaseResidualTransform(tio.transforms.Transform):
    """Emit a ``phase_residual`` image (background-removed measured phase).

    Args:
        kernel_size: side of the box low-pass over the spatial axes (odd, >0).
        keys: subject keys to read the (complex) measurement from.
        output_key: name of the emitted residual image. With multiple ``keys``
            the per-key residuals are emitted as ``<output_key>_<key>``.
    """

    def __init__(
        self,
        kernel_size: int = 9,
        keys: Sequence[str] = ("input",),
        output_key: str = "phase_residual",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer; got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.keys = tuple(keys)
        self.output_key = output_key

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for key in self.keys:
            if key not in subject:
                continue
            image = subject[key]
            data = image.data
            if data.dim() < 3:
                continue
            residual = self._phase_residual(data)
            out_img = copy.deepcopy(image)
            out_img.set_data(residual)
            name = self.output_key if len(self.keys) == 1 else f"{self.output_key}_{key}"
            subject.add_image(out_img, name)
        return subject

    def _phase_residual(self, data: torch.Tensor) -> torch.Tensor:
        z = self._to_complex(data)
        phase = torch.angle(z)
        unit = torch.exp(1j * phase)  # unit phasor (wrap-safe smoothing)
        k, pad = self.kernel_size, self.kernel_size // 2
        re = avg_pool2d(unit.real.unsqueeze(0), k, stride=1, padding=pad)[0]
        im = avg_pool2d(unit.imag.unsqueeze(0), k, stride=1, padding=pad)[0]
        background = torch.atan2(im, re)
        # Wrap (phase - background) into (-pi, pi].
        return torch.angle(torch.exp(1j * (phase - background))).to(
            data.real.dtype if torch.is_complex(data) else data.dtype
        )

    @staticmethod
    def _to_complex(data: torch.Tensor) -> torch.Tensor:
        """Coerce to complex: genuine-complex pass-through; even real channels
        are paired (real, imag); otherwise treat as zero-phase magnitude."""
        if torch.is_complex(data):
            return data
        if data.shape[0] % 2 == 0:
            half = data.shape[0] // 2
            return torch.complex(data[:half], data[half:])
        return torch.complex(data, torch.zeros_like(data))
