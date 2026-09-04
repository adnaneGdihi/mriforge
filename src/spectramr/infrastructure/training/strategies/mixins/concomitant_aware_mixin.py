r"""Concomitant-aware reconstruction mixin (CAUR / idea 1 phase 2).

Wraps a host strategy's forward pass with the
:class:`ConcomitantPhaseOperator` so that the network is trained
against the physically-correct concomitant-phase forward model at
ULF. Implements ``ConcomitantAwareReconStrategy`` from
``TODO/integration_plan_ulf_cheap_fast_mri.md`` §1.2 as a *mixin*
rather than a full strategy class — the same primitive can be added
to any existing reconstruction strategy via cooperative MRO.

Usage
-----
.. code-block:: python

    class CAURReconStrategy(BaseTrainingStrategy, ConcomitantAwareMixin):
        def __init__(self, ...):
            super().__init__(...)
            self.configure_concomitant_correction(
                spatial_shape=(D, H, W),
                voxel_size_mm=(...),
                gradient_amplitudes_mT_per_m=(...),
                field_strength_t=0.064,
                readout_time_s=2e-3,
            )

The host's ``_compute_losses_impl`` calls
``self.apply_concomitant_correction(image)`` before the SENSE
forward, so the loss is evaluated against the correct multi-coil
k-space.

Why a mixin?
------------
Because the per-strategy custom ``__init__`` plumbing in this codebase
is already large; adding a CAUR-specific subclass for every host
strategy would force the choice between ``GANTrainingStrategy`` and
``ReconstructionTrainingStrategy``. A mixin keeps both options open.
"""

from __future__ import annotations

import logging

import torch

from spectramr.infrastructure.physics.concomitant_phase_operator import (
    ConcomitantPhaseOperator,
)

logger = logging.getLogger(__name__)


class ConcomitantAwareMixin:
    """Adds optional concomitant-phase pre-multiplication to a host strategy.

    The mixin defaults to *disabled*. Concrete strategies must call
    :py:meth:`configure_concomitant_correction` before the first
    training step. ``apply_concomitant_correction`` is then a no-op
    until configured (so the mixin is safe to inherit from a strategy
    that may or may not opt into the correction).
    """

    _concomitant_enabled: bool = False
    _concomitant_op: ConcomitantPhaseOperator | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_concomitant_correction(
        self,
        spatial_shape: tuple[int, int, int],
        voxel_size_mm: tuple[float, float, float],
        gradient_amplitudes_mT_per_m: tuple[float, float, float],
        field_strength_t: float,
        readout_time_s: float = 1e-3,
    ) -> None:
        """Activate the concomitant-phase post-processor.

        Args:
            spatial_shape: ``(D, H, W)`` of the volume.
            voxel_size_mm: per-axis voxel size in millimetres.
            gradient_amplitudes_mT_per_m: ``(G_x, G_y, G_z)`` peak
                gradient amplitudes during the readout.
            field_strength_t: static field :math:`B_0` in Tesla. The
                operator emits a ``UserWarning`` when ``B_0 > 0.5 T``
                — the correction is principally an ULF concern.
            readout_time_s: time at which to evaluate the accumulated
                phase (typically half the echo-spacing).
        """
        self._concomitant_op = ConcomitantPhaseOperator(
            spatial_shape=spatial_shape,
            voxel_size_mm=voxel_size_mm,
            gradient_amplitudes_mT_per_m=gradient_amplitudes_mT_per_m,
            field_strength_t=field_strength_t,
            readout_time_s=readout_time_s,
            enabled=True,
        )
        self._concomitant_enabled = True
        logger.info(
            "[ConcomitantAwareMixin] enabled @ B0=%.3f T, readout=%.1f µs",
            field_strength_t,
            readout_time_s * 1e6,
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def concomitant_enabled(self) -> bool:
        """``True`` iff configure_concomitant_correction was called."""
        return self._concomitant_enabled

    @property
    def concomitant_op(self) -> ConcomitantPhaseOperator:
        """Return the configured operator (raises if unset)."""
        if self._concomitant_op is None:
            raise RuntimeError(
                "ConcomitantAwareMixin not configured — "
                "call configure_concomitant_correction() first."
            )
        return self._concomitant_op

    def apply_concomitant_correction(self, image: torch.Tensor) -> torch.Tensor:
        """Apply :math:`e^{i\\phi_c}` to the input image.

        No-op when not configured — host strategies can call this
        unconditionally and opt in via configuration.

        Args:
            image: Real or complex image with the last three dims
                ``(D, H, W)``.

        Returns:
            Same-shape tensor with the concomitant-phase pre-multiply
            applied (or the original image if disabled).
        """
        if not self._concomitant_enabled or self._concomitant_op is None:
            return image
        return self._concomitant_op(image)


__all__ = ["ConcomitantAwareMixin"]
