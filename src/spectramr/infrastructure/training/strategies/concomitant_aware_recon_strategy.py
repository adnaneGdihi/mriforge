"""Concomitant-Aware Reconstruction Strategy (CAUR — idea 1).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §1.

Adds Maxwell concomitant-field phase correction to the reconstruction
forward operator at low B0.  The correction is applied through the
``ConcomitantPhaseOperator`` (already shipped) and an optional
``concomitant_phase_residual`` regulariser pulls the reconstructed
phase toward the predicted concomitant phase.

Audit fix: previously this class silently fell back to vanilla
reconstruction when ``physics.concomitant.enabled`` was False — a
CLAUDE.md #9 silent-fallback violation.  The strategy now **fails
loud** in that case so the user explicitly switches to
``ReconstructionTrainingStrategy`` if they want the un-corrected path.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .mixins.concomitant_aware_mixin import ConcomitantAwareMixin
from .mixins.utils import pick_present
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class ConcomitantAwareReconStrategy(ConcomitantAwareMixin, ReconstructionTrainingStrategy):
    """Reconstruction with Maxwell concomitant-field phase correction.

    Audit fix (2026-05-31): ``__init__`` validated ``physics.concomitant.enabled``
    but never called :py:meth:`configure_concomitant_correction`, so the mixin
    stayed disabled (``_concomitant_enabled`` False) and the correction was a
    silent no-op. The operator's spatial shape depends on the batch resolution,
    which is unknown at construction time, so we configure **lazily** on the
    first ``_compute_losses_impl`` call (guarded by ``_concomitant_configured``)
    and then pre-rotate the network input into the concomitant frame before the
    parent reconstruction loss is computed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = getattr(self.config.physics, "concomitant", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            raise ValueError(
                "ConcomitantAwareReconStrategy requires "
                "physics.concomitant.enabled = true.  If you want the un-corrected "
                "path, switch training.strategy_class to "
                "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy "
                "instead — silent downgrade to vanilla reconstruction is forbidden "
                "(CLAUDE.md #9)."
            )
        # Lazy-configure guard (the operator's spatial_shape needs a batch).
        self._concomitant_configured: bool = False
        logger.info(
            "ConcomitantAwareReconStrategy: correction ENABLED at B0=%s T",
            getattr(self.config.physics, "field_strength", "?"),
        )

    # ------------------------------------------------------------------
    # Lazy configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_image(batch: Any) -> torch.Tensor | None:
        """Resolve the input image tensor from a tensor- or dict-batch."""
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, dict):
            return pick_present(
                batch.get("lr"),
                batch.get("input"),
                batch.get("image"),
                batch.get("kspace"),
                batch.get("hr"),
                batch.get("target"),
            )
        return None

    def _ensure_concomitant_configured(self, image: torch.Tensor) -> None:
        """Configure the concomitant operator once, from the batch spatial shape."""
        if self._concomitant_configured:
            return
        # 2-D recon image is [B, C, H, W]; the operator broadcasts an (D, H, W)
        # phasor, so a 2-D slice maps to spatial_shape=(1, H, W).
        h, w = int(image.shape[-2]), int(image.shape[-1])
        cfg = self.config.physics.concomitant
        # max_gradient_T_per_m -> mT/m for the operator (x 1e3).
        g_mt_per_m = float(getattr(cfg, "max_gradient_T_per_m", 0.040)) * 1e3
        # On a single central slice (D=1 -> z=0) the Bernstein in-plane term
        # (Gx^2+Gy^2)*z^2 vanishes, so an (g, g, 0) configuration would make the
        # concomitant phasor identically 1 (an inert facade — the very silent
        # downgrade this strategy raises to forbid). The surviving central-slice
        # term is 1/4 * Gz^2 * (x^2+y^2) / (2 B0) = Gz^2 (x^2+y^2)/(8 B0), the
        # standard 2-D model used by digital_twin.simulate_concomitant_phase, so
        # the imaging gradient is placed on the slice-select (Gz) axis.
        self.configure_concomitant_correction(
            spatial_shape=(1, h, w),
            voxel_size_mm=(1.0, 1.0, 1.0),
            gradient_amplitudes_mT_per_m=(0.0, 0.0, g_mt_per_m),
            field_strength_t=float(self.config.physics.field_strength),
            readout_time_s=float(getattr(cfg, "readout_time_s", 1e-3)),
        )
        # Strategy is NOT an nn.Module — store the operator's buffers on-device
        # via .to and register any learnable params on the generator optimizer
        # (the operator is buffer-only today, so the param-group add is a no-op,
        # but we follow the canonical registration idiom).
        op = self._concomitant_op
        if op is not None:
            op.to(self.device)
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                existing = {p for g in opt.param_groups for p in g["params"]}
                new = [p for p in op.parameters() if p not in existing]
                if new:
                    opt.add_param_group({"params": new})
        self._concomitant_configured = True

    def _correct_input(self, image: torch.Tensor) -> torch.Tensor:
        """Pre-rotate ``image`` into the concomitant frame, keep input layout.

        ``apply_concomitant_correction`` casts real → complex and multiplies by
        :math:`e^{i\\phi_c}`. We project back to the host network's real channel
        layout: interleaved real/imag when ``C`` is even, magnitude otherwise.
        """
        corrected = self.apply_concomitant_correction(image)
        if not torch.is_complex(corrected):
            return corrected
        if image.ndim == 4 and image.shape[1] % 2 == 0 and not torch.is_complex(image):
            # [B, 2k, H, W] interleaved real/imag: recombine each (real, imag)
            # channel pair into a complex map, multiply by the concomitant
            # phasor, then split back to the network's interleaved real layout.
            cplx = torch.complex(image[:, 0::2], image[:, 1::2]) * self.concomitant_op.phasor
            out = torch.empty_like(image)
            out[:, 0::2] = cplx.real
            out[:, 1::2] = cplx.imag
            return out
        # Real magnitude / single-channel image: concomitant is a pure phase
        # effect, so the magnitude is unchanged — return the original (real) image.
        return image

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Resolve the dict/tensor batch via the shared F6 helper.
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        image = self._batch_image(batch)
        if image is None:
            image = self._batch_image(input_batch)

        if image is not None:
            self._ensure_concomitant_configured(image)

        # Apply the concomitant phase correction to the model INPUT directly and
        # forward the corrected tensor through the parent reconstruction loss.
        # (An earlier approach wrapped + reassigned ``self.env.generator``, but
        # ``env.generator`` is a read-only property with no setter -> it raises
        # AttributeError in production; only a settable stub env masked it.)
        corrected_input = input_batch
        if self._concomitant_configured and isinstance(input_batch, torch.Tensor):
            corrected_input = self._correct_input(input_batch)

        return super()._compute_losses_impl(
            input_batch=corrected_input,
            target_batch=target_batch,
            epoch=epoch,
            **kwargs,
        )
