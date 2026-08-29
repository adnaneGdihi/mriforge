"""Phase-contrast velocity mapping — the trainable backing for ``mri_flow``.

This is phase-contrast MRI, **not** flow matching (see
``config/schemas/training/phase_contrast.py`` for the naming rationale).

The vertical it completes: real physics
(``infrastructure/physics/signal_models/flow_encoding``), a registered
``phase_contrast`` operator, FLOW-tagged losses and metrics have all existed
since the EVAL_ONLY -> PARTIAL promotion, but nothing trainable consumed them —
so ``mri_flow`` could not honestly claim LIVE.

The model estimates a velocity field from the flow-encoded phases; the losses
grade it through the encoding ``phi = pi * v / venc``. Velocity is in **physical
units (m/s)**: the encoding and the ``|v| > venc`` unwrap hinge are meaningless
otherwise, so a generator with a terminal ``tanh`` would silently destroy both.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from mriforge.config.schemas.enums import Regime, Task
from mriforge.infrastructure.physics.signal_models.flow_encoding import (
    four_point_reference_decode,
    four_point_reference_encode,
)
from mriforge.models.capabilities import StrategyCapabilities

from .reconstruction import ReconstructionTrainingStrategy

#: Which channel of a [B, 3, H, W] velocity field is which component.
_COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def encode_velocity_four_point(velocity: torch.Tensor, venc: float) -> torch.Tensor:
    """``[B, 3, H, W]`` velocity -> ``[B, 4, H, W]`` four-point phases.

    A thin channel-first adapter over
    :func:`~mriforge.infrastructure.physics.signal_models.flow_encoding.four_point_reference_encode`,
    which operates on the LAST axis (``[..., 3] -> [..., 4]``). Kept as a
    module-level pure function so the permute is unit-testable without building
    a strategy.
    """
    return four_point_reference_encode(velocity.permute(0, 2, 3, 1), venc).permute(0, 3, 1, 2)


def decode_velocity_four_point(phases: torch.Tensor, venc: float) -> torch.Tensor:
    """``[B, 4, H, W]`` four-point phases -> ``[B, 3, H, W]`` velocity."""
    return four_point_reference_decode(phases.permute(0, 2, 3, 1), venc).permute(0, 3, 1, 2)


class PhaseContrastFlowStrategy(ReconstructionTrainingStrategy):
    """Estimate a velocity field from four-point flow-encoded phases.

    Subclasses :class:`ReconstructionTrainingStrategy` for its optimizer
    plumbing and the ``_apply_builder_image_losses`` folding seam, without the
    diffusion ``timesteps`` coupling (the ``field_flow_strategy`` precedent).

    Batch contract (emitted by ``data.phase_contrast``):
        - ``input``: ``[B, 4, H, W]`` four-point encoded phases (ref, x, y, z)
        - ``velocity``: ``[B, 3, H, W]`` reference velocity, m/s
        - ``inlet_mask`` / ``outlet_mask``: ``[B, 1, H, W]``, if flux_conservation
    """

    #: The trainable backing for the mri_flow regime's LIVE claim. Velocity IS
    #: the estimated physical parameter, hence PARAMETER_MAPPING; RECONSTRUCTION
    #: is declared because the profile supports it and the same objective serves
    #: an undersampled-flow recon arm.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.FLOW}),
        tasks=frozenset({Task.PARAMETER_MAPPING, Task.RECONSTRUCTION}),
    )

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("phase_contrast_flow", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)

        pc = getattr(self.config.data, "phase_contrast", None)
        if pc is None or not pc.enabled:
            raise ValueError(
                "PhaseContrastFlowStrategy requires `data.phase_contrast.enabled: "
                "true` — it supplies the venc and the encoding scheme. Without it "
                "the arm has no velocity encoding and the objective is undefined."
            )
        self._venc = float(pc.venc)
        self._voxel_area = float(pc.voxel_area)
        self._through_plane_index = _COMPONENT_INDEX[pc.through_plane_axis]
        self._flux_conservation = bool(pc.flux_conservation)

        out_channels = int(getattr(self.config.model, "out_channels", 0))
        if out_channels != 3:
            raise ValueError(
                "PhaseContrastFlowStrategy predicts a 3-component velocity field "
                f"(vx, vy, vz); got model.out_channels={out_channels}. A "
                "1-channel generator would broadcast silently into the velocity "
                "losses and grade a scalar against a vector field."
            )

        cfg = getattr(self.config.training, "phase_contrast", None)
        self._validate_encode_roundtrip = (
            bool(getattr(cfg, "validate_encode_roundtrip", False)) if cfg is not None else False
        )
        self._roundtrip_checked = False

        # Resolve every weight once, here, so an all-zero objective fails at
        # construction rather than after an epoch of meaningless training.
        self._lambda_velocity = self._get_loss_weight("phase_contrast_velocity")
        self._lambda_flux = self._get_loss_weight("through_plane_flux_conservation")
        self._lambda_unwrap = self._get_loss_weight("velocity_unwrap_consistency")
        if max(self._lambda_velocity, self._lambda_flux, self._lambda_unwrap) <= 0.0:
            raise ValueError(
                "No FLOW loss carries a non-zero weight, so this arm would train "
                "as a plain regressor while advertising phase-contrast physics "
                "(pitfall #16). Declare at least one of "
                "losses.physics.lambda_{phase_contrast_velocity,"
                "through_plane_flux_conservation,velocity_unwrap_consistency} > 0."
            )
        if self._lambda_flux > 0.0 and not self._flux_conservation:
            raise ValueError(
                "lambda_through_plane_flux_conservation > 0 requires "
                "data.phase_contrast.flux_conservation: true, which is what "
                "mandates the inlet/outlet mask batch keys."
            )

    def _resolve_velocity_target(self, batch: Any) -> torch.Tensor:
        """The reference velocity field, from ``velocity`` or canonical ``target``.

        Both names are legitimate: the velocity field IS this arm's target, and
        the canonical pipeline REQUIRES a ``target`` key (``BatchAdapter.from_dict``
        raises without one), so a flow batch could not traverse it otherwise.
        ``velocity`` is the explicit spelling; ``target`` is the canonical one.

        What makes this safe is not the key name but the SHAPE CONTRACT. The risk
        with reading ``target`` is grading a velocity field against an image
        (pitfall #18) — so the resolved tensor must be a 3-component field, and a
        1-channel magnitude image fails loudly here rather than broadcasting into
        a plausible-looking number.
        """
        velocity = batch.get("velocity")
        if velocity is None:
            velocity = batch.get("target")
        if velocity is None:
            raise ValueError(
                "PhaseContrastFlowStrategy needs a reference velocity field on "
                "the batch, as 'velocity' or the canonical 'target' key "
                "(data.phase_contrast emits it)."
            )
        if velocity.ndim < 2 or velocity.shape[1] != 3:
            raise ValueError(
                "The reference velocity must be a 3-component field "
                f"[B, 3, H, W]; got {tuple(velocity.shape)}. A 1-channel target "
                "means this arm is pointed at a magnitude-image dataset, and the "
                "flow losses would grade velocity against an image (pitfall #18)."
            )
        return velocity

    def _check_encode_roundtrip(self, phases: torch.Tensor, velocity: torch.Tensor) -> None:
        """Assert the batch really is four-point encoded at the declared venc.

        A venc mismatch between config and acquisition rescales every velocity
        linearly. Nothing else would notice: the loss converges happily onto a
        uniformly wrong field, and the metrics grade against the same wrong
        reference. Cheap to check once, impossible to spot later.
        """
        decoded = decode_velocity_four_point(phases, self._venc)
        if not torch.allclose(decoded, velocity, atol=1e-3, rtol=1e-3):
            max_err = (decoded - velocity).abs().max().item()
            raise ValueError(
                "training.phase_contrast.validate_encode_roundtrip is on and the "
                f"check FAILED (max |decode(input) - velocity| = {max_err:.4g} "
                f"m/s at venc={self._venc}). The batch's phases are not "
                "four-point encoded at this venc — every velocity is being "
                "rescaled, and training would silently converge to the wrong "
                "field. Fix data.phase_contrast.venc to match the acquisition."
            )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del target_batch, epoch
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):
            # dict OR TrainingBatch — the canonical pipeline passes the latter.
            raise ValueError(
                "PhaseContrastFlowStrategy requires a mapping batch "
                f"(dict/TrainingBatch) carrying the flow-encoded phases; got "
                f"{type(batch)!r}."
            )

        phases = batch["input"]  # [B, 4, H, W]
        velocity_target = self._resolve_velocity_target(batch)

        if self._validate_encode_roundtrip and not self._roundtrip_checked:
            self._check_encode_roundtrip(phases, velocity_target)
            self._roundtrip_checked = True

        velocity_pred = self.env.generator(phases)
        if isinstance(velocity_pred, tuple):
            velocity_pred = velocity_pred[0]

        # Lazy import: avoids a models <- infrastructure cycle at module load
        # (the field_flow_strategy precedent).
        from mriforge.models.losses.flow_losses import (
            PhaseContrastVelocityLoss,
            ThroughPlaneFluxConservationLoss,
            VelocityUnwrapConsistencyLoss,
        )

        components: dict[str, torch.Tensor] = {}
        total = velocity_pred.new_zeros(())

        # EVERY call below is BY KEYWORD. These losses take positional tensors
        # that are NOT (pred, target)-shaped, so routing them through the generic
        # ComposedLoss.forward(pred, target) would bind the wrong arguments.
        if self._lambda_velocity > 0.0:
            loss_velocity = PhaseContrastVelocityLoss(venc=self._venc)(
                v_pred=velocity_pred, v_target=velocity_target, venc=self._venc
            )
            components["loss_phase_contrast_velocity"] = loss_velocity.detach()
            total = total + self._lambda_velocity * loss_velocity

        if self._lambda_unwrap > 0.0:
            loss_unwrap = VelocityUnwrapConsistencyLoss(venc=self._venc)(
                v_pred=velocity_pred, venc=self._venc
            )
            components["loss_velocity_unwrap_consistency"] = loss_unwrap.detach()
            total = total + self._lambda_unwrap * loss_unwrap

        if self._lambda_flux > 0.0:
            inlet_mask = batch.get("inlet_mask")
            outlet_mask = batch.get("outlet_mask")
            if inlet_mask is None or outlet_mask is None:
                raise ValueError(
                    "through_plane_flux_conservation is weighted > 0 but the "
                    "batch carries no 'inlet_mask'/'outlet_mask'. Skipping the "
                    "term would leave a weighted loss silently uncomputed."
                )
            # voxel_area can ONLY arrive per-call: the loss has no __init__, so
            # the LossBuilder cannot parameterise it from YAML.
            # Slice as idx:idx+1, NOT idx — the latter drops the channel axis to
            # [B, H, W], which then broadcasts against the [B, 1, H, W] masks into
            # [B, B, H, W]: every sample's velocity multiplied by every OTHER
            # sample's mask. The flux reduces over (-2, -1) only, so that yields a
            # [B, B] matrix of cross-sample fluxes and a finite, plausible loss.
            # It never raises. Keeping the axis makes the broadcast a no-op.
            loss_flux = ThroughPlaneFluxConservationLoss()(
                v_through_plane=velocity_pred[
                    :, self._through_plane_index : self._through_plane_index + 1
                ],
                inlet_mask=inlet_mask,
                outlet_mask=outlet_mask,
                voxel_area=self._voxel_area,
            )
            components["loss_through_plane_flux_conservation"] = loss_flux.detach()
            total = total + self._lambda_flux * loss_flux

        # Fold the declarative losses.image_losses block, or it is an inert decoy
        # (pitfall #16). Graded velocity-vs-velocity — the only physically
        # commensurate pair this strategy has.
        folded = self._apply_builder_image_losses(velocity_pred, velocity_target, components)
        if folded is not None:
            total = total + folded

        self._last_prediction = velocity_pred
        self._last_target = velocity_target
        return {"g_total_loss": total, **components}

    def _validation_forward(
        self,
        input_batch: Any,
        batch_context: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the velocity field — what the FLOW metrics actually grade.

        velocity_rmse / peak_velocity_error / divergence / vnr all score a
        velocity field. Returning an image here would be a metric-claim mismatch
        (pitfall #18).
        """
        del kwargs
        phases = (
            batch_context.get("input", input_batch)
            if hasattr(batch_context, "get")
            else input_batch
        )
        out = self.env.generator(phases)
        return out[0] if isinstance(out, tuple) else out


__all__ = [
    "PhaseContrastFlowStrategy",
    "decode_velocity_four_point",
    "encode_velocity_four_point",
]
