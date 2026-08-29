from typing import Any

import torch

from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy
from mriforge.models.losses.registration import LocalCrossCorrelationLoss, SmoothnessLoss


class B0MappingStrategy(BaseTrainingStrategy):
    """Deformable registration-based image alignment (NOT B0 field estimation).

    HONESTY NOTE (audit 2026-07): despite the ``b0_mapping`` name, this strategy
    does **not** estimate a B0 off-resonance field map in Hz. Its
    ``_compute_losses_impl`` runs a generic deformable image registration:

        corrected, flow = generator(moving, fixed)
        loss = λ_sim · LNCC(corrected, fixed) + λ_smooth · Smoothness(flow)

    The network output ``flow`` is a **spatial displacement field (pixels)**, not
    off-resonance in Hz, and there is **no** phase-evolution forward model
    (``φ = 2π·B0·t``) and **no** multi-echo phase-difference fit anywhere in the
    loss. Grading ``flow`` with an image-similarity metric therefore certifies
    registration quality, not field-map accuracy (metric↔claim mismatch,
    pitfall #18). Do not read the output as a field map.

    For genuine B0 field estimation from multi-echo phase (conjugate-product
    phase difference + Tikhonov solve, graded in Hz against a reference with
    ``core.metrics.b0_field_rmse.B0FieldRMSE``) use
    :class:`~mriforge.infrastructure.training.strategies.multi_echo_b0_fit_strategy.MultiEchoB0FitStrategy`
    (SSOT: ``infrastructure.physics.b0_mapping``).

    Loss terms
    ----------
    - **LNCC** (``losses.registration.lambda_sim``): local normalised
      cross-correlation image similarity between the warped moving image and the
      fixed image (intensity-invariant; minimised as a negative correlation).
    - **Smoothness** (``losses.registration.lambda_smooth``): ``||∇flow||²`` on
      the *deformation field* (not a B0 field) to keep the warp diffeomorphic.

    Attributes:
        lncc_loss: LocalCrossCorrelationLoss image-similarity term.
        smooth_loss: SmoothnessLoss deformation-field regulariser.
        device: Computation device (CUDA/CPU).

    References:
        - Balakrishnan et al. (2019): VoxelMorph — unsupervised deformable registration.
        - Hutton et al. (2002): Image registration: Part II.
    """

    def __init__(self, env=None, **kwargs):
        """__init__.

        Args:
            env (Any): Description.
        """
        super().__init__(env=env, **kwargs)

        # Model is already initialized in env.generator via TrainingEnvironmentDirector
        # self.model = self.env.generator

        self.lncc_loss = LocalCrossCorrelationLoss()
        self.smooth_loss = SmoothnessLoss()

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for B0 estimation.
        input_batch: moving/64mT
        target_batch: fixed/3T
        """
        if input_batch is None or target_batch is None:
            raise ValueError("Batch must contain 'input' and 'target' for B0 estimation.")

        # Forward pass
        # model takes (moving, fixed)
        # Using self.env.generator which is the B0Estimator
        corrected, flow = self.env.generator(input_batch, target_batch)

        # Compute losses
        loss_sim = self.lncc_loss(corrected, target_batch)
        loss_smooth = self.smooth_loss(flow)

        # SSOT: read weights from config.losses.registration (RegistrationLossesConfig)
        lambda_sim = self.config.losses.registration.lambda_sim
        lambda_smooth = self.config.losses.registration.lambda_smooth

        # Weighted sum (LNCC is negative, minimize it)
        loss = lambda_sim * loss_sim + lambda_smooth * loss_smooth

        return {
            "g_total_loss": loss,  # Required by BaseTrainingStrategy
            "sim_loss": loss_sim,
            "smooth_loss": loss_smooth,
        }

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation step for registration."""

        # Base class validation_step already unpacks input/target
        # input_batch: moving, target_batch: fixed

        with torch.no_grad():
            self.env.generator.eval()

            # Forward pass
            corrected, flow = self.env.generator(input_batch, target_batch)

            loss_sim = self.lncc_loss(corrected, target_batch)

            # Log images occasionally
            if (
                hasattr(self, "logging_service")
                and self.logging_service
                and hasattr(self.logging_service, "step")
                and hasattr(self.config, "logging")
                and self.logging_service.step % self.config.logging.intervals.log == 0
            ):
                self.logging_service.log_images(
                    "val/corrected", corrected, self.logging_service.step
                )
                self.logging_service.log_images(
                    "val/flow", flow[:, 0:1], self.logging_service.step
                )  # Viz Y-flow

            self.env.generator.train()

        return {"val_loss": loss_sim.detach(), "val_lncc": loss_sim.detach()}
