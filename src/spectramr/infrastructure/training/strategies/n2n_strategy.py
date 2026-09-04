#!/usr/bin/env python
"""Noise-to-Noise Self-Supervised Denoising Strategy

Paradigm: Self-Supervised Denoising for Multi-Repetition Data.
Uses multi-repetition data (M4Raw) to learn signal recovery without ground truth.
Assumption: Noise is zero-mean and uncorrelated between scans.
"""

from typing import Any

import torch

from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer


class NoiseToNoiseStrategy(BaseTrainingStrategy):
    """Self-Supervised Denoising using N2N paradigm.

    Learns to denoise multi-repetition MRI data without requiring ground truth images.
    Uses pairs of noisy acquisitions to train a denoising network under the assumption
    that noise is uncorrelated between repetitions.

    Typical use case: M4Raw dataset with 2-3 repetitions of same slice.

    Attributes:
        env: Training environment with generator and optimizers
        loss_computer: UnifiedReconstructionLossComputer for SSOT loss computation
    """

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        **kwargs: Any,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
        """
        super().__init__(env=env, **kwargs)
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config, device=self.device
        )

    def _setup_strategy_specific_components(self) -> None:
        """No additional components needed for N2N."""
        pass

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Custom batch unpacking for N2N to handle multi-repetition data."""
        input_batch, target_batch = None, None

        # N2N specific data extraction: look for 'image_reps' [B, N, C, H, W]
        if isinstance(batch, dict):
            reps = batch.get("image_reps")
            if reps is not None and isinstance(reps, torch.Tensor) and reps.ndim == 5:
                # N must be at least 2
                if reps.shape[1] >= 2:
                    # Pick two DISTINCT repetitions at random each step. N2N only
                    # needs two independent noisy draws of the same signal, so
                    # fixing 0/1 wasted any extra reps (e.g. rep 2 of 3-rep
                    # M4Raw) and biased training toward one pair (audit 2026-06).
                    i, j = torch.randperm(reps.shape[1])[:2].tolist()
                    input_batch = reps[:, i]
                    target_batch = reps[:, j]
                    return input_batch, target_batch
                # Fail loudly on single-repetition data rather than falling
                # through to the generic "(None, None)" mixin path, which would
                # surface the misleading downstream "requires both" error and
                # hide the real cause (pitfall #9 spirit).
                msg = (
                    "Noise2Noise requires 'image_reps' with at least 2 "
                    f"repetitions; got shape[1]={reps.shape[1]}."
                )
                raise ValueError(msg)

        # Fallback to standard mixin unpacking if N2N-specific keys missing
        return super()._unpack_batch(batch)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute N2N denoising loss via UnifiedReconstructionLossComputer (SSOT).

        Args:
            input_batch: First repetition [B, C, H, W]
            target_batch: Second repetition [B, C, H, W]
            epoch: Current epoch
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss components from the loss computer
        """
        generator = self.env.generator

        if input_batch is None or target_batch is None:
            raise ValueError(
                "Noise2Noise strategy requires both input and target tensors "
                "(two independent noisy repetitions). Check that your dataset "
                "provides 'image_reps' with at least 2 repetitions, or "
                "'input'/'target' keys in each batch."
            )

        # Forward pass: denoise first repetition
        denoised_pred = generator(input_batch)

        # SSOT: Delegate loss computation to UnifiedReconstructionLossComputer
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        # The pipeline forwards the step counter as ``iteration`` (train.py),
        # not ``step``; reading the wrong key froze the loss-weight schedule at
        # iteration 0. Prefer the canonical ``iteration`` key, keep ``step`` as
        # a legacy fallback.
        iteration = kwargs.get("iteration", kwargs.get("step", 0))
        loss_output = self.loss_computer.compute(
            pred=denoised_pred,
            target=target_batch,
            epoch=epoch,
            iteration=iteration,
            losses_dict=env_losses,
        )

        result = {"g_total_loss": loss_output.total}
        result.update(loss_output.components)
        return result
