"""TRELLIS Training Strategy Module

This module contains TRELLIS training strategies.
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F

from mriforge.infrastructure.training.contexts import TrainingEnvironment
from mriforge.models.losses.computers import UnifiedReconstructionLossComputer

from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class TRELLISTrainingStrategy(BaseTrainingStrategy):
    """TRELLIS: Regression-based Efficient 3D Asset Generation from 2D slices.

    Trains 3D volumetric reconstruction model to lift 2D MRI slices into complete
    3D volumes. Uses coarse-to-fine hierarchical generation for efficient computation.\n    ## 3D Reconstruction Challenge

    **Problem**: MRI scanners acquire 2D slices sequentially, not true 3D volumes.
    Each slice has unknown spatial relationships, out-of-plane information missing.
    Goal: Reconstruct full 3D anatomy from stacked 2D acquisitions.

    **TRELLIS Approach**: Tree-structured progressive generation from coarse to fine
    resolution, maintaining consistency across levels.

    ## Architecture

    **Input**: 2D MRI slices (typically 2D+time: video-like sequence)

    **Generation Hierarchy**:
    1. **Coarse Level** (8×8×8 or 16×16×16):
       - Capture global 3D shape
       - Efficient computation
       - Quick convergence

    2. **Medium Level** (32×32×32):
       - Refine anatomical details
       - Add finer structure
       - Maintain coarse consistency

    3. **Fine Level** (64×64×64 or higher):
       - High-resolution reconstruction
       - Preserve micro-structure
       - Computationally expensive but constrained by coarse predictions

    **Loss Propagation**: Coarse predictions inform all finer levels (prevents divergence)

    ## Training Process

    1. **Condition on Slices**: 2D encoder processes input slices → feature embedding
    2. **Coarse Generation**: Generate low-res 3D volume
    3. **Coarse Loss**: Compute loss on low-res prediction
    4. **Medium Generation**: Refine coarse → medium-res  5. **Medium Loss**: Compute loss, backprop through coarse+medium
    6. **Fine Generation**: Refine medium → high-res
    7. **Fine Loss**: Compute loss, backprop through all levels
    8. **Backward**: Aggregate gradients across hierarchy

    ## Configuration

    - `training.training_mode`: 'volumetric' or 'reconstruction'
    - `training.volumetric.coarse_resolution`: Starting level (8, 16)
    - `training.volumetric.fine_resolution`: Target level (64, 128)
    - `training.volumetric.num_levels`: Hierarchy depth (default 3)
    - `objectives.reconstruction.lambda_l1`: Reconstruction weight

    ## Loss Components

    1. **Hierarchical Reconstruction Loss**:
       - Coarse level: L1 on 8×8×8 volume
       - Medium level: L1 on 32×32×32 volume
       - Fine level: L1 on 64×64×64 volume
       - Equal weight per level or pyramidal weighting

    2. **Perceptual Loss** (optional):
       - Apply to each hierarchical level
       - Ensures realistic texture at all scales

    3. **Smoothness Loss** (optional):
       - Encourages spatial coherence across levels
       - Prevents unrealistic discontinuities

    4. **Consistency Loss** (optional):
       - Upsampled coarse output should match fine generation
       - Enforces hierarchical consistency

    ## Key Features

    ✅ **Coarse-to-Fine**: Pyramid structure enables stable training
    ✅ **Scalable**: Fine levels can be generated progressively
    ✅ **Efficient**: Coarse levels computed on low-res (fast)
    ✅ **Consistent**: Each level constrained by coarser prediction
    ✅ **Multi-Scale**: Captures both global shape and fine details

    ## Inference

    1. Input: 2D MRI slices
    2. Encode slices → feature embedding
    3. Generate coarse (8×8×8) volume
    4. Upsample to medium (32×32×32)
    5. Generate medium refinements
    6. Upsample to fine (64×64×64)
    7. Generate fine refinements
    8. Output: 3D volume ready for surgical planning/analysis

    ## Advantages

    - ✅ **Complete 3D**: Reconstructs full volume, not slice-by-slice
    - ✅ **Efficient**: Coarse levels cheap to compute
    - ✅ **Stable Training**: Hierarchical constraints prevent divergence
    - ✅ **Out-of-Plane**: Inherently reconstructs missing out-of-plane information

    ## Disadvantages

    - ❌ **Complexity**: Multiple generation levels to train
    - ❌ **Memory**: Full 3D volume storage even at coarse levels
    - ❌ **Slow**: Progressive refinement takes multiple forward passes
    - ❌ **Limited**: Assumes slices can represent true 3D (fails with large gaps)

    Attributes:
        state: TrainingState with 3D generator
        loss_computer: UnifiedReconstructionLossComputer for hierarchical loss
        coarse_res: Coarse volume resolution
        fine_res: Target fine volume resolution
        device: Computation device (CUDA/CPU)

    References:
        - Karnewar & Vedaldi (2023): TRELLIS: Regression-based Treellis for Efficient
          3D Generative Models
    """

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        **kwargs: object,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
        """
        super().__init__(env=env, **kwargs)

        # Initialize strategy-specific components using unified loss computer
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config, device=self.device
        )
        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize TRELLIS-specific components and perform validation."""
        # TRELLIS extends BaseTrainingStrategy directly (not via the
        # reconstruction mixin that calls StrategyInitializationHelper
        # .initialize_metrics_adapter), so ``metrics_adapter`` is never set by
        # an ancestor. ``validation_step`` reads ``self.metrics_adapter``,
        # which would raise AttributeError at runtime (only masked in tests by
        # a manual MagicMock injection). Default it to None so the guarded
        # block degrades cleanly to base metrics.
        self.metrics_adapter = None
        self._verify_strategy_config(expected_modes=("3d_generation",))
        self._log_config_features(self.logging_service)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for TRELLIS training step with AMP support.

        Phase 4b-2: Enhanced with LossResult infrastructure for metrics tracking.
        3D volumetric generation with multi-scale loss supervision.

        Args:
            input_batch: Low-resolution input tensor batch.
            target_batch: High-resolution target tensor batch.
            epoch: Current epoch index.
            **kwargs: Additional optional context.

        Returns:
            Dictionary of loss tensors.
        """
        if self.state.opt_g is None:
            raise RuntimeError("Generator optimizer is required for TRELLIS training")

        input_batch = input_batch.to(self.device, non_blocking=True)
        target_batch = target_batch.to(self.device, non_blocking=True)

        generated_3d = self.env.generator(input_batch)
        generated_3d = generated_3d.to(target_batch.device)
        target_volume = self._prepare_target_volume(target_batch, generated_3d)

        # [FIX] Use Unified Loss Computer (SSOT)
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        loss_output = self.loss_computer.compute(
            pred=generated_3d,
            target=target_volume,
            epoch=epoch,
            losses_dict=env_losses,
        )

        total_loss = loss_output.total
        components = loss_output.components

        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(components)
        self._loss_dict_reuse["g_total_loss"] = total_loss

        return self._loss_dict_reuse

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Performs validation step for TRELLIS with 3D metrics."""
        batch = (input_batch, target_batch)
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        lr_reals = input_batch
        hr_reals = target_batch

        self.env.generator.eval()
        generated_3d = self.env.generator(lr_reals)
        if generated_3d.device != hr_reals.device:
            generated_3d = generated_3d.to(hr_reals.device)

        target_volume = self._prepare_target_volume(hr_reals, generated_3d)

        if self.metrics_adapter is not None:
            metrics = self.metrics_adapter.summarize(
                generated_3d,
                target_volume,
            )
        else:  # pragma: no cover - legacy fallback
            l1_loss = F.l1_loss(
                generated_3d,
                target_volume,
            )
            mse_loss = F.mse_loss(
                generated_3d,
                target_volume,
            )
            metrics = {
                "mae": l1_loss.detach(),
                "mse": mse_loss.detach(),
            }

        metrics["volume_mse"] = metrics.get("mse", torch.tensor(0.0, device=generated_3d.device))
        metrics["volume_l1"] = F.l1_loss(generated_3d, target_volume).detach()

        return metrics

    def _prepare_target_volume(
        self,
        target_batch: torch.Tensor,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        """Align ground truth volumes with generated outputs."""
        target = target_batch.to(generated.device)
        if target.dtype != generated.dtype:
            target = target.to(generated.dtype)

        if target.ndim == 4:
            target = target.unsqueeze(2)

        if target.shape != generated.shape:
            target = F.interpolate(
                target,
                size=generated.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        return target
