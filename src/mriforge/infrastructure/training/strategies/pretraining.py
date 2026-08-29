"""MAEPretrainingStrategy Training Strategy Module

This module contains MAEPretrainingStrategy training strategies.
"""

from typing import Any

import torch

from mriforge.infrastructure.training.contexts import TrainingEnvironment
from mriforge.models.losses.computers import UnifiedMAELossComputer

from .base import BaseTrainingStrategy


class MAEPretrainingStrategy(BaseTrainingStrategy):
    """Masked Autoencoder (MAE) pretraining strategy for self-supervised learning.

    Implements MAE pretraining where random patches are masked and the model learns
    to reconstruct missing patches from visible context. Achieves strong unsupervised
    feature learning without labels or contrastive pairs.

    ## MAE Concept

    **Objective**: Mask 75% of image patches, train encoder-decoder to predict
    missing patches from visible ones. Forces learning of rich semantic features
    and spatial relationships without supervision.

    **Key Insight**: Unlike contrastive learning (SSL), MAE uses generative task
    (patch prediction) as self-supervision. More directly related to downstream
    reconstruction tasks.

    ## Architecture

    **Asymmetric Encoder-Decoder**:
    - **Encoder**: Processes only visible patches (→3x speed vs full image)
    - **Decoder**: Lightweight, reconstructs from encoder + mask tokens
    - Masking reduces computation significantly during pretraining

    **Patch Tokenization**:
    - Input image → Non-overlapping patches (16×16 typical)
    - Linear embedding for each patch
    - Positional embeddings preserve spatial information

    **Mask Tokens**:
    - Learnable tokens for masked patches
    - Decoder sees concatenation of [encoder output, mask tokens]
    - Symmetry restoration happens in decoder

    ## Training Process

    1. **Input**: Original image (e.g., 224×224)
    2. **Patching**: Divide into 196 patches (16×16 each)
    3. **Masking**: Randomly mask 60-75% of patches
    4. **Encoding**: Pass visible patches through encoder (3x faster)
    5. **Decoding**: Reconstruct all patches via decoder
    6. **Loss**: MSE on masked regions (visible regions as context)
    7. **Backward**: Update encoder and decoder jointly

    ## Configuration

    - `training.training_mode`: 'mae' or 'self_supervised'
    - `training.mae.mask_ratio`: Fraction masked (0.75 = 75%, typical)
    - `training.mae.patch_size`: Patch dimensions (16, 32)
    - `training.mae.decoder_depth`: Decoder layers (8-12)
    - `objectives.reconstruction.lambda_l1`: Reconstruction weight

    ## Loss Components

    - **Reconstruction Loss**: MSE on masked patch pixels (primary)
       - L_mae = ||x_hat - x||² computed only on masked patches
       - Visible patches provide context without loss signal

    - **Optional Perceptual Loss**: VGG features (if configured)

    - **Optional Adversarial Loss**: Discriminator on reconstructed image

    ## Key Features

    ✅ **Asymmetric**: Encoder sees only 25%, decoder reconstructs all (efficient)
    ✅ **Self-Supervised**: No labels or pairs required
    ✅ **Simple**: Generation task more interpretable than contrastive
    ✅ **Transferable**: Learned representations transfer to 15+ downstream tasks
    ✅ **Scalable**: Works with any image size or modality

    ## Pretraining Benefits

    ✅ **Faster Downstream**: 10-20% faster convergence on target task
    ✅ **Better Convergence**: Higher final accuracy than random init
    ✅ **Data Efficient**: Effective with limited labeled data (few-shot)
    ✅ **Robust**: Learned features generalize across domains

    ## Fine-tuning Strategy

    1. **Feature Extraction**: Use pre-trained encoder for downstream
    2. **Task Head**: Train classifier or regression head on encoder output
    3. **Warmup**: Freeze encoder for 1 epoch, then unfreeze for full fine-tuning
    4. **Learning Rate**: Lower LR for encoder (0.1-0.01x task LR)
    5. **Scheduler**: Cosine annealing with warmup restart per epoch

    ## Inference

    1. Input: Image patch (visible regions only)
    2. Encoder: Extract features from visible patches (→3x faster)
    3. Decoder: Reconstruct masked patches
    4. Output: Complete reconstructed image with filled-in patches

    ## Advanced Variations

    **Hierarchical MAE**:
    - Multi-scale patches: small + large simultaneously
    - Coarse-to-fine reconstruction

    **Token Masking**:
    - Instead of patch masking, mask individual tokens
    - Finer-grained reconstruction task

    **Hybrid MAE+Contrastive**:
    - Combine MAE with contrastive (SimCLR, MoCo)
    - Better representations than either alone

    Attributes:
        state: TrainingState with MAE model
        loss_computer: UnifiedMAELossComputer for reconstruction
        mask_ratio: Fraction of patches to mask (default 0.75)
        patch_size: Size of patches (e.g., 16 for 16×16)
        device: Computation device (CUDA/CPU)

    References:
        - He et al. (2022): Masked Autoencoders Are Scalable Vision Learners (MAE)
        - Xie et al. (2022): An Empirical Study of Training Self-Supervised Vision Transformers
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
        self.loss_computer = UnifiedMAELossComputer(config=self.config, device=self.device)
        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize MAE-specific components and perform validation."""
        self._verify_strategy_config(
            expected_modes=(
                "self_supervised",
                "ssl_pretraining",
                "ssl",
                "mae",
                "mae_pretraining",
                "reconstruction",
            ),
        )
        self._log_config_features(self.logging_service)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for MAE pretraining step with AMP support.

        Phase 4b-2: Enhanced with LossResult infrastructure for metrics tracking.
        Self-supervised contrastive loss computation.

        Args:
            input_batch: Low-resolution input tensor batch (unused for MAE).
            target_batch: High-resolution input tensor batch (used as input and target).
            epoch: Current epoch index.
            **kwargs: Additional optional context (may include batch dict).

        Returns:
            Dictionary of loss tensors.
        """
        # Handle both batch formats via the SSOT seam (rather than re-reading
        # raw kwargs): _resolve_legacy_batch checks kwargs["batch"] then
        # input_batch.
        resolved = self._resolve_legacy_batch(input_batch, kwargs)
        if isinstance(resolved, dict) and "target" in resolved:
            # MAE uses batch dict format: 'target' is the High-Res image
            input_batch = resolved["target"]
        else:
            # Use target_batch as input for MAE (self-supervised on high-res images)
            input_batch = target_batch

        # The reconstruction target is the ORIGINAL clean image. In the
        # k-space branch below ``input_batch`` is overwritten with its
        # undersampled-IFFT corruption (that corruption is the *generator
        # input*), so the target must be captured here, before the overwrite.
        # Using the corrupted image as the target makes the identity map
        # optimal — a degenerate objective (CLAUDE.md pitfall #20).
        mae_target = input_batch

        # [Phase 3.3] K-Space Block Masking
        mask_domain = "image"
        if hasattr(self.config, "task_settings") and isinstance(self.config.task_settings, dict):
            mask_domain = self.config.task_settings.get("mask_domain", mask_domain)
        elif hasattr(self.config.training, "task_settings") and isinstance(
            getattr(self.config.training, "task_settings", None), dict
        ):
            mask_domain = self.config.training.task_settings.get("mask_domain", mask_domain)

        if (
            hasattr(self.config, "training")
            and self.config.training is not None
            and hasattr(self.config.training, "mae")
        ):
            mae_config = self.config.training.mae
            if hasattr(mae_config, "mask_domain"):
                mask_domain = mae_config.mask_domain
            elif isinstance(mae_config, dict):
                mask_domain = mae_config.get("mask_domain", mask_domain)

        if mask_domain == "kspace":
            from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
            from mriforge.infrastructure.physics.sampling import MaskGenerator, MaskType

            # Route through the centered-FFT SSOT (fft2c/ifft2c handle the 4-D
            # [B,C,H,W] MAE batch; the old FFTTransformer.fft2 unpacked a 5-D
            # volume and raised ValueError on every k-space-masking step).
            # Treat a real image as complex for the round-trip.
            _was_real = not torch.is_complex(input_batch)
            _x_complex = (
                torch.complex(input_batch, torch.zeros_like(input_batch))
                if _was_real
                else input_batch
            )
            kspace = fft2c(_x_complex)

            # Prefer the typed training.mae block (validated 0 ≤ ratio < 1);
            # fall back to the legacy task_settings dict, then the default
            # (pitfall #15 — mask_ratio was previously read only from the
            # untyped dict, so a typed/validated value was a silent no-op).
            mae_cfg = self.config.training.mae if self.config.training is not None else None
            if mae_cfg is not None and hasattr(mae_cfg, "mask_ratio"):
                mask_ratio = float(mae_cfg.mask_ratio)
            elif hasattr(self.config, "task_settings") and isinstance(
                self.config.task_settings, dict
            ):
                mask_ratio = self.config.task_settings.get("mask_ratio", 0.75)
            else:
                mask_ratio = 0.75

            acceleration = 1.0 / (1.0 - mask_ratio) if mask_ratio < 1.0 else 4.0
            mask_gen = MaskGenerator(seed=epoch)

            mask = mask_gen.generate_mask(
                mask_type=MaskType.VARIABLE_DENSITY,
                shape=input_batch.shape[-2:],
                acceleration_factor=acceleration,
            ).to(input_batch.device)

            masked_kspace = kspace * mask.unsqueeze(0).unsqueeze(0)
            _recovered = ifft2c(masked_kspace)
            input_batch = _recovered.real if _was_real else _recovered

            # Prevent generator from doing standard spatial masking
            kwargs["disable_spatial_masking"] = True

        # The MAE model handles spatial masking internally (unless disabled above)
        outputs = self.env.generator(input_batch, **kwargs)

        # Handle both dict and tensor outputs
        if isinstance(outputs, dict):
            reconstruction = outputs.get("reconstruction", outputs)
        elif isinstance(outputs, tuple):
            # Some models return (reconstruction, ...) tuple
            reconstruction = outputs[0]
        else:
            # Standard tensor output
            reconstruction = outputs

        # [FIX] Use Unified Loss Computer (SSOT)
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        iteration = int(kwargs.get("iteration", 0) or 0)
        loss_output = self.loss_computer.compute(
            pred=reconstruction,
            target=mae_target,
            epoch=epoch,
            iteration=iteration,
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
        val_batch: Any,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation step for MAE pretraining.

        Accepts the standard pipeline calling convention:
        ``strategy.validation_step(val_batch, batch_idx=i)``

        Args:
            val_batch: Batch from validation dataloader (TrainingBatch or dict).
            batch_idx: Index of the current batch (unused but required by interface).
            **kwargs: Additional arguments.

        Returns:
            Dictionary of validation metrics (PSNR, SSIM, MSE).
        """
        # Unpack batch using base class helper (handles TrainingBatch and dict formats)
        input_data, target_data = self._unpack_batch(val_batch)

        # MAE uses target as both input and reconstruction target.
        # Fall back to input if target is None (e.g. unlabelled SSL data).
        source = target_data if target_data is not None else input_data
        if source is None:
            # Last resort: val_batch itself might be a raw tensor
            if isinstance(val_batch, torch.Tensor):
                source = val_batch
            else:
                raise ValueError(
                    f"MAEPretrainingStrategy.validation_step: could not extract "
                    f"tensor from batch of type {type(val_batch).__name__}"
                )

        input_batch = source.to(self.device, non_blocking=True)

        # Reference for metrics is the clean image, captured before the
        # k-space branch overwrites ``input_batch`` with its corruption
        # (otherwise val PSNR grades agreement with the corruption, not
        # reconstruction quality — pitfall #20).
        mae_target = input_batch

        # [Phase 3.3] K-Space Block Masking for Validation
        mask_domain = "image"
        if hasattr(self.config, "task_settings") and isinstance(self.config.task_settings, dict):
            mask_domain = self.config.task_settings.get("mask_domain", mask_domain)
        elif hasattr(self.config.training, "task_settings") and isinstance(
            getattr(self.config.training, "task_settings", None), dict
        ):
            mask_domain = self.config.training.task_settings.get("mask_domain", mask_domain)

        if (
            hasattr(self.config, "training")
            and self.config.training is not None
            and hasattr(self.config.training, "mae")
        ):
            mae_config = self.config.training.mae
            if hasattr(mae_config, "mask_domain"):
                mask_domain = mae_config.mask_domain
            elif isinstance(mae_config, dict):
                mask_domain = mae_config.get("mask_domain", mask_domain)

        gen_kwargs = {}
        if mask_domain == "kspace":
            from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
            from mriforge.infrastructure.physics.sampling import MaskGenerator, MaskType

            # Route through the centered-FFT SSOT (fft2c/ifft2c handle the 4-D
            # [B,C,H,W] MAE batch; the old FFTTransformer.fft2 unpacked a 5-D
            # volume and raised ValueError on every k-space-masking step).
            # Treat a real image as complex for the round-trip.
            _was_real = not torch.is_complex(input_batch)
            _x_complex = (
                torch.complex(input_batch, torch.zeros_like(input_batch))
                if _was_real
                else input_batch
            )
            kspace = fft2c(_x_complex)

            # Prefer the typed training.mae block (validated 0 ≤ ratio < 1);
            # fall back to the legacy task_settings dict, then the default
            # (pitfall #15 — mask_ratio was previously read only from the
            # untyped dict, so a typed/validated value was a silent no-op).
            mae_cfg = self.config.training.mae if self.config.training is not None else None
            if mae_cfg is not None and hasattr(mae_cfg, "mask_ratio"):
                mask_ratio = float(mae_cfg.mask_ratio)
            elif hasattr(self.config, "task_settings") and isinstance(
                self.config.task_settings, dict
            ):
                mask_ratio = self.config.task_settings.get("mask_ratio", 0.75)
            else:
                mask_ratio = 0.75

            acceleration = 1.0 / (1.0 - mask_ratio) if mask_ratio < 1.0 else 4.0

            # Fixed validation seed
            mask_gen = MaskGenerator(seed=42)

            mask = mask_gen.generate_mask(
                mask_type=MaskType.VARIABLE_DENSITY,
                shape=input_batch.shape[-2:],
                acceleration_factor=acceleration,
            ).to(input_batch.device)

            masked_kspace = kspace * mask.unsqueeze(0).unsqueeze(0)
            _recovered = ifft2c(masked_kspace)
            input_batch = _recovered.real if _was_real else _recovered

            gen_kwargs["disable_spatial_masking"] = True

        self.env.generator.eval()
        outputs = self.env.generator(input_batch, **gen_kwargs)

        # Handle different output formats
        if isinstance(outputs, dict):
            reconstruction = outputs.get("reconstruction", outputs)
        elif isinstance(outputs, tuple):
            reconstruction = outputs[0]
        else:
            reconstruction = outputs

        # Standardized Metric Computation
        compute_img_metrics = self.config.validation.scoring.enable_image_metrics

        metrics = {}
        if compute_img_metrics:
            # [Refactor] Use SSOT ValidationMetricsComputer
            try:
                computer = self._get_validation_metrics_computer(self.config)
                computed = computer.compute(reconstruction, mae_target)

                # Apply prefix
                for k, v in computed.items():
                    metrics[f"val_{k}"] = v
            except Exception as e:
                # Do not silently swallow a validation-metric failure: a
                # masked shape/AMP/OOM error here yields a green run with
                # silently-degraded validation (the early-stopping / best
                # monitor then reads missing metrics) — pitfall #10. Log and
                # re-raise so the failure surfaces.
                if hasattr(self, "logging_service"):
                    self.logging_service.log_warning(f"Validation metrics failed: {e}")
                raise

        return metrics
