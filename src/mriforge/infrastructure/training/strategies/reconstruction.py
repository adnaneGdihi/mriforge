"""Reconstruction Training Strategy Module

This module contains Reconstruction training strategies.
"""

# [FORENSIC FIX] - Multislice handling now uses explicit config flag instead of shape heuristics
# to avoid ambiguous channel interpretation. See test_reconstruction_ambiguity.py for validation.

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn

from mriforge.config.schemas.enums import Regime, Task
from mriforge.infrastructure.physics.pinn import PINNModule
from mriforge.infrastructure.training.builders.environment import TrainingEnvironment
from mriforge.infrastructure.training.strategy_helpers import (
    StrategyInitializationHelper,
)
from mriforge.models.capabilities import StrategyCapabilities
from mriforge.models.losses.computers import UnifiedReconstructionLossComputer

from ..utils.training_utils import clamp_to_range
from .loss_folding import (
    declared_loss_weights,
    fold_builder_image_losses,
    inline_managed_with,
)
from .mixins.utils import _callable_accepts_kwarg, pick_present

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


from .base import BaseTrainingStrategy
from .mixins.model_validation import ModelValidationMixin
from .mixins.reconstruction import ReconstructionMixin


class ReconstructionTrainingStrategy(
    ModelValidationMixin, ReconstructionMixin, BaseTrainingStrategy
):
    """Reconstruction training strategy for supervised MRI image reconstruction.

    This strategy implements pure reconstruction-only training without adversarial
    components. It supports multiple loss functions (L1, L2, SSIM, perceptual) and
    optional physics-informed neural network (PINN) constraints for k-space consistency.

    Multislice Support:
    - Uses explicit `data.multislice_enabled` config flag to avoid ambiguous shape heuristics
    - Automatically reshapes multi-slice data from (B, num_slices*2, H, W) to (B*num_slices, 2, H, W)
    - Handles auxiliary tensors (k-space, masks, sensitivities) consistently with slice reshaping

    Typical use cases:
    - Baseline reconstruction models (U-Net, Vision Mamba, Swin Transformer)
    - Transfer learning initialization before adversarial fine-tuning
    - Fast convergence when paired data is available
    - Physics-driven reconstruction with data consistency layers

    Loss Components:
    - **L1**: Mean absolute error (MAE) reconstruction loss
    - **L2**: Mean squared error (MSE) reconstruction loss
    - **SSIM**: Structural similarity index for perceptual quality
    - **Perceptual**: VGG/LPIPS-based perceptual loss
    - **K-space**: Frequency domain reconstruction fidelity
    - **PINN**: Physics-informed constraints (optional)

    Attributes:
        loss_computer: Unified loss computation module handling all reconstruction losses.
        fft_transformer: FFT/IFFT utilities for k-space operations.
        config: Training configuration (read-only).
        device: PyTorch device (CPU or CUDA).

    Config Requirements:
        ```yaml
        training:
          training_mode: reconstruction
        objectives:
          reconstruction:
            lambda_l1: 1.0          # L1 loss weight
            lambda_l2: 0.0          # L2 loss weight (optional)
            lambda_ssim: 0.1        # SSIM loss weight (optional)
            lambda_perceptual: 0.01 # Perceptual loss weight (optional)
        ```

    Raises:
        ValueError: If training_mode is not 'reconstruction'.
        AttributeError: If required config sections are missing.

    Example:
        >>> from mriforge.infrastructure.training.strategies.reconstruction import ReconstructionTrainingStrategy
        >>> from mriforge.infrastructure.training.contexts import TrainingEnvironment
        >>>
        >>> env = TrainingEnvironment(config=TrainingSettings.from_yaml('config.yaml'))
        >>> strategy = ReconstructionTrainingStrategy(env=env)
        >>>
        >>> # Training step
        >>> losses = strategy.train_step(batch, epoch=0, step=100)
        >>> print(losses['g_total_loss'].item())  # Total weighted loss
        >>> print(losses['loss_l1'].item())       # L1 component

    Note:
        This strategy does NOT include:
        - Adversarial losses (use GANTrainingStrategy)
        - VAE KL divergence (use VAETrainingStrategy)
        - Diffusion denoising (use DiffusionTrainingStrategy)
    """

    # Domain axes this paradigm can populate from the batch (via data.expose_*).
    # Widens ConditioningMixin's empty default so model.conditioning passes the
    # Tier-1 conditioning_sources_supported audit on recon arms. severity_vec is
    # deliberately excluded — it is a virtual-fiducial-only quantity.
    _SUPPORTED_CONDITION_SOURCES = (
        "field_strength",
        "scanner_id",
        "site_id",
        "contrast_id",
    )

    #: Registered losses this strategy (or a subclass) computes INLINE, on top of the
    #: universal ``l1``/``l2`` placeholders. A subclass widens this so the term may be
    #: declared on ``losses.image_losses`` — which is what makes it curriculum-
    #: targetable — without :func:`loss_folding.fold_builder_image_losses` applying the
    #: module a second time. Empty here: the base recon strategy folds everything it
    #: does not compute itself.
    _INLINE_MANAGED_EXTRA: ClassVar[tuple[str, ...]] = ()

    #: Workflow tags: structural MRI reconstruction. This is what backs the
    #: ``mri_structural`` regime's LIVE claim in the maturity ledger.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.STRUCTURAL}),
        tasks=frozenset({Task.RECONSTRUCTION}),
    )

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the reconstruction training strategy.

        Args:
            env: The training environment.
            device: Optional torch device.
            **kwargs: Additional args.
        """
        super().__init__(env=env, device=device, **kwargs)

        # Prediction seam (see _compute_losses_impl): the most recent grad-carrying
        # generator output and its loss target, exposed for derived strategies.
        self._last_prediction: torch.Tensor | None = None
        self._last_target: torch.Tensor | None = None

        # Get FFT transformer from env if available, else legacy context
        self.fft_transformer = getattr(self.env, "fft_transformer", None) or getattr(
            self.context, "fft_transformer", None
        )
        if self.fft_transformer is None:
            from mriforge.infrastructure.training.utils.transform_ops import (
                FFTTransformer,
            )

            self.fft_transformer = FFTTransformer(device=self.device)

        # Initialize strategy-specific components using unified loss computer
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config,
            device=self.device,
        )
        # Explicitly set attribute for dynamic resolution if needed by consumer
        self.loss_computer.fft_transformer = self.fft_transformer

        # Try to load metrics adapter for advanced metrics reporting
        try:
            from mriforge.models.trellis.trellis_metrics_adapter import (
                TrellisMetricsAdapter,
            )

            StrategyInitializationHelper.initialize_metrics_adapter(self, TrellisMetricsAdapter)
        except ImportError:  # pragma: no cover - adapter optional
            pass

        # Initialize data consistency layer for physics-aware reconstruction
        physics_config = self.config.physics if hasattr(self.config, "physics") else None
        dc_config = (
            physics_config.data_consistency
            if physics_config and hasattr(physics_config, "data_consistency")
            else None
        )

        use_dc = dc_config.enabled if dc_config and hasattr(dc_config, "enabled") else False
        dc_weight = dc_config.weight if dc_config and hasattr(dc_config, "weight") else 1.0
        dc_method = (
            dc_config.method
            if dc_config and hasattr(dc_config, "method")
            else "projection_2d_consistency"
        )

        StrategyInitializationHelper.initialize_data_consistency(
            self, use_dc=use_dc, dc_weight=dc_weight, dc_method=dc_method
        )

        # Initialize PINN if enabled
        self.pinn_module = None
        if hasattr(self.config, "physics") and hasattr(self.config.physics, "pinn"):
            pinn_config = self.config.physics.pinn
            if pinn_config.enabled:
                from mriforge.infrastructure.physics.pinn import get_pde

                # Pass any extra kwargs from config if needed, for now just basic ones
                pde = get_pde(
                    pinn_config.pde_type,
                    # boundary_condition=pinn_config.boundary_condition # Not used in WaveEquation yet
                )
                self.pinn_module = PINNModule(pde, weight=pinn_config.lambda_pde).to(self.device)

        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize reconstruction-specific components and perform
        validation."""
        # Accept reconstruction and derived strategy modes
        self._verify_strategy_config(
            expected_modes=("reconstruction", "mae_pretraining", "volumetric")
        )
        if self.logging_service:
            self._log_config_features(self.logging_service)

    # _is_implicit_model and _generate_coordinate_grid handled by ReconstructionMixin

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction losses for the training step.

        This method orchestrates the reconstruction training process:
        1.  **Preparation:** Extracts context (mask, sensitivity maps) from the batch.
        2.  **Forward Pass:** Generates reconstruction from the model.
        3.  **Physics Consistency:** Applies data consistency (DC) layers if enabled.
        4.  **Loss Computation:** Calculates L1, Perceptual, Frequency, and PINN losses.

        Args:
            input_batch: The input tensor (e.g., undersampled k-space).
            target_batch: The target tensor (e.g., fully-sampled image).
            epoch: Current epoch index.
            **kwargs: Additional batch context (e.g., `mask`, `sensitivity_maps`).

        Returns:
            A dictionary of computed losses, including `g_total_loss`.
        """
        # Extract batch context (CC=2)
        batch_context = self._prepare_batch_context(input_batch, target_batch, **kwargs)

        # Prepare input tensors and forward kwargs (CC=4)
        lr_image, forward_kwargs = self._prepare_generator_inputs(batch_context, input_batch)

        # Generate predictions with intermediates (CC=3)
        hr_fakes, intermediate_outputs = self._generate_predictions(
            lr_image, forward_kwargs, batch_context
        )

        # Apply physics constraints and compute specialized losses (CC=5)

        # Compute PINN loss if enabled and module is available
        pinn_loss = None
        if self.pinn_module is not None:
            try:
                pinn_loss = self.pinn_module(hr_fakes, batch_context)
                if isinstance(pinn_loss, torch.Tensor) and pinn_loss.requires_grad:
                    # Guard the .item() (a GPU sync) behind the DEBUG level check
                    # so it does NOT run every step at INFO/WARNING — the f-string
                    # argument is otherwise evaluated before log_debug filters it.
                    # backlog_wasted_compute_audit_2026_05_29 TRAIN-9.
                    if self.logging_service and self.logging_service.logger.isEnabledFor(
                        logging.DEBUG
                    ):
                        self.logging_service.log_debug(
                            f"PINN loss computed: {pinn_loss.item():.6f}"
                        )
            except Exception as e:
                # Do not swallow: an *enabled* pinn_module (self.pinn_module is
                # not None) that errors would otherwise silently drop the PINN
                # constraint from the gradient and train as plain reconstruction
                # (pitfall #9/#10). Surface it loudly.
                if self.logging_service:
                    self.logging_service.log_warning(f"PINN loss computation failed: {e!s}")
                raise

        # Get losses from training environment (built by LossBuilder from config)
        env_losses: dict[str, Any] = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}
        elif hasattr(self, "context") and self.context and hasattr(self.context, "loss_fn"):
            env_losses = self.context.loss_fn or {}

        target = batch_context["hr"]

        # Evidential UNets output 4 parameters (mean, var, alpha, beta)
        # We only want to compute structural losses (L1, Perceptual) on the predicted mean
        if (
            self.config.model.model_type == "evidential_unet"
            and hr_fakes.shape[1] > target.shape[1]
        ):
            eval_fakes = hr_fakes[:, : target.shape[1]]
        else:
            eval_fakes = hr_fakes

        # Use Unified Loss Computer.
        # Thread the live ``iteration`` (supplied by the trainer via
        # train_step → g_closure → kwargs) into the computer so its
        # spatial-loss warm-up gate (l1/perceptual/adversarial held at
        # weight 0 while iteration < warmup_iterations) actually advances.
        # Dropping it froze those terms at 0 for the entire run — a silent
        # facade (CLAUDE.md pitfall #16).
        iteration = int(kwargs.get("iteration", 0) or 0)
        # Dynamic loss-schedule overrides are published to ``self.loss_computer``
        # by the paradigm-agnostic ``BaseTrainingStrategy.sync_scheduled_loss_weights``
        # (called by the training loop each step) -- no per-strategy copy needed.
        loss_output = self.loss_computer.compute(
            pred=eval_fakes,
            target=target,
            epoch=epoch,
            iteration=iteration,
            losses_dict=env_losses,
            pinn_loss=pinn_loss,
            intermediate_outputs=intermediate_outputs,
        )

        total_loss = loss_output.total
        components = loss_output.components

        # Build result dict from computed losses
        self._loss_dict_reuse.clear()
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        self._loss_dict_reuse["g_total_loss"] = total_loss
        self._loss_dict_reuse.update(components)

        # Ensure 'loss' exists (alias for g_total_loss)
        if "loss" not in self._loss_dict_reuse:
            self._loss_dict_reuse["loss"] = total_loss

        # Compute training metrics (PSNR, SSIM, MAE) for monitoring. Use the live
        # loop iteration (already resolved above) — ``self.env.step`` is frozen at
        # 0, so it would defeat the train_metric_interval throttle and recompute
        # SSIM/PSNR/MAE every step (pitfall #16).
        train_metrics = self._compute_training_metrics(
            pred=eval_fakes,
            target=target_batch,
            config=self.config,
            current_step=iteration,
        )
        # Convert metrics to tensor for consistent return type
        for k, v in train_metrics.items():
            self._loss_dict_reuse[k] = torch.tensor(v, device=self.device)

        # Prediction seam for subclasses (e.g. SyntheticPathologyAugStrategy) that
        # need the grad-carrying generator output to compute region-weighted losses.
        # The model prediction is never written back into the passed batch dict, so
        # subclasses read it from here instead of a nonexistent batch["prediction"]
        # key. We deliberately expose the *grad-carrying* eval_fakes (NOT detached)
        # and the loss target so derived strategies can fold an extra term into
        # g_total_loss that still backprops to the generator.
        self._last_prediction = eval_fakes
        self._last_target = target

        return self._loss_dict_reuse

    def _declared_loss_weights(self) -> dict[str, float]:
        """Map declarative loss name -> weight from ``config.losses.*_losses``.

        Thin delegate to the SSOT :func:`loss_folding.declared_loss_weights` (shared
        with the GAN-based cross-field arms).
        """
        return declared_loss_weights(self.config)

    def _apply_builder_image_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        components: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Fold the builder-composed declarative image losses onto an inline total.

        This is the loss-SSOT seam for strategies that override
        :meth:`_compute_losses_impl` and compute their objective inline — for those
        the declarative ``losses:`` block would otherwise be an inert decoy (pitfall
        #16). Delegates the fold to :func:`loss_folding.fold_builder_image_losses`
        (the SSOT), passing the ``loss_schedule`` curriculum overrides
        (``loop_state.loss_weight_overrides``) so a ramp on hfen/lpips actually
        modulates the folded term. Returns ``None`` when no builder losses are
        configured, so inline-only arms stay byte-identical.
        """
        env = getattr(self, "env", None)
        env_losses = (getattr(env, "losses", None) or {}) if env is not None else {}
        if not env_losses:
            return None
        loop_state = getattr(self, "loop_state", None)
        scheduled = getattr(loop_state, "loss_weight_overrides", None) or {}
        return fold_builder_image_losses(
            env_losses,
            self._declared_loss_weights(),
            scheduled,
            pred,
            target,
            components,
            inline_managed=inline_managed_with(*self._INLINE_MANAGED_EXTRA),
        )

    # _prepare_batch_context and _prepare_generator_inputs handled by ReconstructionMixin
    # We define alias methods to satisfy internal calls or override Base hooks if needed

    def _prepare_batch_context(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate to ReconstructionMixin."""
        return self._prepare_batch_context_reconstruction(input_batch, target_batch, **kwargs)

    def _prepare_generator_inputs(
        self,
        batch_context: dict[str, Any],
        input_batch: torch.Tensor,
        *,
        validation: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Prepare input tensors and forward kwargs (CC=4 extracted)."""
        multimodal_inputs = batch_context.get("multimodal_inputs")
        measured_kspace = batch_context.get("measured_kspace")
        use_dc = batch_context["use_dc"]

        # Handle multimodal inputs
        if multimodal_inputs is not None:
            input_tensors = list(multimodal_inputs.values())
            lr_image = torch.cat(input_tensors, dim=1)
        # NeRF/Implicit models: generate coordinate grid instead of using image
        elif self._is_implicit_model():
            # Get target shape from hr in batch_context or input_batch
            hr = batch_context.get("hr")
            if hr is not None:
                B, C, H, W = hr.shape
            else:
                B, _, H, W = input_batch.shape  # Fallback to input_batch shape

            # Generate coordinate grid [B, 2, H, W]
            lr_image = self._generate_coordinate_grid(B, H, W, input_batch.device)

            if hasattr(self, "logging_service") and self.logging_service:
                self.logging_service.log_debug(
                    f"[NeRF] Generated coordinate grid for implicit model: {lr_image.shape}"
                )
        else:
            lr_image = input_batch

        # Convert k-space to image if needed
        if use_dc and multimodal_inputs is None:
            kspace_for_input = pick_present(measured_kspace, input_batch)
            trajectory = batch_context.get("trajectory")
            dcf = batch_context.get("dcf")

            if trajectory is not None:
                # Handle Non-Cartesian Trajectories (Spiral/Radial)
                # Standard ifft2c fails for (B, C, N) inputs. We must use NUFFT Adjoint.
                try:
                    import torchkbnufft as tkbn
                except ImportError:
                    raise ImportError("torchkbnufft is required for non-Cartesian reconstruction.")

                # Ensure dimensions are correct for tkbn
                # Handle potential extra batch dimension from collation (B, 1, ...)
                if kspace_for_input.ndim == 4 and kspace_for_input.shape[1] == 1:
                    kspace_for_input = kspace_for_input.squeeze(1)

                if trajectory is not None and trajectory.ndim == 4 and trajectory.shape[1] == 1:
                    trajectory = trajectory.squeeze(1)

                dcf = batch_context.get("dcf")
                if dcf is not None and dcf.ndim == 3 and dcf.shape[1] == 1:
                    dcf = dcf.squeeze(1)

                # kspace: (B, C, N) -> (B, C, N) [OK]
                # traj: (B, N, 2) -> (B, 2, N) [Transpose needed if loaded as (N, 2)]
                # traj: (B, 2, N) [OK]
                if trajectory.ndim == 3 and trajectory.shape[-1] == 2:
                    ktraj = trajectory.transpose(1, 2)  # (B, 2, N)
                else:
                    ktraj = trajectory

                # Get expected image size
                im_size = (256, 256)
                if batch_context.get("hr") is not None:
                    im_size = batch_context["hr"].shape[-2:]
                elif hasattr(self.env.generator, "im_size"):
                    im_size = self.env.generator.im_size

                # Check cache for adjoint operator
                if not hasattr(self, "_cached_adj_op") or self._cached_adj_op.im_size != im_size:
                    self._cached_adj_op = tkbn.KbNufftAdjoint(im_size=im_size, device=self.device)

                # Apply Density Compensation Function if available
                # This pre-weights k-space for better conditioning
                # dcf already retrieved above
                if dcf is not None:
                    # Ensure broadcasting: kspace (B, C, N) * dcf (B, 1, N)
                    if dcf.ndim == 2:
                        dcf = dcf.unsqueeze(1)
                    kspace_for_input = kspace_for_input * dcf

                # Perform Adjoint Operation -> (B, C, H, W)
                # tkbn handles batching if trajectory interacts correctly
                lr_image = self._cached_adj_op(kspace_for_input, ktraj)

                if hasattr(self, "logging_service") and self.logging_service:
                    self.logging_service.log_debug(
                        f"Performed NUFFT Adjoint: kspace {kspace_for_input.shape} -> image {lr_image.shape}"
                    )

            else:
                # Cartesian Fallback
                # Since we are in a mixin, we expect self to have fft_transformer
                # But BaseTrainingStrategy does NOT guarantee fft_transformer attribute.
                # reconstruction.py sets it in __init__.
                # We should check for it safely.
                fft_transformer = getattr(self, "fft_transformer", None)
                if fft_transformer:
                    lr_image = fft_transformer.ifft2c(kspace_for_input)
                else:
                    # Fallback to creating one if not present (defensive)
                    from mriforge.infrastructure.training.utils.transform_ops import (
                        FFTTransformer,
                    )

                    transformer = FFTTransformer(device=self.device)
                    lr_image = transformer.ifft2c(kspace_for_input)

        # Track if we reshapped slices (for reshaping auxiliary tensors)
        num_slices = 1
        was_reshaped = False

        # Handle multi-slice k-space data
        # Use explicit config flag instead of shape-based inference
        config = self.config
        model_in_channels = config.model.in_channels if hasattr(config.model, "in_channels") else 2
        multislice_enabled = config.data.multislice_enabled
        is_multislice = (
            multislice_enabled
            and model_in_channels == 2
            and lr_image.ndim == 4
            and lr_image.shape[1] > 2
            and lr_image.shape[1] % 2 == 0
        )

        if is_multislice:
            # Multi-slice k-space data: (B, num_slices*2, H, W)
            # where channels are [slice1_real, slice1_imag, ...]
            # Reshape to (B*num_slices, 2, H, W) for per-slice processing
            B, channels, H, W = lr_image.shape
            num_slices = channels // 2
            was_reshaped = True
            # Reshape: (B, num_slices*2, H, W) -> (B, num_slices, 2, H, W)
            # -> (B*num_slices, 2, H, W)
            # [PERF] Use contiguous().view() per debugger.md Rule 11
            lr_image = lr_image.contiguous().view(B, num_slices, 2, H, W)
            # [PERF B10] Removed no-op permute(0,1,2,3,4) (identity permutation)
            lr_image = lr_image.contiguous().view(B * num_slices, 2, H, W)
            if hasattr(self, "logging_service") and self.logging_service:
                self.logging_service.log_debug(
                    f"Multi-slice reshape: input (B={B}, channels={channels}, "
                    f"H={H}, W={W}) -> output (B={B * num_slices}, 2, {H}, {W}), "
                    f"num_slices={num_slices}"
                )
        # Handle complex k-space data: (B, num_slices, H, W) complex
        # -> (B*num_slices, 2, H, W) real/imag
        elif model_in_channels == 2 and torch.is_complex(lr_image) and lr_image.ndim == 4:
            # (B, num_slices, H, W) complex -> (B, num_slices, H, W, 2)
            real_imag = torch.view_as_real(lr_image)
            B, num_slices_complex, H, W, _ = real_imag.shape
            num_slices = num_slices_complex
            was_reshaped = True
            # Permute to (B, num_slices, 2, H, W) then reshape
            # [PERF] Use contiguous().view() per debugger.md Rule 11
            real_imag = real_imag.permute(0, 1, 4, 2, 3)  # (B, num_slices, 2, H, W)
            lr_image = real_imag.contiguous().view(B * num_slices, 2, H, W)

        # If we reshaped the main input, also reshape auxiliary tensors to match
        if was_reshaped and num_slices > 1:
            # Reshape measured_kspace if present
            if measured_kspace is not None:
                # measured_kspace shape: (B, C, H, W) or (B, H, W)
                # Only reshape if it can be divided by num_slices
                if measured_kspace.ndim == 4:
                    B_orig, C, H_k, W_k = measured_kspace.shape
                    # Only reshape if C is divisible by num_slices
                    # (accounts for multi-coil data)
                    if C % num_slices == 0:
                        C_per_slice = C // num_slices
                        # (B, C, H, W) -> (B*num_slices, C_per_slice, H, W)
                        # [PERF] Use contiguous().view() per debugger.md Rule 11
                        measured_kspace = measured_kspace.contiguous().view(
                            B_orig, num_slices, C_per_slice, H_k, W_k
                        )
                        # [PERF B10] Removed no-op permute(0,1,2,3,4) (identity permutation)
                        measured_kspace = measured_kspace.contiguous().view(
                            B_orig * num_slices, C_per_slice, H_k, W_k
                        )
                        batch_context["measured_kspace"] = measured_kspace
                    else:
                        pass
                elif measured_kspace.ndim == 3:
                    # (B, H, W) -> (B*num_slices, H, W) by repeating for each slice
                    B_orig, H_k, W_k = measured_kspace.shape
                    measured_kspace = measured_kspace.repeat_interleave(num_slices, dim=0)
                    batch_context["measured_kspace"] = measured_kspace

            # Reshape mask if present
            if batch_context.get("mask") is not None:
                mask = batch_context["mask"]
                # mask shape: (B, H, W) or (B, 1, H, W)
                if mask.ndim == 4 and mask.shape[1] == 1:
                    # (B, 1, H, W) -> (B*num_slices, 1, H, W)
                    # by repeating for each slice
                    B_orig = mask.shape[0]
                    mask = mask.repeat(num_slices, 1, 1, 1)
                    batch_context["mask"] = mask
                elif mask.ndim == 3:
                    # (B, H, W) -> (B*num_slices, H, W) by repeating for each slice
                    B_orig = mask.shape[0]
                    mask = mask.repeat(num_slices, 1, 1)
                    batch_context["mask"] = mask

            # Reshape coil_sensitivities if present
            if batch_context.get("coil_sensitivities") is not None:
                coil_sens = batch_context["coil_sensitivities"]
                # coil_sens shape: (B, C, H, W)
                if coil_sens.ndim == 4:
                    # (B, C, H, W) -> (B*num_slices, C, H, W)
                    # by repeating for each slice
                    B_orig = coil_sens.shape[0]
                    C = coil_sens.shape[1]
                    coil_sens = coil_sens.repeat(num_slices, 1, 1, 1)
                    batch_context["coil_sensitivities"] = coil_sens

            # Reshape hr if present
            if batch_context.get("hr") is not None:
                hr = batch_context["hr"]
                # hr shape: (B, C_hr, H, W)
                if hr.ndim == 4:
                    B_orig, C_hr, H_hr, W_hr = hr.shape
                    if C_hr == num_slices:
                        # Case: HR has 1 channel per slice (e.g. magnitude)
                        # (B, num_slices, H, W) -> (B, num_slices, 1, H, W)
                        # -> (B*num_slices, 1, H, W)
                        hr = hr.unsqueeze(2)  # (B, num_slices, 1, H, W)
                        hr = hr.contiguous().view(B_orig * num_slices, 1, H_hr, W_hr)
                        batch_context["hr"] = hr
                    elif C_hr == num_slices * 2:  # Assuming 2 output channels per slice?
                        # (B, num_slices*2, H, W) -> (B, num_slices, 2, H, W)
                        # -> (B*num_slices, 2, H, W)
                        hr = hr.contiguous().view(B_orig, num_slices, 2, H_hr, W_hr)
                        # [PERF B10] Removed no-op permute(0,1,2,3,4) (identity permutation)
                        hr = hr.contiguous().view(B_orig * num_slices, 2, H_hr, W_hr)
                        batch_context["hr"] = hr

        # Build forward kwargs
        forward_kwargs: dict[str, Any] = {}
        generator = cast(nn.Module, self.env.generator)
        generator_forward = getattr(generator, "forward", None)

        if generator_forward is not None:
            if measured_kspace is not None and _callable_accepts_kwarg(
                generator_forward, "kspace_measured"
            ):
                forward_kwargs["kspace_measured"] = batch_context.get(
                    "measured_kspace", measured_kspace
                )
            if batch_context.get("mask") is not None and _callable_accepts_kwarg(
                generator_forward, "mask"
            ):
                forward_kwargs["mask"] = batch_context["mask"]
            if batch_context.get("coil_sensitivities") is not None and _callable_accepts_kwarg(
                generator_forward, "sensitivity_maps"
            ):
                forward_kwargs["sensitivity_maps"] = batch_context["coil_sensitivities"]
            if batch_context.get("trajectory") is not None and _callable_accepts_kwarg(
                generator_forward, "k_trajectory"
            ):
                forward_kwargs["k_trajectory"] = batch_context["trajectory"]

            # Inject slice_indices if model requires it (e.g., for MedGS folded Gaussians)
            if batch_context.get("slice_indices") is not None and _callable_accepts_kwarg(
                generator_forward, "slice_indices"
            ):
                forward_kwargs["slice_indices"] = batch_context["slice_indices"]

            # Inject timesteps if model requires it (e.g. for time-conditioned reconstruction or LPD)
            if _callable_accepts_kwarg(generator_forward, "timesteps"):
                # Use batch size from lr_image
                B = lr_image.shape[0]
                if validation:
                    # Deterministic eval time so validation metrics are
                    # reproducible across runs/epochs — a fresh random draw each
                    # val pass made the early-stopping / best monitor noisy.
                    timesteps = torch.zeros((B,), device=lr_image.device)
                else:
                    # Random timesteps [0, 1] for training.
                    timesteps = torch.rand((B,), device=lr_image.device)
                forward_kwargs["timesteps"] = timesteps

            # Field-strength conditioning for cross-field renderers
            # (``AnatomyFieldRenderer`` / ``CrossFieldRenderer`` & friends) that
            # take ``field_strength`` as a keyword-only, REQUIRED argument. This is
            # the LIVE seam used by both training (``_compute_losses``) and
            # validation (``_validation_forward`` -> here); the field-aware override
            # in ``xfield_fm_strategy`` re-sets it after ``super()`` so those arms
            # are unaffected. Prefer the TARGET field (render at the requested
            # field) over the source. Signature-gated via ``_callable_accepts_kwarg``
            # so plain models are untouched, and NEVER defaulted — a model that
            # requires ``field_strength`` with no field in the batch still raises,
            # which is the honest signal (pitfall #9). Fixes the mrixfields
            # ``b17_dice_risk_calibration`` crash: a calibration arm inherits this
            # generic validation forward with no field-aware override, so before
            # this injection the renderer raised "missing 1 required keyword-only
            # argument: 'field_strength'" -> zero successful validation batches
            # (CLAUDE.md #10). NOTE: the look-alike ``_prepare_generator_inputs_
            # reconstruction`` in mixins/reconstruction.py carries the same block
            # but has no callers — THIS method is the one the run actually uses.
            if _callable_accepts_kwarg(generator_forward, "field_strength"):
                fs = batch_context.get("field_strength_target", batch_context.get("field_strength"))
                if fs is not None:
                    forward_kwargs["field_strength"] = fs
            if batch_context.get("contrast_id") is not None and _callable_accepts_kwarg(
                generator_forward, "contrast_id"
            ):
                forward_kwargs["contrast_id"] = batch_context["contrast_id"]

        return lr_image, forward_kwargs

    def _generate_predictions(
        self,
        lr_image: torch.Tensor,
        forward_kwargs: dict[str, Any],
        batch_context: dict[str, Any],
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Generate predictions with optional intermediates (CC=3 extracted)."""
        generator = cast(nn.Module, self.env.generator)
        # Strategy-level FiLM conditioning (model.conditioning). No-op object
        # passthrough when disabled; modulates lr_image on the configured domain
        # axes (contrast_id / field_strength / scanner_id / site_id) otherwise.
        lr_image = self._apply_input_conditioning(lr_image, batch_context)
        hr_fakes = generator(lr_image, **forward_kwargs)

        # Handle models that return tuples (e.g., DisentangledMRI returns (output, mu, logvar))
        if isinstance(hr_fakes, tuple):
            # First element is the main output, rest are auxiliary outputs (mu, logvar, etc.)
            hr_fakes = hr_fakes[0]

        intermediate_outputs = None
        if hasattr(generator, "forward_with_intermediates"):
            try:
                intermediate_outputs = generator.forward_with_intermediates(
                    lr_image, **forward_kwargs
                )
                hr_fakes = intermediate_outputs[-1]
                # Also handle tuple case for intermediates
                if isinstance(hr_fakes, tuple):
                    hr_fakes = hr_fakes[0]
            except Exception as e:
                # A model that advertises ``forward_with_intermediates`` but
                # errors would otherwise silently lose its deep-supervision loss
                # path and train on the plain forward only (pitfall #10). Log
                # and re-raise so the broken intermediate path is surfaced.
                if self.logging_service:
                    self.logging_service.log_warning(
                        f"Failed to get intermediate outputs from generator: {e}",
                        model_type=self.config.model.model_type,
                    )
                raise

        if getattr(self.config.training, "enforce_output_range", False):
            hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

        # [DC CONSOLIDATION] DC is now applied internally by the generator if enabled.
        # This prevents "double DC" application and maintains SSOT in the model layer.
        pass

        return hr_fakes, intermediate_outputs

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Use ReconstructionMixin to generate validation predictions."""
        lr_image, forward_kwargs = self._prepare_generator_inputs(
            batch_context, input_batch, validation=True
        )
        generator = cast(nn.Module, self.env.generator)
        lr_image = self._apply_input_conditioning(lr_image, batch_context)
        hr_fakes = generator(lr_image, **forward_kwargs)

        return hr_fakes
