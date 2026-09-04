"""Test-Time Optimizer for physics-guaranteed Gaussian splatting."""

import logging

import torch
import torch.nn as nn
import torch.optim as optim

from spectramr.infrastructure.physics.implementations.gaussian_fourier_op import (
    GaussianCloud,
    GaussianFourierProjector,
)

from .gaussian_updater import GaussianUpdater
from .kspace_renderer import KSpaceRenderer
from .parameter_extractor import ParameterExtractor
from .physics_regularizer import PhysicsRegularizer

logger = logging.getLogger(__name__)


class TestTimeOptimizer:
    """
    Test-Time Optimizer (TTO) for Physics-Guaranteed Generative Splatting.

    Refines the Gaussian primitives directly against the measured k-space data
    using the Differentiable Fourier-Gaussian Projector (DFGP).

    Architecture: Uses composition of 4 specialized components:
    - ParameterExtractor: Handles multi-format Gaussian cloud parsing
    - KSpaceRenderer: Manages differentiable k-space rendering
    - PhysicsRegularizer: Applies all 4 physics constraints
    - GaussianUpdater: Updates original Gaussian with optimized params

    This composition reduces the main refine() method from CC=45 (Grade F)
    to CC=11 (Grade C), a 75% complexity reduction.
    """

    def __init__(
        self,
        projector: GaussianFourierProjector,
        num_steps: int = 50,
        optimizer_lr: float = 0.01,
        lambda_sparsity: float = 0.001,
        lambda_knn: float = 0.0,
        knn_k: int = 5,
        lambda_bloch: float = 0.0,
        bloch_t1_idx: int = 0,
        bloch_t2_idx: int = 1,
        lambda_div_free: float = 0.0,
        div_vel_indices: tuple[int, int, int] = (0, 1, 2),
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            projector: The differentiable renderer (DFGP).
            num_steps: Number of optimization steps.
            optimizer_lr: Learning rate for Gaussian parameters.
            lambda_sparsity: Weight for opacity sparsity regularization.
            lambda_knn: Weight for KNN graph regularization.
            knn_k: Number of neighbors for KNN graph regularization.
            lambda_bloch: Weight for Bloch constraint (T1/T2 physical plausibility).
            bloch_t1_idx: Index of feature dimension corresponding to T1.
            bloch_t2_idx: Index of feature dimension corresponding to T2.
            lambda_div_free: Weight for Divergence-Free regularizer (4D Flow).
            div_vel_indices: Indices of velocity components in output channels.
        """
        self.projector = projector
        self.num_steps = num_steps
        self.optimizer_lr = optimizer_lr
        self.device = device

        # Ensure projector is on correct device
        if isinstance(self.projector, nn.Module):
            self.projector.to(device)

        # Initialize physics regularizer (handles all constraints)
        self.regularizer = PhysicsRegularizer(
            lambda_sparsity=lambda_sparsity,
            lambda_knn=lambda_knn,
            knn_k=knn_k,
            lambda_bloch=lambda_bloch,
            bloch_t1_idx=bloch_t1_idx,
            bloch_t2_idx=bloch_t2_idx,
            lambda_div_free=lambda_div_free,
            div_vel_indices=div_vel_indices,
            device=device,
        )

        # Initialize parameter extraction and rendering helpers
        self.param_extractor = ParameterExtractor(device=device)
        self.renderer = KSpaceRenderer(device=device)

    def refine(
        self,
        gaussians: GaussianCloud,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
    ) -> tuple[GaussianCloud, list[float]]:
        """Refine Gaussians through test-time optimization.

        Iteratively optimizes Gaussian parameters to minimize data consistency loss
        subject to physics-based regularization and hard constraints.

        Args:
            gaussians: Gaussian cloud to optimize (various formats supported).
            kspace_measured: Target k-space measurements (B, C, ...).
            mask: Optional k-space undersampling mask, same shape as kspace_measured.
            sensitivity_maps: Optional multi-coil sensitivity maps (not used in current version).
            trajectory: Optional non-Cartesian k-space trajectory (M, 3).
                       If None, assumes Cartesian grid.

        Returns:
            tuple: (refined_gaussians, loss_history)
                - refined_gaussians: Updated Gaussian cloud
                - loss_history: List of total loss values per optimization step

        Raises:
            Logs warning and returns input unchanged if gaussians cannot be parsed.
        """
        # Move inputs to device
        kspace_measured = kspace_measured.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if sensitivity_maps is not None:
            sensitivity_maps = sensitivity_maps.to(self.device)
        if trajectory is not None:
            trajectory = trajectory.to(self.device)

        # Step 1: Extract optimizable parameters from Gaussian cloud
        params = self.param_extractor.extract(gaussians)
        if not params.is_valid():
            logger.warning(
                f"Input is not a valid Gaussian object (type={type(gaussians).__name__}), "
                "skipping TTO"
            )
            return gaussians, []

        # Collect all parameters for optimizer
        params_to_optimize = params.to_list()
        optimizer = optim.Adam(params_to_optimize, lr=self.optimizer_lr)
        loss_history = []

        # The k-space coordinate grid depends only on the (constant)
        # measured-k-space shape and trajectory, so build it ONCE rather
        # than rebuilding a meshgrid every optimization step.
        k_coords = self.renderer._prepare_kspace_coords(kspace_measured.shape, trajectory)

        # Step 2: Optimization loop
        # NOTE: The inner optimization creates its OWN computation graph.
        # We detach from the outer training graph to prevent
        # "backward through graph a second time" errors when the outer
        # training loop calls loss.backward().
        with torch.enable_grad():
            for step in range(self.num_steps):
                optimizer.zero_grad()

                # 2a. Render k-space from current parameters
                means_eval = params.means[:, 0, :] if params.means.dim() == 3 else params.means
                kspace_pred = self.renderer.render(
                    params, kspace_measured.shape, trajectory=trajectory
                )

                # 2b. Match target shape (batch size and channels)
                kspace_pred = self.renderer.match_target_shape(kspace_pred, kspace_measured)

                # 2c. Apply mask if provided
                if mask is not None:
                    kspace_target = kspace_measured * mask
                    if kspace_pred.shape[-2:] == mask.shape[-2:]:
                        kspace_pred = kspace_pred * mask
                else:
                    kspace_target = kspace_measured

                # 2d. Compute data consistency loss
                dc_loss = torch.nn.functional.mse_loss(kspace_pred, kspace_target)

                # 2e. Compute physics regularization (k_coords hoisted above).
                # The per-component loss_dict is monitoring-only and unused
                # here, so we don't bind it (its .item() syncs only fire when
                # a regularizer is enabled).
                reg_loss, _ = self.regularizer.compute_regularization_loss(
                    params=params,
                    means_eval=means_eval,
                    kspace_pred=kspace_pred,
                    k_coords=k_coords,
                    dc_loss=dc_loss,
                    step=step,
                )

                # 2f. Total loss = data consistency + regularization
                total_loss = dc_loss + reg_loss

                # Logging
                if step == 0:
                    logger.info(
                        f"TTO Step 0: kspace_pred={kspace_pred.shape}, "
                        f"target={kspace_measured.shape}, dc_loss={dc_loss.item():.4f}"
                    )

                # Check for NaN
                if torch.isnan(total_loss):
                    logger.error(f"NaN detected in total_loss at step {step}")
                    break

                # 2g. Backward pass and optimization step
                # retain_graph=True allows multiple backward passes through
                # shared intermediate tensors (e.g. projector parameters)
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                optimizer.step()

                loss_history.append(total_loss.item())

                # 2h. Apply hard constraints
                self.regularizer.post_step_constraints(params)

        # Step 3: Update original Gaussian cloud with optimized parameters
        refined_gaussians = GaussianUpdater.update(gaussians, params)
        return refined_gaussians, loss_history
