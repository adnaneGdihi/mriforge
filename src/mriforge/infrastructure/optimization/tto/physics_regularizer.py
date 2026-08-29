"""Physics regularization for Test-Time Optimization."""

import logging

import torch

from mriforge.infrastructure.optimization.constraints.bloch import BlochConstraint
from mriforge.infrastructure.optimization.regularizers.divergence import (
    DivergenceFreeRegularizer,
)
from mriforge.infrastructure.optimization.regularizers.knn_graph import KNNGraphRegularizer

from .parameter_extractor import OptimizableParameters

logger = logging.getLogger(__name__)


class PhysicsRegularizer:
    """Applies physics-based regularization and constraints during TTO.

    Supports 4 complementary regularization types:
    1. Sparsity: Encourage zero opacity for unnecessary Gaussians
    2. KNN Local Consistency: Smooth feature variation across neighboring Gaussians
    3. Bloch Constraints: Enforce MRI physics (T1/T2 relationships)
    4. Divergence-Free: Enforce incompressible flow (4D cardiac MRI)
    """

    def __init__(
        self,
        lambda_sparsity: float = 0.0,
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
            lambda_sparsity: Weight for opacity sparsity loss. 0 disables.
            lambda_knn: Weight for KNN graph regularization. 0 disables.
            knn_k: Number of neighbors for KNN graph. Ignored if lambda_knn=0.
            lambda_bloch: Weight for Bloch constraint. 0 disables.
            bloch_t1_idx: Index of T1 parameter in feature channels. Used if lambda_bloch>0.
            bloch_t2_idx: Index of T2 parameter in feature channels. Used if lambda_bloch>0.
            lambda_div_free: Weight for divergence-free regularizer. 0 disables.
            div_vel_indices: Indices of velocity components in output. Used if lambda_div_free>0.
            device: Computation device.
        """
        self.lambda_sparsity = lambda_sparsity
        self.lambda_knn = lambda_knn
        self.lambda_bloch = lambda_bloch
        self.lambda_div_free = lambda_div_free
        self.bloch_t1_idx = bloch_t1_idx
        self.bloch_t2_idx = bloch_t2_idx
        self.div_vel_indices = div_vel_indices
        self.device = device

        # Initialize optional regularizers
        self.knn_regularizer = KNNGraphRegularizer(k=knn_k).to(device) if lambda_knn > 0 else None
        self.bloch_constraint = BlochConstraint().to(device) if lambda_bloch > 0 else None
        self.div_regularizer = (
            DivergenceFreeRegularizer(mode="fourier").to(device) if lambda_div_free > 0 else None
        )

    def compute_regularization_loss(
        self,
        params: OptimizableParameters,
        means_eval: torch.Tensor,
        kspace_pred: torch.Tensor,
        k_coords: torch.Tensor,
        dc_loss: torch.Tensor,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total regularization loss from all enabled constraints.

        Args:
            params: Optimizable parameters (for opacity and features).
            means_eval: Evaluated means (N, 3) for spatial reference.
            kspace_pred: Predicted k-space for divergence-free term.
            k_coords: K-space coordinates for divergence-free term.
            dc_loss: Data consistency loss (for logging relative magnitudes).
            step: Current optimization step (for logging).

        Returns:
            tuple: (total_regularization_loss, loss_dict)
                where loss_dict contains individual loss components for monitoring.
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_dict = {}

        # 1. Sparsity loss: encourage zero opacity for unnecessary Gaussians
        if self.lambda_sparsity > 0 and params.opacity is not None:
            sparsity_loss = torch.mean(torch.abs(params.opacity))
            weighted_loss = self.lambda_sparsity * sparsity_loss
            total_loss = total_loss + weighted_loss
            loss_dict["sparsity"] = sparsity_loss.item()

        # 2. KNN graph regularization: smooth feature variation locally
        if self.knn_regularizer is not None and params.features is not None:
            knn_loss = self.knn_regularizer(positions=means_eval, attributes=params.features)
            if isinstance(knn_loss, torch.Tensor):
                weighted_loss = self.lambda_knn * knn_loss
                total_loss = total_loss + weighted_loss
                loss_dict["knn"] = knn_loss.item()
            else:
                loss_dict["knn"] = float(knn_loss)

        # 3. Bloch constraint: enforce MRI physics (T1/T2 relationships)
        if (
            self.bloch_constraint is not None
            and params.features is not None
            and params.features.shape[-1] >= 2
        ):
            t1_vals = params.features[..., self.bloch_t1_idx]
            t2_vals = params.features[..., self.bloch_t2_idx]
            bloch_loss = self.bloch_constraint(t1_vals, t2_vals)
            weighted_loss = self.lambda_bloch * bloch_loss
            total_loss = total_loss + weighted_loss
            loss_dict["bloch"] = bloch_loss.item()

        # 4. Divergence-free regularization: enforce incompressible flow (4D cardiac)
        if self.div_regularizer is not None:
            div_loss = self.div_regularizer(
                parameter_map=kspace_pred,
                velocity_indices=self.div_vel_indices,
                k_trajectory=k_coords,
            )
            if isinstance(div_loss, torch.Tensor):
                weighted_loss = self.lambda_div_free * div_loss
                total_loss = total_loss + weighted_loss
                loss_dict["divergence_free"] = div_loss.item()
            else:
                loss_dict["divergence_free"] = float(div_loss)

        # Log at first step
        if step == 0 and any(self.lambda_sparsity > 0 for _ in [1]):  # Log weights
            logger.info(
                f"TTO regularization weights: "
                f"sparsity={self.lambda_sparsity}, knn={self.lambda_knn}, "
                f"bloch={self.lambda_bloch}, div_free={self.lambda_div_free}"
            )

        return total_loss, loss_dict

    def post_step_constraints(self, params: OptimizableParameters) -> None:
        """Apply hard constraints after each optimization step.

        Args:
            params: Optimizable parameters to constrain in-place.
        """
        # Clamp opacity to [0, 1]
        if params.opacity is not None:
            params.opacity.data.clamp_(0, 1)
