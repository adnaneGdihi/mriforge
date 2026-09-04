"""3D Gaussian Splatting Reconstructor.

Integrates GaussianCloud and AdaptiveController for MRI reconstruction.
"""

import torch.nn as nn
from jaxtyping import Complex, Float
from torch import Tensor

from spectramr.models.registry import register_model

from .adaptive_control import AdaptiveController
from .gaussian_primitives import GaussianCloud


@register_model(name="gs_mri", training_mode="reconstruction")
class GSMRReconstructor(nn.Module):
    """End-to-End Reconstructor using 3D Gaussians."""

    def __init__(
        self,
        volume_shape: tuple[int, int, int] = (128, 128, 128),
        num_gaussians: int = 1000,
        init_scale: float = 0.1,
        adaptive_hparams: dict | None = None,
    ):
        """__init__.

        Args:
            volume_shape (tuple[int, int, int]): Description.
            num_gaussians (int): Description.
            init_scale (float): Description.
            adaptive_hparams (Optional[dict]): Description.
        """
        super().__init__()
        self.cloud = GaussianCloud(
            num_gaussians=num_gaussians,
            volume_shape=volume_shape,
            init_scale=init_scale,
        )

        hparams = adaptive_hparams or {}
        self.controller = AdaptiveController(**hparams)

    def forward(
        self,
        k_coords: Float[Tensor, "B K 3"],
        iter_step: int = 0,
        training: bool = True,
    ) -> Complex[Tensor, "B K"]:
        """Predict k-space measurements.

        Args:
            k_coords: Coordinates in k-space.
            iter_step: Current training iteration (for adaptive control).
            training: Whether to enable adaptive densification.

        forward method for GSMRReconstructor.

        Executes PyTorch tensor operations.

        Args:
            k_coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            iter_step (int): Expected input tensor.
            training (bool): Expected input tensor.

        Returns:
            Complex[Tensor, 'B K']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Batch learning:
        # We share the same single GaussianCloud across the batch (assuming single volume reconstruction)
        # OR is this learning a prior?
        # 3DGS is typically "Overfit this scene". So batch dimension usually implies
        # parts of the SAME scene or multispectral measurements of SAME scene.
        # If we are doing reconstruction of *one* patient, B=1 usually.
        # If B > 1, this implies multi-coil or time-dynamic?
        # Analytical Fourier transform in GaussianCloud supports B * K coords?

        B, K, _ = k_coords.shape

        # Handle batching by flattening or looping
        # GaussianCloud.analytical_fourier_transform takes [K, 3]
        # and returns [K].
        # We can reshape k_coords to [BK, 3] and reshape back.

        k_coords_flat = k_coords.view(-1, 3)  # [BK, 3]

        k_pred_flat = self.cloud.analytical_fourier_transform(k_coords_flat)  # [BK]
        k_pred = k_pred_flat.view(B, K)

        if training:
            # Adaptive Density Control (Kerbl et al., SIGGRAPH 2023)
            # We accumulate gradients here; actual split/clone happens after backward()
            # when refine() is called by the training loop.
            # Register hook to accumulate xyz gradients for densification
            if self.cloud._xyz.requires_grad:
                self.cloud._xyz.register_hook(lambda grad: self._accumulate_xyz_grad(grad))

        return k_pred

    def _accumulate_xyz_grad(self, grad: Tensor) -> Tensor:
        """Hook to accumulate xyz gradients for adaptive control."""
        # Store gradient norm for densification decisions
        if hasattr(self, "_xyz_grad_accum"):
            self._xyz_grad_accum += grad.norm(dim=-1)
            self._grad_count += 1
        else:
            self._xyz_grad_accum = grad.norm(dim=-1)
            self._grad_count = 1
        return grad

    def refine(self, iter_step: int):
        """Call adaptive controller to refine cloud."""
        self.controller.refine(self.cloud, iter_step)

    def accumulate_grads(self):
        """Accumulate gradients for refining."""
        # Since analytical FT uses all gaussians globally (unless we threshold),
        # all are technically visible.
        # We can implement a metric of "contribution" later.
        if self.cloud._xyz.grad is not None:
            self.controller.update_grad_stats(self.cloud)
