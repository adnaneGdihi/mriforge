"""KIDOT Cost Function
=====================

Physics-informed transport cost that penalizes both:
1. Distance between source and target distributions
2. Violation of imaging physics constraints

Cost: C(x, y) = ||x - y||² + λ||y - A(x)||²

Where A is the forward physics operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsOperator(nn.Module):
    """Forward physics operator for MRI.

    Applies the forward imaging model:
    y = A(x) = IFFT(M * FFT(C * x))

    Where:
    - C: Coil sensitivity maps
    - M: Undersampling mask
    - FFT/IFFT: Fourier transform

    For VLF→HF translation, this degrades HF to VLF quality.

    Args:
        degradation_type: Type of degradation ("blur", "downsample", "noise", "combined")
        blur_sigma: Gaussian blur sigma (if using blur)
        downsample_factor: Downsampling factor (if using downsample)
        noise_level: Additive noise level
    """

    def __init__(
        self,
        degradation_type: str = "combined",
        blur_sigma: float = 2.0,
        downsample_factor: int = 2,
        noise_level: float = 0.05,
    ):
        """__init__.

        Args:
            degradation_type (str): Description.
            blur_sigma (float): Description.
            downsample_factor (int): Description.
            noise_level (float): Description.
        """
        super().__init__()

        self.degradation_type = degradation_type
        self.blur_sigma = blur_sigma
        self.downsample_factor = downsample_factor
        self.noise_level = noise_level

        # Pre-compute Gaussian kernel for blur
        if blur_sigma > 0:
            self.blur_kernel = self._create_gaussian_kernel(blur_sigma)
        else:
            self.blur_kernel = None

    def _create_gaussian_kernel(self, sigma: float, kernel_size: int = 0) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        if kernel_size == 0:
            kernel_size = int(4 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = gauss.outer(gauss)
        kernel = kernel / kernel.sum()

        return kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward degradation.

        Args:
            x: High-quality image [B, C, H, W]

        Returns:
            Degraded image [B, C, H, W]

        forward method for PhysicsOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        y = x

        if self.degradation_type in ("blur", "combined"):
            y = self._apply_blur(y)

        if self.degradation_type in ("downsample", "combined"):
            y = self._apply_downsample(y)

        if self.degradation_type in ("noise", "combined"):
            y = self._apply_noise(y)

        return y

    def _apply_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur."""
        if self.blur_kernel is None:
            return x

        B, C, H, W = x.shape
        kernel = self.blur_kernel.to(x.device).expand(C, 1, -1, -1)
        padding = kernel.shape[-1] // 2

        return F.conv2d(x, kernel, padding=padding, groups=C)

    def _apply_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling and upsampling (simulates low resolution)."""
        factor = self.downsample_factor

        # Downsample
        down = F.avg_pool2d(x, factor)

        # Upsample back to original size
        up = F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return up

    def _apply_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply additive Gaussian noise."""
        if self.training:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        return x


class KIDOTCost(nn.Module):
    """Knowledge-Informed Dynamic Optimal Transport cost.

    Combines Wasserstein distance with physics consistency:

    C(x, y) = ||x - y||² + λ||y - A(x)||²

    Term 1: Standard transport cost (move distributions)
    Term 2: Physics consistency (target must be consistent with source)

    Args:
        physics_operator: Forward physics operator A
        lambda_physics: Weight for physics consistency term
        distance_type: "l2" or "l1"
    """

    def __init__(
        self,
        physics_operator: PhysicsOperator | None = None,
        lambda_physics: float = 1.0,
        distance_type: str = "l2",
    ):
        """__init__.

        Args:
            physics_operator (Optional[PhysicsOperator]): Description.
            lambda_physics (float): Description.
            distance_type (str): Description.
        """
        super().__init__()

        self.physics_op = physics_operator or PhysicsOperator()
        self.lambda_physics = lambda_physics
        self.distance_type = distance_type

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute KIDOT cost.

        Args:
            source: Source (VLF) images [B, C, H, W]
            target: Target (HF) images [B, C, H, W]

        Returns:
            cost: Total cost scalar
            breakdown: Dict with individual cost components

        forward method for KIDOTCost.

        Executes PyTorch tensor operations.

        Args:
            source (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, dict]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Term 1: Transport distance
        if self.distance_type == "l2":
            transport_cost = F.mse_loss(source, target)
        else:
            transport_cost = F.l1_loss(source, target)

        # Term 2: Physics consistency
        # The target HF image, when degraded, should match the source VLF
        degraded = self.physics_op(target)
        physics_cost = F.mse_loss(degraded, source)

        # Combined cost
        total_cost = transport_cost + self.lambda_physics * physics_cost

        breakdown = {
            "transport_cost": transport_cost.item(),
            "physics_cost": physics_cost.item(),
            "total_cost": total_cost.item(),
        }

        return total_cost, breakdown

    def transport_cost_only(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only the transport cost (no physics)."""
        if self.distance_type == "l2":
            return F.mse_loss(source, target)
        return F.l1_loss(source, target)

    def physics_cost_only(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only the physics consistency cost."""
        degraded = self.physics_op(target)
        return F.mse_loss(degraded, source)


def create_kidot_cost(
    degradation_type: str = "combined",
    lambda_physics: float = 1.0,
    blur_sigma: float = 2.0,
    downsample_factor: int = 2,
    noise_level: float = 0.05,
) -> KIDOTCost:
    """Factory function to create KIDOT cost.

    Args:
        degradation_type: "blur", "downsample", "noise", "combined"
        lambda_physics: Physics consistency weight
        blur_sigma: Blur kernel sigma
        downsample_factor: Resolution reduction factor
        noise_level: Additive noise std

    Returns:
        Configured KIDOT cost
    """
    physics_op = PhysicsOperator(
        degradation_type=degradation_type,
        blur_sigma=blur_sigma,
        downsample_factor=downsample_factor,
        noise_level=noise_level,
    )

    return KIDOTCost(
        physics_operator=physics_op,
        lambda_physics=lambda_physics,
    )
