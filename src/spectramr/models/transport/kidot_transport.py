"""KIDOT Transport Model
=======================

Main transport model that integrates the velocity field to transport
VLF images to HF quality using physics-informed optimal transport.

Uses simple Euler integration (can be upgraded to torchdiffeq ODE solver).
"""

import logging

import torch
import torch.nn as nn

from spectramr.models.registry import register_model

from .kidot_cost import create_kidot_cost
from .velocity_network import VelocityNetwork

logger = logging.getLogger(__name__)


@register_model(name="kidot_transport", training_mode="transport")
class KIDOTTransport(nn.Module):
    """KIDOT: Knowledge-Informed Dynamic Optimal Transport."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        num_steps: int = 10,
        lambda_physics: float = 1.0,
        degradation_type: str = "combined",
        feature_extractor: nn.Module | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
            num_layers (int): Description.
            num_steps (int): Description.
            lambda_physics (float): Description.
            degradation_type (str): Description.
            feature_extractor (Optional[nn.Module]): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_steps = num_steps
        self.lambda_physics = lambda_physics

        self.velocity_net = VelocityNetwork(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )

        self.cost = create_kidot_cost(
            degradation_type=degradation_type,
            lambda_physics=lambda_physics,
        )

        # Initialize Feature Matching Loss
        # If no extractor provided, try to load VGG, else fallback to identity (Pixel MMD)
        if feature_extractor is None:
            try:
                import torchvision.models as models

                vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
                # Use first few layers for low-level features or deeper for semantic
                # Slicing up to relu3_3 (approx index 16)
                feature_extractor = nn.Sequential(*list(vgg.children())[:16])
                # Freeze
                for p in feature_extractor.parameters():
                    p.requires_grad = False
            except Exception:
                logger.debug("Warning: Could not load VGG for KIDOT. Using Identity (Pixel MMD).")
                feature_extractor = nn.Flatten()

        self.kid_loss = KIDFeatureMatchingLoss(feature_extractor)

    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """forward.

        Args:
            x (torch.Tensor): Description.
            return_trajectory (bool): Description.
        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, list]]: Description.

        forward method for KIDOTTransport.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            return_trajectory (bool): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        trajectory = [x.clone()] if return_trajectory else None
        dt = 1.0 / self.num_steps

        for step in range(self.num_steps):
            t = torch.tensor(step / self.num_steps, device=x.device)
            v = self.velocity_net(x, t)
            x = x + dt * v
            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    def transport(
        self,
        vlf: torch.Tensor,
        hf_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """transport.

        Args:
            vlf (torch.Tensor): Description.
            hf_target (Optional[torch.Tensor]): Description.
        Returns:
            tuple[torch.Tensor, dict]: Description.
        """
        transported = self.forward(vlf)
        losses = {}

        if hf_target is not None:
            # We can use KID for transport loss too if desired,
            # but standard transport usually relies on the cost function provided (Physics)
            # Here we combine Physics Cost + KID Feature Matching
            physics_loss, breakdown = self.cost(transported, hf_target)
            losses.update(breakdown)

            # Add KID loss
            kid = self.kid_loss(transported, hf_target)
            losses["kid_loss"] = kid.item()

            losses["total_loss"] = physics_loss + kid
        else:
            # Physics consistency only
            physics_loss = self.cost.physics_cost_only(vlf, transported)
            losses["physics_consistency"] = physics_loss.item()
            losses["total_loss"] = physics_loss

        return transported, losses

    def compute_training_loss(
        self,
        vlf: torch.Tensor,
        hf: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """compute_training_loss.

        Args:
            vlf (torch.Tensor): Description.
            hf (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, dict]: Description.
        """
        transported = self.forward(vlf)

        # 1. Distribution matching: KID (MMD)
        dist_loss = self.kid_loss(hf, transported)  # Minimize MMD(Real, Fake)

        # 2. Physics consistency
        physics_loss = self.cost.physics_cost_only(vlf, transported)

        # Total loss
        total_loss = dist_loss + self.lambda_physics * physics_loss

        breakdown = {
            "dist_loss": dist_loss.item(),
            "physics_loss": physics_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, breakdown

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """inverse.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps - 1, -1, -1):
            t = torch.tensor(step / self.num_steps, device=x.device)
            v = self.velocity_net(x, t)
            x = x - dt * v
        return x


class KIDFeatureMatchingLoss(nn.Module):
    """
    KID-Based Feature Matching Loss using MMD.
    Minimizes Maximum Mean Discrepancy between feature distributions.
    """

    def __init__(self, feature_extractor, degree=3, coef0=1.0):
        """__init__.

        Args:
            feature_extractor (Any): Description.
            degree (Any): Description.
            coef0 (Any): Description.
        """
        super().__init__()
        self.encoder = feature_extractor
        self.degree = degree
        self.coef0 = coef0
        # Ensure encoder is frozen
        for param in self.encoder.parameters():
            param.requires_grad = False

    def polynomial_kernel(self, X, Y):
        # K(x,y) = (gamma * x^T y + coef0)^degree
        """polynomial_kernel.

        Args:
            X (Any): Description.
            Y (Any): Description.
        Returns:
            Any: Description.
        """
        gamma = 1.0 / X.shape[1]
        return (gamma * torch.mm(X, Y.t()) + self.coef0).pow(self.degree)

    def mmd_squared(self, X, Y):
        """Unbiased estimator of Maximum Mean Discrepancy."""
        m = X.shape[0]
        n = Y.shape[0]

        K_XX = self.polynomial_kernel(X, X)
        K_YY = self.polynomial_kernel(Y, Y)
        K_XY = self.polynomial_kernel(X, Y)

        # Remove diagonal for unbiased estimator
        term1 = (K_XX.sum() - K_XX.diag().sum()) / (m * (m - 1) + 1e-6)
        term2 = (K_YY.sum() - K_YY.diag().sum()) / (n * (n - 1) + 1e-6)
        term3 = (2 * K_XY.sum()) / (m * n + 1e-6)

        return term1 + term2 - term3

    def forward(self, real_img, fake_img):
        # Handle channel mismatch if encoder expects RGB (3 channels) but MRI is 1
        """forward.

        Args:
            real_img (Any): Description.
            fake_img (Any): Description.
        Returns:
            Any: Description.

        forward method for KIDFeatureMatchingLoss.

        Executes PyTorch tensor operations.

        Args:
            real_img (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            fake_img (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if real_img.shape[1] == 1:
            real_in = real_img.repeat(1, 3, 1, 1)
            fake_in = fake_img.repeat(1, 3, 1, 1)
        else:
            real_in, fake_in = real_img, fake_img

        # Extract features [B, C, H, W] -> [B, Dim]
        # We assume encoder returns tensor. If it's VGG features, we flatten.
        f_real = self.encoder(real_in)
        f_fake = self.encoder(fake_in)

        f_real = f_real.view(f_real.size(0), -1)
        f_fake = f_fake.view(f_fake.size(0), -1)

        return self.mmd_squared(f_real, f_fake)


class ODEIntegrator(nn.Module):
    # ... (No changes needed here, but included for context in file)
    """ODEIntegrator class."""

    def __init__(self, velocity_fn, method="euler", num_steps=10):
        """__init__.

        Args:
            velocity_fn (Any): Description.
            method (Any): Description.
            num_steps (Any): Description.
        """
        super().__init__()
        self.velocity_fn = velocity_fn
        self.method = method
        self.num_steps = num_steps

    # ...


def create_kidot_transport(
    in_channels: int = 1,
    hidden_channels: int = 64,
    num_layers: int = 4,
    num_steps: int = 10,
    lambda_physics: float = 1.0,
    degradation_type: str = "combined",
) -> KIDOTTransport:
    """create_kidot_transport.

    Args:
        in_channels (int): Description.
        hidden_channels (int): Description.
        num_layers (int): Description.
        num_steps (int): Description.
        lambda_physics (float): Description.
        degradation_type (str): Description.
    Returns:
        KIDOTTransport: Description.
    """
    return KIDOTTransport(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_steps=num_steps,
        lambda_physics=lambda_physics,
        degradation_type=degradation_type,
    )
