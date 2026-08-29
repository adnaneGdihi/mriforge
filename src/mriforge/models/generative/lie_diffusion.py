"""Lie Group Diffusion: SO(3) Manifold Diffusion.

Innovation XII (SOTA 2.0): Manifold-Aware Generative Models.

Mathematical Foundation:
    - Lie Algebra: so(3) (Skew-symmetric matrices)
    - Exponential Map: so(3) -> SO(3)
    - Diffusion: R_t = R_{t-1} * exp(epsilon_t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.registry import register_model


class SO3:
    """Helper for SO(3) Lie Group operations."""

    @staticmethod
    def hat(v: torch.Tensor) -> torch.Tensor:
        """Map vector to skew-symmetric matrix (hat map).

        Args:
           v: [B, 3]

        Returns:
           Matrix: [B, 3, 3]
        """
        B = v.shape[0]
        x, y, z = v[:, 0], v[:, 1], v[:, 2]

        zero = torch.zeros(B, device=v.device)

        # Row 0: [0, -z, y]
        # Row 1: [z, 0, -x]
        # Row 2: [-y, x, 0]

        row0 = torch.stack([zero, -z, y], dim=1)
        row1 = torch.stack([z, zero, -x], dim=1)
        row2 = torch.stack([-y, x, zero], dim=1)

        return torch.stack([row0, row1, row2], dim=1)

    @staticmethod
    def exp_map(v: torch.Tensor) -> torch.Tensor:
        """Map so(3) vector to SO(3) matrix (Rodrigues formula).

        Args:
            v: [B, 3] omega vector.

        Returns:
            R: [B, 3, 3]
        """
        B = v.shape[0]
        theta = torch.norm(v, dim=1, keepdim=True)  # [B, 1]
        K = SO3.hat(v)
        I = torch.eye(3, device=v.device).unsqueeze(0).expand(B, 3, 3)

        # Numerical stability for small theta
        # sin(theta)/theta -> 1 - theta^2/6
        # (1-cos(theta))/theta^2 -> 1/2 - theta^2/24

        # We can use Taylor expansion or just safe division
        # Let's use torch.where for conditional logic

        theta2 = theta**2

        sin_term = torch.sin(theta) / theta
        cos_term = (1 - torch.cos(theta)) / theta2

        # Logic for theta ~ 0
        eps = 1e-6
        is_small = theta < eps

        sin_term = torch.where(is_small, 1.0 - theta2 / 6.0, sin_term)
        cos_term = torch.where(is_small, 0.5 - theta2 / 24.0, cos_term)

        # Reshape terms for broadcasting
        sin_term = sin_term.unsqueeze(-1)
        cos_term = cos_term.unsqueeze(-1)

        R = I + sin_term * K + cos_term * torch.bmm(K, K)
        return R

    @staticmethod
    def log_map(R: torch.Tensor) -> torch.Tensor:
        """Map SO(3) matrix to so(3) vector.

        Args:
            R: [B, 3, 3]

        Returns:
            v: [B, 3]
        """
        # Theta = arccos((tr(R) - 1) / 2)
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        val = (trace - 1) / 2
        # Clamp val to [-1, 1] for numerical stability
        val = torch.clamp(val, -1 + 1e-6, 1 - 1e-6)

        theta = torch.acos(val).unsqueeze(1)  # [B, 1]

        sin_theta = torch.sin(theta)

        # v = (R - R^T) / (2 * sin(theta)/theta) ? -> No
        # Standard: v_hat = theta / (2 sin(theta)) * (R - R^T)

        factor = theta / (2 * sin_theta)

        # Small angle check
        eps = 1e-6
        is_small = theta < eps
        # Limit theta/(2 sin theta) -> 1/2
        factor = torch.where(is_small, 0.5 + theta**2 / 12, factor)

        R_minus_RT = R - R.permute(0, 2, 1)

        v_hat = factor.unsqueeze(-1) * R_minus_RT

        # Vee map (inverse hat)
        # v_hat = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        # x = v_hat[2, 1]
        # y = v_hat[0, 2]
        # z = v_hat[1, 0]

        x = v_hat[:, 2, 1]
        y = v_hat[:, 0, 2]
        z = v_hat[:, 1, 0]

        return torch.stack([x, y, z], dim=1)


@register_model(name="lie_diffusion", training_mode="diffusion")
class LieDiffusion(nn.Module):
    """Diffusion Model on SO(3)."""

    def __init__(self, hidden_dim=64):
        """__init__.

        Args:
            hidden_dim (Any): Description.
        """
        super().__init__()
        # Score network: takes rotation R (flattened? or as log map vectors?) and time t
        # Predicts epsilon in tangent space (so(3) vector)

        # Input to network:
        # R is 3x3. Flattening destroys geometry?
        # Usually we map R -> R^9 or use Quaternion R^4 or R6 representation.
        # Or we work in tangent space always?
        # Simple for now: Flatten 9 + t

        self.net = nn.Sequential(
            nn.Linear(9 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Predicts 3-vector (so(3))
        )

    def forward(self, R0: torch.Tensor, t: int = 0):
        """
        R0: Ground truth rotation [B, 3, 3]

        forward method for LieDiffusion.

        Executes PyTorch tensor operations.

        Args:
            R0 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Noise injection
        # Random vector in so(3)
        B = R0.shape[0]
        epsilon = torch.randn(B, 3, device=R0.device) * 0.1 * (t + 1)  # Noise scale grows with t

        noise_rot = SO3.exp_map(epsilon)

        # R_noised = R0 * noise_rot (Intrinsic rotation) or noise_rot * R0?
        # Let's say right multiplication
        Rt = torch.bmm(R0, noise_rot)

        # Predict noise (epsilon) from Rt
        Rt_flat = Rt.view(B, -1)
        t_tensor = torch.tensor([t], device=R0.device, dtype=torch.float32).expand(B, 1)

        inp = torch.cat([Rt_flat, t_tensor], dim=1)
        epsilon_pred = self.net(inp)

        # Loss: Geodesic distance between predicted and actual noise
        # Or MSE on tangent vectors (approx geodesic)
        # Simple MSE on epsilon vectors

        loss = F.mse_loss(epsilon_pred, epsilon)
        return loss

    @torch.no_grad()
    def sample(self, R_T: torch.Tensor) -> torch.Tensor:
        """Denoise from R_T to R_0."""
        # Single step denoising for demo (or loop steps)
        # Assume T=1 for simplicity here
        B = R_T.shape[0]
        t_val = 1
        t = torch.tensor([t_val], device=R_T.device).expand(B, 1)

        Rt_flat = R_T.view(B, -1)
        inp = torch.cat([Rt_flat, t], dim=1)

        eps_pred = self.net(inp)

        # Reverse step
        # R_t = R_0 * exp(eps)
        # R_0 = R_t * exp(eps)^T

        noise_rot = SO3.exp_map(eps_pred)
        noise_rot_inv = noise_rot.transpose(1, 2)

        R_0 = torch.bmm(R_T, noise_rot_inv)
        return R_0


def create_lie_diffusion() -> LieDiffusion:
    """create_lie_diffusion.

    Returns:
        LieDiffusion: Description.
    """
    return LieDiffusion()
