"""Residual-Error Shifting Diffusion for Motion Correction (Res-MoCoDiff).

Innovation II from SOTA Roadmap: Motion artifact removal via residual-shifting diffusion.

Key Insight:
    Standard diffusion models denoise from Gaussian noise. Motion artifacts are NOT
    Gaussian - they are structured, coherent ghosting patterns. Res-MoCoDiff defines
    the forward process as degradation toward the motion-corrupted image itself.

Mathematical Formulation:
    Forward: q(x_t | x_0, y_motion) = N(α_t·x_0 + (1-α_t)·y_motion, β_t·I)
    At t=T: x_T ≈ y_motion (not pure noise)
    Reverse: Network predicts residual artifact (y_motion - x_0)

Benefits:
    - Short-range diffusion (motion-corrupted → clean)
    - Preserves geometry (no hallucination)
    - Swin Transformer captures periodic ghosting patterns

References:
    - [6] PMC12167925 - MRI super-resolution with residual shifting
    - [7] PMC12532098 - Res-MoCoDiff for brain MRI

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import canonical WindowAttention from blocks/swin.py (consolidated per REFACTORING_RULES.md)
from mriforge.models.blocks.swin import SwinBlock as SwinTransformerBlock
from mriforge.models.registry import register_model


class ResMoCoDiffDenoiser(nn.Module):
    """CNN-based denoiser for Res-MoCoDiff.

    Simple U-Net with residual blocks. Predicts the residual artifact.

    Args:
        in_channels: Input channels (2 for complex as real/imag)
        base_dim: Base feature dimension
        num_blocks: Residual blocks per stage
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_dim: int = 64,
        num_blocks: int = 2,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            base_dim (int): Description.
            num_blocks (int): Description.
        """
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )

        # Initial conv (x_t and y_motion concatenated)
        self.in_conv = nn.Conv2d(in_channels * 2, base_dim, 3, padding=1)

        # Encoder
        self.enc1 = nn.Sequential(*[ResBlock(base_dim) for _ in range(num_blocks)])
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(*[ResBlock(base_dim * 2) for _ in range(num_blocks)])
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[ResBlock(base_dim * 4) for _ in range(num_blocks)])

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(*[ResBlock(base_dim * 4) for _ in range(num_blocks)])
        self.proj2 = nn.Conv2d(base_dim * 4, base_dim * 2, 1)

        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(*[ResBlock(base_dim * 2) for _ in range(num_blocks)])
        self.proj1 = nn.Conv2d(base_dim * 2, base_dim, 1)

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, in_channels, 3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        y_motion: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict residual artifact.

        forward method for ResMoCoDiffDenoiser.

        Executes PyTorch tensor operations.

        Args:
            x_t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y_motion (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1) if t.dim() == 1 else t)

        # Concatenate inputs
        x = torch.cat([x_t, y_motion], dim=1)
        x = self.in_conv(x)

        # Encoder
        h1 = self.enc1(x)
        x = self.down1(h1)

        h2 = self.enc2(x)
        x = self.down2(h2)

        # Bottleneck + time
        x = self.bottleneck(x)
        x = x + t_emb[:, :, None, None]

        # Decoder
        x = self.up2(x)
        x = torch.cat([x, h2], dim=1)
        x = self.dec2(x)
        x = self.proj2(x)

        x = self.up1(x)
        x = torch.cat([x, h1], dim=1)
        x = self.dec1(x)
        x = self.proj1(x)

        return self.out_conv(x)


class ResBlock(nn.Module):
    """Simple residual block."""

    def __init__(self, dim: int) -> None:
        """__init__.

        Args:
            dim (int): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return x + self.net(x)


@register_model(name="res_mocodiff", training_mode="diffusion")
class ResMoCoDiff(nn.Module):
    """Residual-Error Shifting Diffusion Model.

    Forward: Diffuse from clean x_0 toward motion-corrupted y_motion
    Reverse: Learn to predict the artifact residual (y_motion - x_0)

    Args:
        denoiser: Swin Transformer denoiser network
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting noise schedule value
        beta_end: Ending noise schedule value
    """

    def __init__(
        self,
        denoiser: ResMoCoDiffDenoiser | None = None,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        # Config args
        in_channels: int = 2,
        base_channels: int = 64,
    ) -> None:
        """__init__.

        Args:
            denoiser (ResMoCoDiffDenoiser | None): Description.
            num_timesteps (int): Description.
            beta_start (float): Description.
            beta_end (float): Description.
            in_channels (int): Description.
            base_channels (int): Description.
        """
        super().__init__()

        if denoiser is None:
            denoiser = ResMoCoDiffDenoiser(
                in_channels=in_channels,
                base_dim=base_channels,
            )

        self.denoiser = denoiser
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def q_sample(
        self,
        x_0: torch.Tensor,
        y_motion: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion sampling.

        q(x_t | x_0, y_motion) = N(α_t·x_0 + (1-α_t)·y_motion, β_t·I)

        Args:
            x_0: Clean image (B, 2, H, W)
            y_motion: Motion-corrupted image (B, 2, H, W)
            t: Timesteps (B,)
            noise: Optional noise tensor

        Returns:
            Noisy sample x_t (B, 2, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get coefficients for timestep t
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # Mean shifts from x_0 toward y_motion
        # Standard: mean = sqrt_alpha * x_0
        # Res-MoCoDiff: mean = sqrt_alpha * x_0 + (1 - sqrt_alpha) * y_motion
        mean = sqrt_alpha * x_0 + (1 - sqrt_alpha) * y_motion

        return mean + sqrt_one_minus_alpha * noise

    def compute_loss(
        self,
        x_0: torch.Tensor,
        y_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss.

        Predicts residual artifact (y_motion - x_0).

        Args:
            x_0: Clean image (B, 2, H, W)
            y_motion: Motion-corrupted image (B, 2, H, W)

        Returns:
            MSE loss
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffusion
        x_t = self.q_sample(x_0, y_motion, t)

        # Predict residual
        pred_residual = self.denoiser(x_t, y_motion, t.float() / self.num_timesteps)

        # Target is the artifact (y_motion - x_0)
        target_residual = y_motion - x_0

        return F.mse_loss(pred_residual, target_residual)

    @torch.no_grad()
    def sample(
        self,
        y_motion: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Reverse diffusion sampling (motion correction).

        Args:
            y_motion: Motion-corrupted image (B, 2, H, W)
            num_steps: Number of sampling steps (default: all)

        Returns:
            Corrected image (B, 2, H, W)
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        B = y_motion.shape[0]
        device = y_motion.device

        # Start from near motion-corrupted (t=T)
        x = y_motion + 0.1 * torch.randn_like(y_motion)

        # Subsample timesteps if needed
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps).long().to(device)

        for t in timesteps:
            t_batch = torch.full((B,), t.item(), device=device, dtype=torch.long)
            t_norm = t.float() / self.num_timesteps

            # Predict residual
            pred_residual = self.denoiser(x, y_motion, t_norm.expand(B))

            # Estimate x_0 from residual prediction
            x_0_pred = y_motion - pred_residual

            if t > 0:
                # Add noise for next step
                noise = torch.randn_like(x)
                x = self.q_sample(x_0_pred, y_motion, t_batch - 1, noise)
            else:
                x = x_0_pred

        return x


def create_res_mocodiff(
    image_size: int = 256,
    base_dim: int = 64,
    num_timesteps: int = 1000,
) -> ResMoCoDiff:
    """Factory function for Res-MoCoDiff.

    Args:
        image_size: Input image size (unused, kept for API compat)
        base_dim: Base feature dimension
        num_timesteps: Diffusion timesteps

    Returns:
        Configured Res-MoCoDiff model
    """
    denoiser = ResMoCoDiffDenoiser(
        in_channels=2,
        base_dim=base_dim,
        num_blocks=2,
    )

    return ResMoCoDiff(
        denoiser=denoiser,
        num_timesteps=num_timesteps,
    )


__all__ = [
    "ResMoCoDiff",
    "ResMoCoDiffDenoiser",
    "SwinTransformerBlock",
    "create_res_mocodiff",
]
