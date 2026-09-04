"""Score-Based Motion Disentanglement.

Innovation XVII from SOTA Roadmap: Separate anatomy from motion in latent space.

Key Insight:
    Motion artifacts corrupt MRI but the underlying anatomy is unchanged.
    Score-based models can learn to disentangle content (anatomy) from
    style (motion) using classifier-free guidance, enabling motion removal.

Mathematical Foundation:
    Score function: s_θ(x, α) = ∇_x log p(x|α)

    where α is motion severity.

    Classifier-free guidance:
        s̃(x, α) = s(x, ∅) + w·(s(x, α) - s(x, ∅))

    Setting α=0 guides toward motion-free images.

Benefits:
    - Explicit motion control
    - Preserves anatomical details
    - Works on arbitrary motion types

References:
    - [10] Score-based generative models
    - [58] Motion artifact disentanglement

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.registry import register_model


class ScoreNetwork(nn.Module):
    """Score network for motion-conditioned denoising.

    Estimates ∇_x log p(x|α, σ) where α is motion level.

    Args:
        in_channels: Input channels
        hidden_channels: Feature channels
        num_motion_levels: Number of motion conditioning levels
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_motion_levels: int = 10,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
            num_motion_levels (int): Description.
        """
        super().__init__()

        # Motion embedding
        self.motion_embed = nn.Embedding(num_motion_levels + 1, hidden_channels)  # +1 for null

        # Noise level embedding
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # U-Net style score network
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
        )

        # Bottleneck with conditioning
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )

        self.out_conv = nn.Conv2d(hidden_channels * 2, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        motion_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate score.

        Args:
            x: Noisy input (B, C, H, W)
            sigma: Noise level (B,)
            motion_level: Motion severity (B,) or None for unconditional

        Returns:
            Score estimate (B, C, H, W)
        """
        B = x.shape[0]

        # Get embeddings
        sigma_emb = self.sigma_embed(sigma.view(B, 1))

        if motion_level is not None:
            motion_emb = self.motion_embed(motion_level)
        else:
            # Null embedding for unconditional
            motion_emb = self.motion_embed(
                torch.full((B,), self.motion_embed.num_embeddings - 1, device=x.device)
            )

        cond = sigma_emb + motion_emb  # (B, hidden)

        # Encode
        h1 = self.enc1(x)
        h2 = self.enc2(h1)

        # Bottleneck with conditioning
        h = self.bottleneck(h2)
        # Multiply rather than add to avoid dimension issues
        # Scale activations by learned conditioning
        scale = torch.sigmoid(cond.mean(dim=-1, keepdim=True))
        h = h * scale.view(B, 1, 1, 1)

        # Decode
        d2 = self.dec2(h)
        out = self.out_conv(torch.cat([d2, h1], dim=1))

        return out


@register_model(name="score_disent", training_mode="diffusion")
class ScoreDisentangle(nn.Module):
    """Score-based motion disentanglement model.

    Uses classifier-free guidance to separate anatomy from motion.

    Args:
        in_channels: Input channels
        hidden_channels: Feature channels
        num_motion_levels: Motion conditioning levels
        guidance_scale: Classifier-free guidance weight
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_motion_levels: int = 10,
        guidance_scale: float = 2.0,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
            num_motion_levels (int): Description.
            guidance_scale (float): Description.
        """
        super().__init__()

        self.score_net = ScoreNetwork(in_channels, hidden_channels, num_motion_levels)
        self.guidance_scale = guidance_scale
        self.num_motion_levels = num_motion_levels

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        motion_level: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with conditional score.

        forward method for ScoreDisentangle.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sigma (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            motion_level (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.score_net(x, sigma, motion_level)

    def guided_score(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        source_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classifier-free guided score.

        Args:
            x: Input (B, C, H, W)
            sigma: Noise level (B,)
            source_motion: Current motion level (B,)

        Returns:
            Guided score (B, C, H, W)
        """
        # Unconditional score
        score_uncond = self.score_net(x, sigma, None)

        # Conditional score
        score_cond = self.score_net(x, sigma, source_motion)

        # Classifier-free guidance toward target (motion=0)
        # Move away from source motion, toward no motion
        guided = score_uncond + self.guidance_scale * (score_uncond - score_cond)

        return guided

    @torch.no_grad()
    def remove_motion(
        self,
        x: torch.Tensor,
        motion_level: torch.Tensor,
        num_steps: int = 50,
        sigma_max: float = 1.0,
        sigma_min: float = 0.01,
    ) -> torch.Tensor:
        """Remove motion artifacts using guided sampling.

        Args:
            x: Motion-corrupted image (B, C, H, W)
            motion_level: Motion severity (B,)
            num_steps: Sampling steps
            sigma_max: Maximum noise level
            sigma_min: Minimum noise level

        Returns:
            Motion-corrected image (B, C, H, W)
        """
        # Noise schedule
        sigmas = torch.linspace(sigma_max, sigma_min, num_steps, device=x.device)

        # Start from motion-corrupted image with noise
        x_t = x + sigmas[0] * torch.randn_like(x)

        for i in range(num_steps - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Guided score
            score = self.guided_score(
                x_t,
                torch.full((x.shape[0],), sigma, device=x.device),
                motion_level,
            )

            # Euler step
            dt = sigma_next - sigma
            x_t = x_t + score * dt

            # Add noise if not last step
            if i < num_steps - 2:
                x_t = x_t + torch.sqrt(sigma_next) * torch.randn_like(x_t) * 0.1

        return x_t


def create_score_disentangle(
    hidden_channels: int = 64,
    guidance_scale: float = 2.0,
) -> ScoreDisentangle:
    """Factory function for Score-Disent.

    Args:
        hidden_channels: Feature channels
        guidance_scale: CFG weight

    Returns:
        Configured model
    """
    return ScoreDisentangle(
        in_channels=2,
        hidden_channels=hidden_channels,
        guidance_scale=guidance_scale,
    )


__all__ = [
    "ScoreDisentangle",
    "ScoreNetwork",
    "create_score_disentangle",
]
