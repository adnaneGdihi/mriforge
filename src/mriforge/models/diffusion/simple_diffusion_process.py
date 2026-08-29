"""Simple Diffusion Process
======================

A lightweight diffusion process for testing and integration.
"""

import torch
from torch import nn


class SimpleDiffusionProcess(nn.Module):
    """Simple diffusion process that handles timestep generation,
    noise scheduling, and noise prediction for diffusion models.

    This implements the core diffusion model components:
    - Forward diffusion (adding noise)
    - Reverse diffusion (predicting and removing noise)
    - Timestep sampling and embedding
    """

    def __init__(self, timesteps: int = 1000, device: str = "cpu"):
        """__init__.

        Args:
            timesteps (int): Description.
            device (str): Description.
        """
        super().__init__()
        self.timesteps = timesteps
        self.device = device

        # Simple linear beta schedule
        betas = torch.linspace(0.0001, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training.

        Uses canonical implementation from training_utils.
        """
        from mriforge.infrastructure.training.utils.training_utils import (
            sample_diffusion_timesteps,
        )

        return sample_diffusion_timesteps(batch_size, self.timesteps, self.device)

    def add_noise(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to input according to diffusion schedule"""
        noise = torch.randn_like(x)

        # Get alpha values for the timesteps
        alpha_cumprod = self.alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while len(alpha_cumprod.shape) < len(x.shape):
            alpha_cumprod = alpha_cumprod.unsqueeze(-1)

        # Apply noise
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

        noisy_x = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise

        return noisy_x, noise

    def predict_noise(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict the noise that was added to the input at the given timesteps.

        This is the core of the diffusion model - learning to predict the noise
        that was added during the forward diffusion process.

        Args:
            x: Noisy input tensor
            timesteps: Timestep indices for each sample in the batch

        Returns:
            Predicted noise tensor of same shape as input

        """
        # Get alpha values for the timesteps
        alpha_cumprod = self.alphas_cumprod[timesteps]  # type: ignore

        # Reshape for broadcasting
        while len(alpha_cumprod.shape) < len(x.shape):
            alpha_cumprod = alpha_cumprod.unsqueeze(-1)

        # Compute predicted noise using reverse diffusion formula
        # x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        # noise = (x_t - sqrt(alpha_cumprod) * x_0) / sqrt(1 - alpha_cumprod)

        # For a simple baseline, we'll use a learnable noise prediction
        # In practice, this would be done by a neural network (e.g., UNet)
        batch_size = x.shape[0]

        # Simple noise prediction using convolutional layers
        # This replaces the placeholder discriminator logic
        conv1 = nn.Conv2d(x.shape[1], 64, 3, padding=1).to(x.device)
        conv2 = nn.Conv2d(64, 32, 3, padding=1).to(x.device)
        conv3 = nn.Conv2d(32, x.shape[1], 3, padding=1).to(x.device)

        # Add timestep embedding (simple sinusoidal embedding)
        timestep_embed = self._get_timestep_embedding(timesteps, 32)
        timestep_embed = timestep_embed.to(x.device)
        timestep_embed = timestep_embed.view(batch_size, -1, 1, 1)

        # Forward pass
        h = torch.relu(conv1(x))
        h = torch.relu(conv2(h))
        h = h + timestep_embed  # Add timestep information
        predicted_noise = conv3(h)

        return predicted_noise

    def _get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        embed_dim: int,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            timesteps: Timestep indices
            embed_dim: Embedding dimension

        Returns:
            Timestep embeddings

        """
        half_dim = embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def to(self, device):
        """Move to device"""
        self.device = device
        return super().to(device)
