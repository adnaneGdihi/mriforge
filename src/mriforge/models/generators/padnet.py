"""PaD-Net: Physics-Informed Generative Latent Volume
==================================================

Implementation of Experiment 1.1 from the Research Roadmap.
Combines a Multimodal Encoder, Conditional Latent Diffusion, and
Differentiable Bloch Decoder.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.infrastructure.physics.integration import BlochEquationSolver
from mriforge.models.diffusion.base_diffusion import Diffusion
from mriforge.models.diffusion.components import StandardDiffusionUNet
from mriforge.models.diffusion.ddim_sampler import DDIMSampler
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.reconstruction.unet import StandardUNetGenerator as StandardUNet
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


class MultimodalEncoder(nn.Module):
    """Encoder for multimodal MRI data with Modality-Dropout."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 32,
        dropout_prob: float = 0.0,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
            dropout_prob (float): Description.
        """
        super().__init__()
        self.dropout_prob = dropout_prob
        # Use StandardUNet as the encoder backbone
        # We use a U-Net to preserve spatial resolution for the latent volume
        self.backbone = StandardUNet(
            in_channels=in_channels, out_channels=latent_dim, bilinear=False
        )

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor [B, C, H, W]
            training: Whether in training mode (for modality dropout)

        Returns:
            Latent representation [B, latent_dim, H, W]

        forward method for MultimodalEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            training (bool): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if training and self.dropout_prob > 0 and x.shape[1] > 1:
            # Independent channel dropout (PADNet-style modality dropout)
            # Each channel has dropout_prob chance of being zeroed
            B, C, H, W = x.shape

            # Generate independent dropout mask per channel per sample
            keep_prob = 1 - self.dropout_prob
            mask = torch.bernoulli(torch.full((B, C, 1, 1), keep_prob, device=x.device))

            # Ensure at least one channel is kept per sample
            # If all channels dropped, randomly keep one
            all_dropped = (mask.sum(dim=1, keepdim=True) == 0).float()
            if all_dropped.any():
                # Pick random channel to keep for samples with all dropped
                random_channel = torch.randint(0, C, (B, 1, 1, 1), device=x.device)
                channel_mask = torch.zeros(B, C, 1, 1, device=x.device)
                channel_mask.scatter_(1, random_channel, 1.0)
                mask = mask + all_dropped * channel_mask

            x = x * mask

        return self.backbone(x)


@register_model(name="padnet", training_mode="reconstruction")
class PaDNet(nn.Module, IGenerator):
    """Physics-Informed Generative Latent Volume Network."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 32,
        diffusion_timesteps: int = 1000,
        q_map_channels: int = 3,  # T1, T2, PD
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            latent_dim (int): Description.
            diffusion_timesteps (int): Description.
            q_map_channels (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.q_map_channels = q_map_channels

        # 1. Multimodal Encoder
        self.encoder = MultimodalEncoder(
            in_channels=in_channels, latent_dim=latent_dim, dropout_prob=0.2
        )

        # 2. Conditional Latent Diffusion Model
        # We concatenate latent code with noisy q-maps, so input channels = q_map_channels + latent_dim
        self.diffusion_model = StandardDiffusionUNet(
            in_channels=q_map_channels + latent_dim,
            out_channels=q_map_channels,  # Predicts noise for q-maps
            time_emb_dim=256,
            channels=(64, 128, 256, 512),
            num_res_blocks=2,
        )

        # 3. Physics-Informed Decoder (Bloch Simulator)
        self.bloch_solver = BlochEquationSolver()

        # 4. Diffusion Process & Sampler
        self.diffusion = Diffusion(timesteps=diffusion_timesteps)
        self.sampler = DDIMSampler(self.diffusion)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "PaDNet"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.encoder.parameters()) + sum(
            p.numel() for p in self.diffusion_model.parameters()
        )

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor = None,
        context: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass.

        Handles both training (diffusion) and inference (generation).

        Args:
            x: Input tensor.
               - For training: Noisy q-maps [B, 3, H, W]
               - For inference: Condition image [B, C, H, W]
            timesteps: Diffusion timesteps [B] (Training only)
            context: Condition image [B, C, H, W] (Training only)
            **kwargs: Additional arguments (e.g. sequence_params for inference)

        Returns:
            - Training: Predicted noise [B, 3, H, W]
            - Inference: Synthesized image [B, 1, H, W]

        forward method for PaDNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if timesteps is not None:
            # Training Mode
            if context is None:
                # Try to find context in kwargs
                context = kwargs.get("condition", kwargs.get("cond"))

            if context is None:
                raise ValueError("PaDNet training requires 'context' (condition image).")

            return self.forward_diffusion(x, timesteps, context)
        else:
            # Inference Mode
            # Assume x is the condition image
            sequence_params = kwargs.get("sequence_params", {})
            return self.generate(x, sequence_params=sequence_params, **kwargs)

    def forward_diffusion(
        self,
        noisy_q_maps: torch.Tensor,
        timesteps: torch.Tensor,
        condition_image: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for diffusion training.

        Args:
            noisy_q_maps: Noisy latent q-maps [B, 3, H, W]
            timesteps: Timesteps [B]
            condition_image: Conditioning image (e.g. T1w) [B, C, H, W]

        Returns:
            Predicted noise [B, 3, H, W]
        """
        # 1. Encode condition
        latent_code = self.encoder(condition_image)

        # 2. Concatenate
        # noisy_q_maps: [B, 3, H, W]
        # latent_code: [B, 32, H, W]
        # input: [B, 35, H, W]
        diffusion_input = torch.cat([noisy_q_maps, latent_code], dim=1)

        # 3. Predict noise
        noise_pred = self.diffusion_model(diffusion_input, timesteps)

        return noise_pred

    def predict_q_maps(self, condition_image: torch.Tensor) -> torch.Tensor:
        """Predict quantitative maps from condition image.

        Used for Physics-Informed Training loop.

        Args:
            condition_image: [B, C, H, W]

        Returns:
            Predicted q-maps [B, 3, H, W] (T1, T2, PD)
        """
        # 1. Encode condition
        latent_code = self.encoder(condition_image)

        # 2. Prepare input for Diffusion UNet
        # We use the UNet as a generator.
        # Input: Concatenation of (Noise, Latent).
        # We use random noise to allow stochasticity, or zeros for deterministic.
        # For training stability, let's use random noise but with t=0 embedding.
        noise = torch.randn(
            condition_image.shape[0],
            self.q_map_channels,
            condition_image.shape[2],
            condition_image.shape[3],
            device=condition_image.device,
        )

        diffusion_input = torch.cat([noise, latent_code], dim=1)

        # 3. Predict q-maps (using t=0)
        timesteps = torch.zeros(
            condition_image.shape[0], device=condition_image.device, dtype=torch.long
        )

        # The model output is technically "noise prediction" in standard diffusion.
        # But here we interpret it as q-map prediction (or residual).
        # If we want it to predict q-maps directly, we train it to do so via the Bloch loss.
        # So we take the raw output.
        raw_output = self.diffusion_model(diffusion_input, timesteps)

        # 4. Map to physical range (Positivity)
        # T1 ~ [0, 3000+], T2 ~ [0, 500+], PD ~ [0, 1+]
        # We use Softplus and scaling.
        t1 = F.softplus(raw_output[:, 0:1]) * 2000.0
        t2 = F.softplus(raw_output[:, 1:2]) * 500.0
        pd = F.softplus(raw_output[:, 2:3])

        return torch.cat([t1, t2, pd], dim=1)

    def generate(
        self,
        condition_image: torch.Tensor,
        sequence_params: dict[str, float],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate synthesized image from condition.

        Args:
            condition_image: Input image (e.g. T1w)
            sequence_params: Target sequence parameters (TR, TE)

        Returns:
            Synthesized image (e.g. T2w)
        """
        # 1. Encode condition
        latent_code = self.encoder(condition_image)

        # 2. Sample from diffusion model (Reverse Process)
        # Wrapper for DDIMSampler to match signature
        def model_wrapper(x, t, cond=None):
            """model_wrapper.

            Args:
                x (Any): Description.
                t (Any): Description.
                cond (Any): Description.
            Returns:
                Any: Description.
            """
            if cond is None:
                raise ValueError("Condition required for PaDNet sampling")
            # Concatenate noisy input and latent code
            diffusion_input = torch.cat([x, cond], dim=1)
            return self.diffusion_model(diffusion_input, t)

        # Initial shape for q-maps
        shape = (
            condition_image.shape[0],
            self.q_map_channels,
            condition_image.shape[2],
            condition_image.shape[3],
        )

        # Run DDIM Sampling
        q_maps = self.sampler.sample(
            model=model_wrapper,
            shape=shape,
            cond=latent_code,
            num_inference_steps=50,  # Default to 50 steps
        )

        # 3. Physics-Informed Decode
        # Bloch solver expects m0 (PD), T1, T2.
        # q_maps channels: 0=T1, 1=T2, 2=PD
        t1_map = q_maps[:, 0:1]
        t2_map = q_maps[:, 1:2]
        pd_map = q_maps[:, 2:3]

        # Construct m0 tensor for Bloch solver: [B, H, W, 3] (Mx, My, Mz)
        # Initially Mz = PD, Mx=My=0
        m0 = torch.zeros(q_maps.shape[0], q_maps.shape[2], q_maps.shape[3], 3, device=q_maps.device)
        m0[..., 2] = pd_map.squeeze(1)

        # Update solver params
        self.bloch_solver.tr = sequence_params.get("tr", 500.0)
        self.bloch_solver.te = sequence_params.get("te", 10.0)
        self.bloch_solver.t1 = t1_map.squeeze(
            1
        )  # This might need to be passed per voxel if solver supports it
        self.bloch_solver.t2 = t2_map.squeeze(1)

        # Solve
        signal = self.bloch_solver(m0)

        # Signal is [B, H, W, 2] (real, imag)
        # Return magnitude
        magnitude = torch.sqrt(signal[..., 0] ** 2 + signal[..., 1] ** 2)
        return magnitude.unsqueeze(1)  # [B, 1, H, W]
