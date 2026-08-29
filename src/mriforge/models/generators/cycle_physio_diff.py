import torch
import torch.nn as nn

from mriforge.infrastructure.physics.bloch_simulation import DifferentiableBlochSimulator
from mriforge.models.generators.latent_diffusion_generator import (
    LatentDiffusionGenerator,
    LatentDiffusionGeneratorConfig,
)
from mriforge.models.generators.swin_transformer_generator import SwinTransformerGenerator
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@register_model(name="cycle_physio_diff", training_mode="diffusion")
class CyclePhysioDiffGenerator(nn.Module, IGenerator):
    """
    Cycle-Physio-Diff: The Grand Unification Architecture.

    Combines:
    1. Swin Transformer (SwinUNETR) for Quantitative Map Prediction (PD, T1, T2).
    2. Differentiable Bloch Simulator for Physics-based Coarse Reconstruction.
    3. Latent Diffusion Model for Texture Refinement and Unpaired Translation.

    Forward pass (Training):
    - Input -> Swin -> Q-Maps -> Bloch -> Coarse Image
    - Coarse Image (Condition) + Target (GT) -> Diffusion Training (Noise Prediction)

    Inference:
    - Input -> Swin -> Q-Maps -> Bloch -> Coarse Image
    - Coarse Image -> Diffusion Sampling -> Refined Image
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 256,
        patch_size: int = 4,
        embed_dim: int = 96,
        sequence_type: str = "SE",
        diffusion_timesteps: int = 1000,
        latent_channels: int = 4,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            img_size (int): Description.
            patch_size (int): Description.
            embed_dim (int): Description.
            sequence_type (str): Description.
            diffusion_timesteps (int): Description.
            latent_channels (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size

        # 1. Physics Branch (SwinUNETR -> Q-Maps)
        # Output 3 channels: PD, T1, T2
        self.physics_encoder = SwinTransformerGenerator(
            in_channels=in_channels,
            out_channels=3,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            **kwargs,
        )

        # 2. Bloch Simulator
        self.bloch_simulator = DifferentiableBlochSimulator(sequence_type=sequence_type)
        self.default_tr = 3000.0
        self.default_te = 100.0

        # 3. Diffusion Refiner (Latent Diffusion)
        # Conditioned on Coarse Image (SR3-style)
        diff_config = LatentDiffusionGeneratorConfig(
            in_channels=out_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            timesteps=diffusion_timesteps,
            conditional_translation=True,  # Enable SR3 conditioning
            base_channels=64,
        )
        self.diffusion = LatentDiffusionGenerator(config=diff_config)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "CyclePhysioDiff"

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass (Training/Inference).

        Args:
            x: Input Image [B, C, H, W]
            target: Ground Truth Image [B, C, H, W] (needed for Diffusion Training)

        Returns:
            refined_img: Final output (if inference) or Diffusion Recon (if training)
            coarse_img: Physics-based output
            q_maps: Quantitative maps [B, 3, H, W]

        forward method for CyclePhysioDiffGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Physics Branch
        q_maps = torch.abs(self.physics_encoder(x))  # [B, 3, H, W]

        tr = kwargs.get("tr", self.default_tr)
        te = kwargs.get("te", self.default_te)

        # Coarse Image via Bloch
        coarse_img = self.bloch_simulator(q_maps, TR=tr, TE=te)  # [B, 1, H, W]

        # 2. Diffusion Branch
        if self.training:
            # If target provided, train diffusion to Denoise Target conditioned on Coarse
            if target is None:
                target = x

            # Encode target to latent space for diffusion training
            z_0 = self.diffusion.encode_to_latent(target)

            # Sample random timesteps
            t = torch.randint(
                0, self.diffusion.diffusion.timesteps, (x.shape[0],), device=x.device
            ).long()

            # Add noise to latents (Forward diffusion)
            noise = torch.randn_like(z_0)
            z_t = self.diffusion.diffusion.q_sample(z_0, t, noise=noise)

            # Predict noise (or refined image) conditioned on coarse image
            # LatentDiffusionGenerator.forward expects (noisy_latents, timesteps, condition_image)
            refined_img = self.diffusion(
                z_t,
                timesteps=t,
                condition_image=coarse_img,  # Guide
            )
        else:
            # Inference: Sample from Diffusion conditioned on Coarse
            # Output shape: coarse_img shape (B, C, H, W)

            refined_img = self.diffusion.sample(
                shape=coarse_img.shape,
                condition_image=coarse_img,
                device=str(coarse_img.device),
            )

        return refined_img, coarse_img, q_maps

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        refined, _, _ = self.forward(x, **kwargs)
        return refined

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return self.physics_encoder.get_parameter_count() + self.diffusion.get_parameter_count()
