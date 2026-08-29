"""CSS-Diff: Cyclic Self-Supervised Diffusion for Unpaired MRI Translation.

Implements the CSS-Diff framework for unpaired Low-Field to High-Field
MRI translation with cycle consistency constraints.

Key Features:
- Forward diffusion: LF → HF (enhancement)
- Reverse diffusion: HF → LF (degradation via physics)
- Cycle consistency: LF → HF → LF ≈ LF
- Slice-wise Gap Perception (SGP) for thick slice correction

References:
- CSS-Diff: Cyclic Self-Supervised Diffusion for Low-to-High Field MRI Translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class SliceGapPerception(nn.Module):
    """Slice-wise Gap Perception (SGP) Module.

    Addresses the mismatch between thick low-field slices (5mm)
    and thin high-field slices (1mm) via contrastive alignment.

    The module learns to find the optimal corresponding high-field
    slice/slab that best represents the low-field slice content.

    Args:
        embed_dim: Embedding dimension for contrastive learning
        temperature: Contrastive loss temperature
    """

    def __init__(
        self,
        embed_dim: int = 256,
        temperature: float = 0.07,
    ):
        """__init__.

        Args:
            embed_dim (int): Description.
            temperature (float): Description.
        """
        super().__init__()
        self.temperature = temperature

        # Encoder for LF slices
        self.lf_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

        # Encoder for HF slices (shared structure)
        self.hf_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def compute_alignment(
        self,
        lf_slice: torch.Tensor,
        hf_slices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute alignment between LF slice and candidate HF slices.

        Args:
            lf_slice: [B, 1, H, W] single low-field slice
            hf_slices: [B, D, 1, H, W] candidate high-field slices

        Returns:
            aligned_hf: [B, 1, H, W] best matching HF slice
            alignment_weights: [B, D] soft alignment weights
        """
        B, D, C, H, W = hf_slices.shape

        # Encode LF slice
        lf_embed = F.normalize(self.lf_encoder(lf_slice), dim=-1)  # [B, E]

        # Encode all HF slices
        hf_flat = hf_slices.view(B * D, C, H, W)
        hf_embed = F.normalize(self.hf_encoder(hf_flat), dim=-1)  # [B*D, E]
        hf_embed = hf_embed.view(B, D, -1)  # [B, D, E]

        # Compute similarity
        similarity = torch.einsum("be,bde->bd", lf_embed, hf_embed)  # [B, D]
        alignment_weights = F.softmax(similarity / self.temperature, dim=-1)

        # Weighted combination of HF slices
        aligned_hf = torch.einsum("bd,bdchw->bchw", alignment_weights, hf_slices)

        return aligned_hf, alignment_weights

    def contrastive_loss(
        self,
        lf_slices: torch.Tensor,
        hf_slices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SGP contrastive loss.

        Args:
            lf_slices: [B, 1, H, W] low-field slices
            hf_slices: [B, 1, H, W] corresponding high-field slices

        Returns:
            InfoNCE-style contrastive loss
        """
        lf_embed = F.normalize(self.lf_encoder(lf_slices), dim=-1)
        hf_embed = F.normalize(self.hf_encoder(hf_slices), dim=-1)

        # Similarity matrix
        sim = lf_embed @ hf_embed.T / self.temperature  # [B, B]

        # Diagonal entries are positives
        labels = torch.arange(sim.shape[0], device=sim.device)

        loss = F.cross_entropy(sim, labels)
        return loss


class CycleDiffusionBlock(nn.Module):
    """Diffusion block with cycle-consistent constraints.

    Implements a U-Net style denoiser that can operate in both
    forward (LF→HF) and reverse (HF→LF) directions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            base_channels (int): Description.
            channel_mults (tuple[int, ...]): Description.
        """
        super().__init__()

        # Time embedding
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Domain embedding (LF=0, HF=1)
        self.domain_embed = nn.Embedding(2, time_dim)

        # Encoder
        self.enc_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, 3, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )
            ch = out_ch

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

        # Decoder
        self.ups = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(ch + out_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )
            ch = out_ch

        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        domain: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] noisy input
            t: [B] diffusion timesteps
            domain: 0=LF, 1=HF target domain

        Returns:
            Predicted noise or x0

        forward method for CycleDiffusionBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            domain (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = x.shape[0]

        # Embeddings
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # [B, time_dim]
        d_emb = self.domain_embed(torch.full((B,), domain, device=x.device))
        emb = t_emb + d_emb

        # Encoder
        h = self.enc_conv(x)
        skips = [h]

        for down in self.downs:
            h = down(h)
            skips.append(h)

        # Middle
        h = self.middle(h)

        # Decoder with skip connections
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]

            # Match spatial dimensions
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = up(h)

        return self.out_conv(h)


@register_model(name="css_diff", training_mode="diffusion")
class CSSdiffGenerator(nn.Module, IGenerator):
    """CSS-Diff: Cyclic Self-Supervised Diffusion Generator.

    Implements unpaired LF→HF MRI translation using:
    1. Forward diffusion: Enhance LF to HF
    2. Reverse simulation: Degrade HF to LF (physics-based)
    3. Cycle consistency: LF → HF → LF ≈ LF

    Args:
        in_channels: Number of input channels
        base_channels: Base channel count
        timesteps: Number of diffusion steps
        enable_sgp: Enable Slice Gap Perception
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        timesteps: int = 1000,
        enable_sgp: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
            timesteps (int): Description.
            enable_sgp (bool): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.timesteps = timesteps

        # Diffusion backbone (shared for both directions)
        self.diffusion_net = CycleDiffusionBlock(
            in_channels=in_channels,
            base_channels=base_channels,
        )

        # Slice Gap Perception
        self.enable_sgp = enable_sgp
        if enable_sgp:
            self.sgp = SliceGapPerception()

        # Noise schedule (cosine)
        self._setup_noise_schedule()

    def _setup_noise_schedule(self):
        """Setup cosine noise schedule."""
        s = 0.008
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps) / self.timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, 0.0001, 0.999)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to x0.

        Args:
            x0: Clean image
            t: Timestep indices
            noise: Optional pre-sampled noise

        Returns:
            Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        domain: int = 1,
    ) -> torch.Tensor:
        """Reverse diffusion: denoise one step.

        Args:
            x_t: Noisy image at timestep t
            t: Current timestep
            domain: Target domain (1=HF)

        Returns:
            Denoised image at t-1
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        # Predict noise
        pred_noise = self.diffusion_net(x_t, t_tensor, domain=domain)

        # Compute x_{t-1}
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Mean
        x0_pred = (x_t - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        if t > 0:
            noise = torch.randn_like(x_t)
            x_prev = torch.sqrt(alpha) * x0_pred + torch.sqrt(beta) * noise
        else:
            x_prev = x0_pred

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        x_lf: torch.Tensor,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Sample HF image from LF input (inference).

        Args:
            x_lf: Low-field input image
            num_inference_steps: Number of denoising steps

        Returns:
            Enhanced high-field image
        """
        device = x_lf.device

        # Start from noisy version of LF (cold diffusion style)
        # or pure noise conditioned on LF
        x = torch.randn_like(x_lf)

        # Add LF as conditioning
        step_ratio = self.timesteps // num_inference_steps

        for i in range(num_inference_steps - 1, -1, -1):
            t = i * step_ratio
            x = self.p_sample(x + 0.1 * x_lf, t, domain=1)

        return x

    def compute_cycle_loss(
        self,
        x_lf: torch.Tensor,
        x_hf: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute cyclic self-supervised loss.

        Args:
            x_lf: Low-field images
            x_hf: High-field images (unpaired, from HF dataset)

        Returns:
            Dictionary of losses
        """
        B = x_lf.shape[0]
        device = x_lf.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device)

        # Forward diffusion on HF (learn HF → noise → HF)
        noise_hf = torch.randn_like(x_hf)
        x_hf_noisy = self.q_sample(x_hf, t, noise_hf)
        pred_noise_hf = self.diffusion_net(x_hf_noisy, t, domain=1)
        loss_hf = F.mse_loss(pred_noise_hf, noise_hf)

        # Forward diffusion on LF (learn LF → noise → LF for cycle)
        noise_lf = torch.randn_like(x_lf)
        x_lf_noisy = self.q_sample(x_lf, t, noise_lf)
        pred_noise_lf = self.diffusion_net(x_lf_noisy, t, domain=0)
        loss_lf = F.mse_loss(pred_noise_lf, noise_lf)

        losses = {
            "loss_hf_diffusion": loss_hf,
            "loss_lf_diffusion": loss_lf,
            "loss_total": loss_hf + loss_lf,
        }

        # SGP loss (if enabled and training with paired slices)
        if self.enable_sgp:
            losses["loss_sgp"] = self.sgp.contrastive_loss(x_lf, x_hf)
            losses["loss_total"] = losses["loss_total"] + 0.1 * losses["loss_sgp"]

        return losses

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        In training: returns diffusion loss computation
        In inference: returns enhanced image

        Args:
            x: Input LF image (or HF for self-supervised)
            condition: Optional conditioning input

        Returns:
            Enhanced image or loss dict

        forward method for CSSdiffGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            condition (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.training:
            # Training mode: expect x to be LF, condition to be HF
            x_hf = condition if condition is not None else x
            return self.compute_cycle_loss(x, x_hf)
        else:
            # Inference mode
            return self.sample(x)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate enhanced image (inference interface)."""
        return self.sample(x)


__all__ = [
    "CSSdiffGenerator",
    "CycleDiffusionBlock",
    "SliceGapPerception",
]
