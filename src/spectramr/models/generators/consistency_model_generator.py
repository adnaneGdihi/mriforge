"""Consistency Model Generator for MRI Reconstruction.

Implements Consistency Models for fast 1-2 step inference.
The model learns to map any point on the ODE trajectory directly to x_0.

References:
- Song et al., "Consistency Models," ICML 2023
- Application to inverse problems: Chung et al., ICLR 2023
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class ConsistencyUNet(nn.Module):
    """U-Net backbone for consistency model."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] | None = None,
        time_dim: int = 256,
    ):
        """__init__.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            features (list[int] | None): Channel sizes per U-Net level. Defaults to [64, 128, 256, 512].
            time_dim (int): Time embedding dimension.
        """
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.time_dim = time_dim

        # Time embedding (sinusoidal + MLP)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.encoder = nn.ModuleList()
        self.time_projections = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ch, feat, 3, padding=1),
                    nn.GroupNorm(8, feat),
                    nn.SiLU(),
                    nn.Conv2d(feat, feat, 3, padding=1),
                    nn.GroupNorm(8, feat),
                    nn.SiLU(),
                )
            )
            self.time_projections.append(nn.Linear(time_dim, feat))
            ch = feat

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(features[i] + features[i - 1], features[i - 1], 3, padding=1),
                    nn.GroupNorm(8, features[i - 1]),
                    nn.SiLU(),
                    nn.Conv2d(features[i - 1], features[i - 1], 3, padding=1),
                    nn.GroupNorm(8, features[i - 1]),
                    nn.SiLU(),
                )
            )

        self.out = nn.Conv2d(features[0], out_channels, 3, padding=1)

    def get_timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, C, H, W]
            t: Timestep [B]

        Returns:
            Output [B, C, H, W]

        forward method for ConsistencyUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        skips = []
        h = x
        for i, (block, time_proj) in enumerate(
            zip(self.encoder, self.time_projections, strict=False)
        ):
            h = block(h)
            # Add time conditioning
            h = h + time_proj(t_emb).view(-1, h.shape[1], 1, 1)
            skips.append(h)
            if i < len(self.encoder) - 1:
                h = F.avg_pool2d(h, 2)

        # Decoder
        for i, block in enumerate(self.decoder):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            skip = skips[-(i + 2)]
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        return self.out(h)


@register_model(
    name="consistency_model",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class ConsistencyModelGenerator(nn.Module, IGenerator):
    """Consistency Model for MRI Reconstruction.

    Key idea: Learn a consistency function f_θ(x_t, t) that maps any point
    on the ODE trajectory to x_0, satisfying f(x_t, t) = f(x_t', t') for all
    t, t' on the same trajectory.

    For inference, this allows 1-step generation: x_0 = f_θ(x_T, T)

    Training uses Consistency Training (CT) or Consistency Distillation (CD).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] | None = None,
        time_dim: int = 256,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        distillation_steps: int = 2,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            features (list[int] | None): Channel sizes per U-Net level. Defaults to [64, 128, 256, 512].
            time_dim (int): Time embedding dimension.
            sigma_min (float): Minimum noise sigma.
            sigma_max (float): Maximum noise sigma.
            rho (float): Schedule exponent (rho=7 matches CM paper).
            distillation_steps (int): Number of distillation steps for CT.
        """
        if features is None:
            features = [64, 128, 256, 512]
        logger.debug(
            f"DEBUG: ConsistencyModelGenerator Init - in_channels={in_channels}, kwargs_keys={list(kwargs.keys())}"
        )
        super().__init__()

        # Override params from nested denoising_model config if present
        if "denoising_model" in kwargs:
            dm = kwargs["denoising_model"]
            if isinstance(dm, dict) and "in_channels" in dm:
                in_channels = dm["in_channels"]
            elif hasattr(dm, "in_channels"):
                try:
                    in_channels = int(dm.in_channels)
                except (ValueError, AttributeError):
                    pass

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.distillation_steps = distillation_steps

        # Main network
        self.net = ConsistencyUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            time_dim=time_dim,
        )

        # Skip connection scaling (from Consistency Models paper)
        # c_skip(t) and c_out(t) for parameterization
        self.register_buffer("_sigma_data", torch.tensor(0.5))

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Skip connection coefficient."""
        return (self._sigma_data**2) / (sigma**2 + self._sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Output coefficient."""
        return sigma * self._sigma_data / torch.sqrt(sigma**2 + self._sigma_data**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Input scaling."""
        return 1.0 / torch.sqrt(sigma**2 + self._sigma_data**2)

    def t_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert discrete timestep to sigma."""
        # Using the schedule from Consistency Models
        rho_inv = 1.0 / self.rho
        sigma = (
            self.sigma_max**rho_inv + t * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return sigma

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        measurement: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (noisy or measurement) [B, C, H, W]
            t: Timestep [B] (optional, defaults to sigma_max for inference)
            measurement: K-space measurement for DC [B, C, H, W]
            mask: Sampling mask [B, 1, H, W]

        Returns:
            Denoised output [B, C, H, W]

        forward method for ConsistencyModelGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measurement (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = x.shape[0]

        if t is None:
            # Default to highest noise level for single-step inference
            t = torch.ones(B, device=x.device)

        sigma = self.t_to_sigma(t)
        sigma_view = sigma.view(B, 1, 1, 1)

        # Apply consistency model parameterization
        c_skip = self.c_skip(sigma_view)
        c_out = self.c_out(sigma_view)
        c_in = self.c_in(sigma_view)

        # Network prediction
        F_theta = self.net(c_in * x, t)

        # Skip connection + output scaling
        output = c_skip * x + c_out * F_theta

        # Apply data consistency if available
        if measurement is not None and mask is not None:
            output = output * (1 - mask) + measurement * mask

        return output

    def multistep_inference(
        self,
        x: torch.Tensor,
        steps: int = 2,
        measurement: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-step inference for improved quality.

        Args:
            x: Initial noise [B, C, H, W]
            steps: Number of denoising steps (1-2 typical)
            measurement: K-space measurement
            mask: Sampling mask

        Returns:
            Reconstructed output
        """
        B = x.shape[0]

        # Create timestep schedule
        ts = torch.linspace(1.0, 0.0, steps + 1, device=x.device)[:-1]

        z = x
        for t_val in ts:
            t = torch.full((B,), t_val, device=x.device)
            z = self.forward(z, t, measurement, mask)

        return z

    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,  # Not used, for API compatibility
        measurement: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Consistency Training loss.

        Uses the CT objective: ||f(x + σn, σ) - f(x + σ'n', σ')||^2
        where (σ, σ') are adjacent timesteps.
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample timesteps
        t = torch.rand(B, device=device)
        t_next = (t - 1.0 / self.distillation_steps).clamp(min=0)

        sigma = self.t_to_sigma(t)
        sigma_next = self.t_to_sigma(t_next)

        # Add noise
        noise = torch.randn_like(x_0)
        x_t = x_0 + sigma.view(B, 1, 1, 1) * noise
        x_t_next = x_0 + sigma_next.view(B, 1, 1, 1) * noise

        # Get predictions (target uses stop_gradient)
        with torch.no_grad():
            target = self.forward(x_t_next, t_next)

        pred = self.forward(x_t, t)

        # Consistency loss
        loss = F.mse_loss(pred, target)

        return loss

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate output from noise."""
        return self.multistep_inference(z, steps=self.distillation_steps, **kwargs)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "consistency_model_generator"
