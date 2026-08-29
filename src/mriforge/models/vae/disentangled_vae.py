import torch
import torch.nn as nn

from mriforge.models.registry import register_model


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.
    Conditioning is injected into the scale (gamma) and shift (beta) of GroupNorm.
    """

    def __init__(self, channels, style_dim, groups=32):
        """__init__.

        Args:
            channels (Any): Description.
            style_dim (Any): Description.
            groups (Any): Description.
        """
        super().__init__()
        # Ensure groups <= channels
        groups = min(groups, channels)
        self.norm = nn.GroupNorm(groups, channels, affine=False)
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, channels * 2),
            nn.SiLU(),  # SiLU (Swish) is standard for Diffusion/VAE
        )

        # Initialize zero to start as identity
        nn.init.zeros_(self.style_mlp[0].weight)
        nn.init.zeros_(self.style_mlp[0].bias)

    def forward(self, x, style_code):
        # x: [B, C, H, W]
        # style_code: [B, style_dim] (The Physics Vector)

        """forward.

        Args:
            x (Any): Description.
            style_code (Any): Description.
        Returns:
            Any: Description.

        forward method for AdaGN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            style_code (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.norm(x)

        # Project style to modulation parameters
        style = self.style_mlp(style_code).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, dim=1)

        return (1 + gamma) * h + beta


class DisentangledAnatomyEncoder(nn.Module):
    """
    Encodes image to spatial anatomy latent.
    Uses InstanceNorm to strip contrast info.
    """

    def __init__(self, in_channels=1, latent_channels=4):
        """__init__.

        Args:
            in_channels (Any): Description.
            latent_channels (Any): Description.
        """
        super().__init__()
        # Standard Downsampling Encoder
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            # Bottleneck
            nn.Conv2d(256, 2 * latent_channels, 3, padding=1),  # mu, logvar
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for DisentangledAnatomyEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class PhysicsConditionedDecoder(nn.Module):
    """
    Decodes anatomy latent + physics vector -> Image.
    """

    def __init__(self, out_channels=1, latent_channels=4, physics_dim=4):
        """__init__.

        Args:
            out_channels (Any): Description.
            latent_channels (Any): Description.
            physics_dim (Any): Description.
        """
        super().__init__()

        # Physics Embedding (TR, TE, TI, B0)
        self.phys_embed = nn.Sequential(nn.Linear(physics_dim, 128), nn.SiLU(), nn.Linear(128, 128))

        # Upsampling Layers with AdaGN
        self.conv1 = nn.Conv2d(latent_channels, 256, 3, padding=1)
        self.adagn1 = AdaGN(256, 128)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.adagn2 = AdaGN(128, 128)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.adagn3 = AdaGN(64, 128)

        self.final = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, z_anat, physics_vector):
        # Embed physics once
        """forward.

        Args:
            z_anat (Any): Description.
            physics_vector (Any): Description.
        Returns:
            Any: Description.

        forward method for PhysicsConditionedDecoder.

        Executes PyTorch tensor operations.

        Args:
            z_anat (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            physics_vector (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        style = self.phys_embed(physics_vector)

        x = self.conv1(z_anat)
        x = self.adagn1(x, style)
        x = nn.functional.silu(x)

        x = self.up1(x)
        x = self.conv2(x)
        x = self.adagn2(x, style)
        x = nn.functional.silu(x)

        x = self.up2(x)
        x = self.conv3(x)
        x = self.adagn3(x, style)
        x = nn.functional.silu(x)

        return torch.tanh(self.final(x))  # Norm [-1, 1] output


@register_model("physics_vae", training_mode="vae")
class PhysicsConditionedVAE(nn.Module):
    """PhysicsConditionedVAE class."""

    def __init__(self, in_channels=1, latent_dim=4, physics_dim=4, **kwargs):
        """__init__.

        Args:
            in_channels (Any): Description.
            latent_dim (Any): Description.
            physics_dim (Any): Description.
        """
        super().__init__()
        self.encoder = DisentangledAnatomyEncoder(in_channels, latent_dim)
        self.decoder = PhysicsConditionedDecoder(in_channels, latent_dim, physics_dim)

    def reparameterize(self, mu, logvar):
        """reparameterize.

        Args:
            mu (Any): Description.
            logvar (Any): Description.
        Returns:
            Any: Description.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, physics):
        """forward.

        Args:
            x (Any): Description.
            physics (Any): Description.
        Returns:
            Any: Description.

        forward method for PhysicsConditionedVAE.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            physics (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, physics)
        return recon, mu, logvar

    def decode(self, z, physics):
        """decode.

        Args:
            z (Any): Description.
            physics (Any): Description.
        Returns:
            Any: Description.
        """
        return self.decoder(z, physics)
