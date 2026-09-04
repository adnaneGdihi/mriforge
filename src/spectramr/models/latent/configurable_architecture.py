"""Configurable Latent Architecture System
=========================================

Provides flexible encoder/decoder architectures with configurable latent spaces
for VAE, diffusion, and other latent-based models.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentConfig:
    """Configuration for latent space structure."""

    # Latent dimensions
    latent_dim: int = 256
    latent_channels: int = 4  # For spatial latent spaces
    latent_height: int | None = None  # None = fully connected
    latent_width: int | None = None

    # Distribution type
    distribution: Literal["gaussian", "categorical", "vq", "flow"] = "gaussian"

    # For VQ-VAE
    num_embeddings: int = 512
    commitment_cost: float = 0.25

    # For categorical/discrete
    num_categories: int = 10
    temperature: float = 1.0


@dataclass
class EncoderConfig:
    """Configuration for encoder architecture."""

    # Architecture type
    arch_type: Literal["conv", "resnet", "attention", "hybrid"] = "conv"

    # Dimensions
    in_channels: int = 1
    hidden_dims: list[int] = None

    # Components
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    use_spectral_norm: bool = False
    dropout_rate: float = 0.1
    activation: Literal["relu", "silu", "gelu", "leaky_relu"] = "relu"

    # Attention (if arch_type="attention" or "hybrid")
    use_attention: bool = False
    num_heads: int = 8
    attention_layers: list[int] = None  # Which layers get attention

    # Residual connections
    use_residual: bool = False

    def __post_init__(self):
        """__post_init__.

        Returns:
            Any: Description.
        """
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256]
        if self.attention_layers is None:
            self.attention_layers = [2, 3]  # Add attention at these layers


@dataclass
class DecoderConfig:
    """Configuration for decoder architecture."""

    # Architecture type
    arch_type: Literal["conv", "resnet", "attention", "hybrid"] = "conv"

    # Dimensions
    out_channels: int = 1
    hidden_dims: list[int] = None

    # Components
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    use_spectral_norm: bool = False
    dropout_rate: float = 0.1
    activation: Literal["relu", "silu", "gelu", "leaky_relu"] = "relu"

    # Attention
    use_attention: bool = False
    num_heads: int = 8
    attention_layers: list[int] = None

    # Residual connections
    use_residual: bool = False

    # Output activation
    output_activation: Literal["tanh", "sigmoid", "none"] = "tanh"

    def __post_init__(self):
        """__post_init__.

        Returns:
            Any: Description.
        """
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64, 32]
        if self.attention_layers is None:
            self.attention_layers = [0, 1]


class FlexibleEncoder(nn.Module):
    """Configurable encoder with multiple architecture options."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        latent_config: LatentConfig,
    ):
        """__init__.

        Args:
            encoder_config (EncoderConfig): Description.
            latent_config (LatentConfig): Description.
        """
        super().__init__()
        self.encoder_config = encoder_config
        self.latent_config = latent_config

        # Build encoder backbone
        self.encoder = self._build_encoder()

        # Build latent projection heads
        self.latent_projection = self._build_latent_projection()

    def _build_encoder(self) -> nn.Module:
        """Build encoder backbone based on configuration."""
        cfg = self.encoder_config

        if cfg.arch_type == "conv":
            return self._build_conv_encoder()
        elif cfg.arch_type == "resnet":
            return self._build_resnet_encoder()
        elif cfg.arch_type == "attention":
            return self._build_attention_encoder()
        elif cfg.arch_type == "hybrid":
            return self._build_hybrid_encoder()
        else:
            raise ValueError(f"Unknown architecture type: {cfg.arch_type}")

    def _build_conv_encoder(self) -> nn.Module:
        """Build convolutional encoder."""
        cfg = self.encoder_config
        modules = []

        in_ch = cfg.in_channels
        for i, h_dim in enumerate(cfg.hidden_dims):
            # Convolution
            conv = nn.Conv2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1)
            if cfg.use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            modules.append(conv)

            # Normalization
            if cfg.use_batch_norm:
                modules.append(nn.BatchNorm2d(h_dim))
            elif cfg.use_layer_norm:
                modules.append(nn.GroupNorm(1, h_dim))  # Layer norm equivalent

            # Activation
            modules.append(self._get_activation(cfg.activation))

            # Dropout
            if cfg.dropout_rate > 0:
                modules.append(nn.Dropout2d(cfg.dropout_rate))

            # Attention (if hybrid)
            if cfg.use_attention and i in cfg.attention_layers:
                modules.append(SelfAttention2D(h_dim, num_heads=cfg.num_heads))

            in_ch = h_dim

        return nn.Sequential(*modules)

    def _build_resnet_encoder(self) -> nn.Module:
        """Build ResNet-style encoder with residual connections."""
        cfg = self.encoder_config
        modules = []

        # Initial conv
        in_ch = cfg.in_channels
        modules.append(nn.Conv2d(in_ch, cfg.hidden_dims[0], kernel_size=7, stride=2, padding=3))
        if cfg.use_batch_norm:
            modules.append(nn.BatchNorm2d(cfg.hidden_dims[0]))
        modules.append(self._get_activation(cfg.activation))

        # ResBlocks
        for i in range(len(cfg.hidden_dims) - 1):
            modules.append(
                ResBlock(
                    cfg.hidden_dims[i],
                    cfg.hidden_dims[i + 1],
                    stride=2,
                    use_batch_norm=cfg.use_batch_norm,
                    activation=cfg.activation,
                    dropout_rate=cfg.dropout_rate,
                )
            )

            if cfg.use_attention and i in cfg.attention_layers:
                modules.append(SelfAttention2D(cfg.hidden_dims[i + 1], num_heads=cfg.num_heads))

        return nn.Sequential(*modules)

    def _build_attention_encoder(self) -> nn.Module:
        """Build attention-based encoder (ViT-style).

        Uses a convolutional stem to extract spatial features then applies
        multi-head self-attention at every stage.  Falls back to the hybrid
        conv+attention encoder with attention enabled on all layers.
        """
        cfg = self.encoder_config
        # Re-use the conv encoder but force attention on every layer
        original_attention = cfg.use_attention
        original_layers = list(cfg.attention_layers or [])
        cfg.use_attention = True
        cfg.attention_layers = list(range(len(cfg.hidden_dims)))
        encoder = self._build_conv_encoder()
        # Restore original config (dataclass is mutable)
        cfg.use_attention = original_attention
        cfg.attention_layers = original_layers
        return encoder

    def _build_hybrid_encoder(self) -> nn.Module:
        """Build hybrid Conv+Attention encoder."""
        return self._build_conv_encoder()  # With attention already added

    def _build_latent_projection(self) -> nn.Module:
        """Build projection to latent space."""
        latent_cfg = self.latent_config
        cfg = self.encoder_config

        # Calculate feature size after encoder
        with torch.no_grad():
            x = torch.randn(1, cfg.in_channels, 256, 256)
            features = self.encoder(x)
            feature_size = features.numel() // features.shape[0]

        if latent_cfg.distribution == "gaussian":
            # VAE-style: output mu and logvar
            return nn.ModuleDict(
                {
                    "mu": nn.Linear(feature_size, latent_cfg.latent_dim),
                    "logvar": nn.Linear(feature_size, latent_cfg.latent_dim),
                }
            )
        elif latent_cfg.distribution == "categorical":
            return nn.Linear(feature_size, latent_cfg.num_categories)
        elif latent_cfg.distribution == "vq":
            return VectorQuantizer(
                num_embeddings=latent_cfg.num_embeddings,
                embedding_dim=latent_cfg.latent_dim,
                commitment_cost=latent_cfg.commitment_cost,
            )
        else:
            # Simple projection
            return nn.Linear(feature_size, latent_cfg.latent_dim)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation module."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "silu": nn.SiLU(inplace=True),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        }
        if activation not in activations:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of {sorted(activations)}"
            )
        return activations[activation]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode input to latent space.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Dictionary with latent parameters

        forward method for FlexibleEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Pass through encoder
        features = self.encoder(x)
        features_flat = features.view(features.size(0), -1)

        # Project to latent space
        if self.latent_config.distribution == "gaussian":
            mu = self.latent_projection["mu"](features_flat)
            logvar = self.latent_projection["logvar"](features_flat)
            return {"mu": mu, "logvar": logvar}

        elif self.latent_config.distribution == "vq":
            # VQ-VAE quantization
            quantized, vq_loss, perplexity = self.latent_projection(features_flat)
            return {"latent": quantized, "vq_loss": vq_loss, "perplexity": perplexity}

        else:
            latent = self.latent_projection(features_flat)
            return {"latent": latent}


class FlexibleDecoder(nn.Module):
    """Configurable decoder with multiple architecture options."""

    def __init__(
        self,
        decoder_config: DecoderConfig,
        latent_config: LatentConfig,
    ):
        """__init__.

        Args:
            decoder_config (DecoderConfig): Description.
            latent_config (LatentConfig): Description.
        """
        super().__init__()
        self.decoder_config = decoder_config
        self.latent_config = latent_config

        # Initial projection from latent
        self.initial_projection = self._build_initial_projection()

        # Build decoder backbone
        self.decoder = self._build_decoder()

        # Final output layer
        self.final_layer = self._build_final_layer()

    def _build_initial_projection(self) -> nn.Module:
        """Build initial projection from latent space."""
        cfg = self.decoder_config
        latent_dim = self.latent_config.latent_dim

        # Project to initial spatial dimensions
        initial_h, initial_w = 16, 16  # Starting spatial size
        initial_channels = cfg.hidden_dims[0]

        return nn.Sequential(
            nn.Linear(latent_dim, initial_channels * initial_h * initial_w),
            nn.ReLU(),
        )

    def _build_decoder(self) -> nn.Module:
        """Build decoder backbone."""
        cfg = self.decoder_config

        if cfg.arch_type in ["conv", "hybrid"]:
            return self._build_conv_decoder()
        elif cfg.arch_type == "resnet":
            return self._build_resnet_decoder()
        elif cfg.arch_type == "attention":
            return self._build_attention_decoder()
        else:
            raise ValueError(f"Unknown architecture type: {cfg.arch_type}")

    def _build_conv_decoder(self) -> nn.Module:
        """Build convolutional decoder."""
        cfg = self.decoder_config
        modules = []

        for i in range(len(cfg.hidden_dims) - 1):
            in_ch = cfg.hidden_dims[i]
            out_ch = cfg.hidden_dims[i + 1]

            # Transposed convolution
            deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            if cfg.use_spectral_norm:
                deconv = nn.utils.spectral_norm(deconv)
            modules.append(deconv)

            # Normalization
            if cfg.use_batch_norm:
                modules.append(nn.BatchNorm2d(out_ch))
            elif cfg.use_layer_norm:
                modules.append(nn.GroupNorm(1, out_ch))

            # Activation
            modules.append(self._get_activation(cfg.activation))

            # Dropout
            if cfg.dropout_rate > 0:
                modules.append(nn.Dropout2d(cfg.dropout_rate))

            # Attention
            if cfg.use_attention and i in cfg.attention_layers:
                modules.append(SelfAttention2D(out_ch, num_heads=cfg.num_heads))

        return nn.Sequential(*modules)

    def _build_resnet_decoder(self) -> nn.Module:
        """Build ResNet-style decoder."""
        cfg = self.decoder_config
        modules = []

        for i in range(len(cfg.hidden_dims) - 1):
            modules.append(
                ResBlock(
                    cfg.hidden_dims[i],
                    cfg.hidden_dims[i + 1],
                    stride=1,  # Upsample separately
                    use_batch_norm=cfg.use_batch_norm,
                    activation=cfg.activation,
                    dropout_rate=cfg.dropout_rate,
                    upsample=True,
                )
            )

            if cfg.use_attention and i in cfg.attention_layers:
                modules.append(SelfAttention2D(cfg.hidden_dims[i + 1], num_heads=cfg.num_heads))

        return nn.Sequential(*modules)

    def _build_attention_decoder(self) -> nn.Module:
        """Build attention-based decoder.

        Re-uses the convolutional decoder with attention enabled at every
        stage, mirroring the encoder's attention-everywhere design.
        """
        cfg = self.decoder_config
        original_attention = cfg.use_attention
        original_layers = list(cfg.attention_layers or [])
        cfg.use_attention = True
        cfg.attention_layers = list(range(len(cfg.hidden_dims) - 1))
        decoder = self._build_conv_decoder()
        cfg.use_attention = original_attention
        cfg.attention_layers = original_layers
        return decoder

    def _build_final_layer(self) -> nn.Module:
        """Build final output layer."""
        cfg = self.decoder_config

        layers = [
            nn.ConvTranspose2d(
                cfg.hidden_dims[-1],
                cfg.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        ]

        # Output activation
        if cfg.output_activation == "tanh":
            layers.append(nn.Tanh())
        elif cfg.output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        # else: no activation

        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation module."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "silu": nn.SiLU(inplace=True),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        }
        if activation not in activations:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of {sorted(activations)}"
            )
        return activations[activation]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space.

        Args:
            z: Latent tensor [B, latent_dim]

        Returns:
            Reconstructed output [B, C, H, W]

        forward method for FlexibleDecoder.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Project from latent
        x = self.initial_projection(z)

        # Reshape to spatial
        B = x.shape[0]
        x = x.view(B, self.decoder_config.hidden_dims[0], 16, 16)

        # Decode
        x = self.decoder(x)

        # Final output
        x = self.final_layer(x)

        return x


class SelfAttention2D(nn.Module):
    """2D Self-Attention module."""

    def __init__(self, channels: int, num_heads: int = 8):
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
        """
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]

        forward method for SelfAttention2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Flatten spatial dimensions
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]

        # Apply attention
        x_attn, _ = self.mha(x_flat, x_flat, x_flat)

        # Residual + norm
        x_flat = self.ln(x_flat + x_attn)

        # Reshape back
        return x_flat.transpose(1, 2).view(B, C, H, W)


class ResBlock(nn.Module):
    """Residual block with optional upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        upsample: bool = False,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            stride (int): Description.
            use_batch_norm (bool): Description.
            activation (str): Description.
            dropout_rate (float): Description.
            upsample (bool): Description.
        """
        super().__init__()
        self.upsample = upsample

        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride)
            else:
                self.shortcut = nn.Identity()

        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Activation
        activations = {
            "relu": nn.ReLU(inplace=True),
            "silu": nn.SiLU(inplace=True),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        }
        if activation not in activations:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of {sorted(activations)}"
            )
        self.activation = activations[activation]

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
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + shortcut
        out = self.activation(out)

        return out


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        """__init__.

        Args:
            num_embeddings (int): Description.
            embedding_dim (int): Description.
            commitment_cost (float): Description.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input.

        Args:
            z: Latent tensor [B, D]

        Returns:
            Tuple of (quantized, vq_loss, perplexity)

        forward method for VectorQuantizer.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Flatten if needed
        z_flat = z.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )

        # Find nearest embedding
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices)

        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), z_flat)
        q_latent_loss = F.mse_loss(quantized, z_flat.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z_flat + (quantized - z_flat).detach()

        # Perplexity
        avg_probs = torch.mean(F.one_hot(encoding_indices, self.num_embeddings).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape back
        quantized = quantized.view(z.shape)

        return quantized, vq_loss, perplexity


__all__ = [
    "DecoderConfig",
    "EncoderConfig",
    "FlexibleDecoder",
    "FlexibleEncoder",
    "LatentConfig",
    "ResBlock",
    "SelfAttention2D",
    "VectorQuantizer",
]
