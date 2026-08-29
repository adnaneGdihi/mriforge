"""Discriminator Implementation
===========================

Concrete implementation of IDiscriminator following SOLID principles.
Single responsibility for discrimination tasks.
"""

import torch
from torch import nn

from mriforge.models.interfaces.models import IDiscriminator
from mriforge.models.registry import register_model


@register_model(name="patch_gan", training_mode="gan")
@register_model(name="patchgan_discriminator", training_mode="gan")
@register_model(name="patch_latent_discriminator", training_mode="gan")
@register_model(name="multiscale_latent_discriminator", training_mode="gan")
class PatchGANDiscriminator(IDiscriminator, nn.Module):
    """PatchGAN discriminator implementation following SOLID principles.

    Single Responsibility: Only handles patch-based discrimination
    Open/Closed: Can be extended without modification
    Liskov Substitution: Fully substitutable for any IDiscriminator
    Interface Segregation: Only implements needed interfaces
    Dependency Inversion: Depends on abstractions, not concretions
    """

    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        spectral_norm: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            ndf (int): Description.
            n_layers (int): Description.
            spectral_norm (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.ndf = ndf
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm

        # Build layers
        layers = [
            self._create_conv_layer(
                in_channels,
                ndf,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        n = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                self._create_conv_layer(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(min(32, ndf * nf_mult), ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            self._create_conv_layer(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(min(32, ndf * nf_mult), ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            self._create_conv_layer(
                ndf * nf_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        ]

        self.model = nn.Sequential(*layers)

    def _create_conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> nn.Module:
        """Create a convolutional layer with optional spectral
        normalization.
        """
        conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        if self.spectral_norm:
            return nn.utils.spectral_norm(conv)
        return conv

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "PatchGANDiscriminator"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator.

        forward method for PatchGANDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.model(x)

    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Discriminate real vs fake samples."""
        result = self.forward(x)
        return result

    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract feature maps for feature matching.

        Note: This runs in eval mode to prevent double-updating Batch Norm statistics
        if called in the same iteration as forward().
        """
        training_state = self.training
        # Temporarily switch to eval mode to freeze BN stats
        self.eval()

        try:
            features = {}
            layer_idx = 0

            # Use internal _forward_impl logic if possible, or iterate
            # Since self.model is Sequential, we iterate layers.
            # We strictly don't want to update BN running stats here.

            # Need to clone x? No.
            current_x = x

            for _i, layer in enumerate(self.model):
                if isinstance(layer, nn.Conv2d):
                    current_x = layer(current_x)
                    features[f"conv_{layer_idx}"] = current_x
                    layer_idx += 1
                elif isinstance(layer, (nn.BatchNorm2d, nn.LeakyReLU)):
                    current_x = layer(current_x)
                else:
                    # Final activation
                    current_x = layer(current_x)

            return features

        finally:
            # Restore original training state
            if training_state:
                self.train()

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
