"""Wavelet KAN VAE.

VAE using Wavelet KAN layers for multiresolution feature extraction/prior.
"""

import torch
import torch.nn as nn

# In-repo import: an ImportError here is a bug that must surface — never
# substitute a plain-Conv2d mock for the wavelet KAN layer (pitfall #9/#16).
from spectramr.models.layers.kan.kan_convs.wav_kan import WavKANConv2DLayer
from spectramr.models.registry import register_model

# "daubechies" is the Daubechies-family type WavKANConv2DLayer supports;
# the historical "db2" label was only ever accepted by the removed mock.
_WAVELET_TYPE = "daubechies"


@register_model(name="vae_wavelet_kan_vae", training_mode="vae")
class WaveletKANVAE(nn.Module):
    """VAE with Wavelet KAN Encoder/Decoder."""

    def __init__(self, in_channels=1, latent_dim=128, **kwargs):
        """__init__.

        Args:
            in_channels (Any): Description.
            latent_dim (Any): Description.
        """
        super().__init__()
        # Blueprint: Use Wavelet KAN layers
        self.encoder = nn.Sequential(
            WavKANConv2DLayer(
                in_channels, 32, kernel_size=3, padding=1, wavelet_type=_WAVELET_TYPE
            ),
            nn.MaxPool2d(2),
            WavKANConv2DLayer(32, 64, kernel_size=3, padding=1, wavelet_type=_WAVELET_TYPE),
            nn.MaxPool2d(2),  # Added pooling for reduction
            nn.AdaptiveAvgPool2d((16, 16)),  # Ensure fixed spatial dims regardless of input size
            nn.Flatten(),
        )

        # Fixed flatten size after AdaptiveAvgPool2d(16, 16) with 64 channels
        flat_size = 64 * 16 * 16
        self.fc_mu_logvar = nn.Linear(flat_size, 2 * latent_dim)

        self.decoder_input = nn.Linear(latent_dim, flat_size)

        self.decoder_layers = nn.Sequential(
            WavKANConv2DLayer(64, 32, kernel_size=3, padding=1, wavelet_type=_WAVELET_TYPE),
            nn.Upsample(scale_factor=2),
            WavKANConv2DLayer(
                32, in_channels, kernel_size=3, padding=1, wavelet_type=_WAVELET_TYPE
            ),
            nn.Upsample(scale_factor=2),
            nn.Sigmoid(),  # or Identity
        )
        self._in_channels = in_channels

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

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for WaveletKANVAE.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        _orig_size = x.shape[2:]  # Store original spatial size
        flat = self.encoder(x)
        mu_logvar = self.fc_mu_logvar(flat)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        d_in = self.decoder_input(z)
        d_view = d_in.view(-1, 64, 16, 16)
        out = self.decoder_layers(d_view)

        # Resize to match original input spatial dims
        if out.shape[2:] != _orig_size:
            out = nn.functional.interpolate(
                out, size=_orig_size, mode="bilinear", align_corners=False
            )

        return out, mu, logvar
