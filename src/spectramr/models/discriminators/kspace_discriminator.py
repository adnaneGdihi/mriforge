"""K-Space Informed Discriminator
==============================

Frequency-domain discriminator for MRI super-resolution that operates on
k-space representations to better capture frequency-domain artifacts and
improve reconstruction quality.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.interfaces import IDiscriminator
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


@register_model(name="kspace_discriminator", training_mode="gan")
class KSpaceDiscriminator(nn.Module):
    """Discriminator that operates in both spatial and frequency domains.

    Combines spatial PatchGAN with frequency-domain discrimination on k-space
    magnitude and phase information.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 3,
        use_phase: bool = True,
        spatial_weight: float = 0.7,
        frequency_weight: float = 0.3,
        spectral_norm: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            base_channels (int): Description.
            num_layers (int): Description.
            use_phase (bool): Description.
            spatial_weight (float): Description.
            frequency_weight (float): Description.
            spectral_norm (bool): Description.
        """
        super().__init__()
        self.use_phase = use_phase
        self.spatial_weight = spatial_weight
        self.frequency_weight = frequency_weight
        self.spectral_norm = spectral_norm

        # Spatial discriminator (PatchGAN-like)
        self.spatial_disc = self._build_spatial_discriminator(
            in_channels,
            base_channels,
            num_layers,
            spectral_norm,
        )

        # Frequency discriminator
        # fft2c converts (B, C_real, H, W) → (B, 1, H, W) complex
        # so magnitude → 1ch, phase → 1ch, combined → 2ch
        freq_channels = 2 if use_phase else 1
        self.frequency_disc = self._build_frequency_discriminator(
            freq_channels,
            base_channels,
            num_layers,
            spectral_norm,
        )

        # Output layers — compute final channels dynamically from network depth
        final_channels = min(base_channels * (2**num_layers), 512)
        self.spatial_out = nn.Conv2d(final_channels, 1, kernel_size=1)
        self.frequency_out = nn.Conv2d(final_channels, 1, kernel_size=1)

    def _build_spatial_discriminator(
        self,
        in_channels: int,
        base_channels: int,
        num_layers: int,
        spectral_norm: bool = True,
    ) -> nn.Sequential:
        """Build spatial discriminator network."""
        layers = []
        channels = in_channels

        def conv_layer(in_c, out_c, k, s, p):
            """conv_layer.

            Args:
                in_c (Any): Description.
                out_c (Any): Description.
                k (Any): Description.
                s (Any): Description.
                p (Any): Description.
            Returns:
                Any: Description.
            """
            conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            return nn.utils.spectral_norm(conv) if spectral_norm else conv

        for i in range(num_layers + 1):
            out_channels = min(base_channels * (2**i), 512)
            layers.extend(
                [
                    conv_layer(channels, out_channels, 4, 2, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ],
            )
            channels = out_channels

        return nn.Sequential(*layers)

    def _build_frequency_discriminator(
        self,
        in_channels: int,
        base_channels: int,
        num_layers: int,
        spectral_norm: bool = True,
    ) -> nn.Sequential:
        """Build frequency discriminator network."""
        layers = []
        channels = in_channels

        def conv_layer(in_c, out_c, k, s, p):
            """conv_layer.

            Args:
                in_c (Any): Description.
                out_c (Any): Description.
                k (Any): Description.
                s (Any): Description.
                p (Any): Description.
            Returns:
                Any: Description.
            """
            conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            return nn.utils.spectral_norm(conv) if spectral_norm else conv

        for i in range(num_layers + 1):
            out_channels = min(base_channels * (2**i), 512)
            layers.extend(
                [
                    conv_layer(channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv_layer(out_channels, out_channels, 3, 2, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ],
            )
            channels = out_channels

        return nn.Sequential(*layers)

    def _to_kspace(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert spatial image to k-space representation."""
        # Apply 2D FFT using physics module for proper centering
        kspace = fft2c(x)

        # Get magnitude
        magnitude = torch.abs(kspace)

        # Get phase if requested
        phase = None
        if self.use_phase:
            phase = torch.angle(kspace)

        return magnitude, phase

    def _combine_frequency_features(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine magnitude and phase into feature tensor."""
        if phase is not None:
            # Concatenate magnitude and phase
            return torch.cat([magnitude, phase], dim=1)
        return magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining spatial and frequency discrimination.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Discriminator output of shape (B, 1, H', W')

        """
        # Spatial discrimination
        spatial_features = self.spatial_disc(x)
        spatial_out = self.spatial_out(spatial_features)

        # Frequency discrimination
        magnitude, phase = self._to_kspace(x)
        freq_features = self._combine_frequency_features(magnitude, phase)
        freq_features = self.frequency_disc(freq_features)
        freq_out = self.frequency_out(freq_features)

        # Resize frequency output to match spatial output size
        if freq_out.shape[-2:] != spatial_out.shape[-2:]:
            freq_out = F.interpolate(
                freq_out,
                size=spatial_out.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Combine spatial and frequency predictions
        combined_out = self.spatial_weight * spatial_out + self.frequency_weight * freq_out

        return combined_out


@register_model(name="frequency_domain_discriminator", training_mode="gan")
class FrequencyDomainDiscriminator(nn.Module):
    """Pure frequency-domain discriminator operating only on k-space.

    Useful for applications where frequency-domain artifacts are the primary
    concern (e.g., undersampling artifacts, aliasing).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 4,
        use_phase: bool = True,
        center_crop: tuple[int, int] | None = None,
        spectral_norm: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            base_channels (int): Description.
            num_layers (int): Description.
            use_phase (bool): Description.
            center_crop (Optional[tuple[int, int]]): Description.
            spectral_norm (bool): Description.
        """
        super().__init__()
        self.use_phase = use_phase
        self.center_crop = center_crop
        self.spectral_norm = spectral_norm

        # Frequency feature extraction
        freq_channels = in_channels * 2 if use_phase else in_channels
        self.freq_encoder = self._build_frequency_encoder(
            freq_channels,
            base_channels,
            num_layers,
            spectral_norm,
        )

        # Output layer
        self.output = nn.Conv2d(
            base_channels * (2 ** (num_layers - 1)),
            1,
            kernel_size=1,
        )

    def _build_frequency_encoder(
        self,
        in_channels: int,
        base_channels: int,
        num_layers: int,
        spectral_norm: bool = True,
    ) -> nn.Sequential:
        """Build frequency encoder network."""
        layers = []
        channels = in_channels

        def conv_layer(in_c, out_c, k, s, p):
            """conv_layer.

            Args:
                in_c (Any): Description.
                out_c (Any): Description.
                k (Any): Description.
                s (Any): Description.
                p (Any): Description.
            Returns:
                Any: Description.
            """
            conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            return nn.utils.spectral_norm(conv) if spectral_norm else conv

        for i in range(num_layers):
            out_channels = base_channels * (2**i)
            layers.extend(
                [
                    conv_layer(channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv_layer(out_channels, out_channels, 3, 2, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ],
            )
            channels = out_channels

        return nn.Sequential(*layers)

    def _apply_center_crop(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply center cropping to focus on low-frequency components."""
        if self.center_crop is None:
            return magnitude, phase

        h_crop, w_crop = self.center_crop
        h, w = magnitude.shape[-2:]

        if h_crop >= h or w_crop >= w:
            return magnitude, phase

        h_start = (h - h_crop) // 2
        w_start = (w - w_crop) // 2

        magnitude = magnitude[
            ...,
            h_start : h_start + h_crop,
            w_start : w_start + w_crop,
        ]
        if phase is not None:
            phase = phase[..., h_start : h_start + h_crop, w_start : w_start + w_crop]

        return magnitude, phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frequency-domain discriminator.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Discriminator output of shape (B, 1, H', W')

        """
        # Convert to k-space
        magnitude, phase = self._to_kspace(x)

        # Apply center cropping if specified
        magnitude, phase = self._apply_center_crop(magnitude, phase)

        # Combine features
        freq_features = self._combine_frequency_features(magnitude, phase)

        # Encode frequency features
        encoded = self.freq_encoder(freq_features)

        # Output prediction
        out = self.output(encoded)

        return out

    def _to_kspace(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert spatial image to k-space representation."""
        # Apply 2D FFT using physics module for proper centering
        kspace = fft2c(x)

        # Get magnitude
        magnitude = torch.abs(kspace)

        # Get phase if requested
        phase = None
        if self.use_phase:
            phase = torch.angle(kspace)

        return magnitude, phase

    def _combine_frequency_features(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine magnitude and phase into feature tensor."""
        if phase is not None:
            # Concatenate magnitude and phase
            return torch.cat([magnitude, phase], dim=1)
        return magnitude


@register_model(name="kspace_aware_discriminator", training_mode="gan")
class KSpaceAwareDiscriminator(IDiscriminator):
    """Full implementation of k-space aware discriminator implementing
    IDiscriminator interface.

    Combines spatial and frequency discrimination with configurable weighting.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 3,
        use_phase: bool = True,
        spatial_weight: float = 0.6,
        frequency_weight: float = 0.4,
        center_crop: tuple[int, int] | None = None,
        spectral_norm: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            base_channels (int): Description.
            num_layers (int): Description.
            use_phase (bool): Description.
            spatial_weight (float): Description.
            frequency_weight (float): Description.
            center_crop (Optional[tuple[int, int]]): Description.
            spectral_norm (bool): Description.
        """
        super().__init__()
        self.spatial_disc = KSpaceDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            use_phase=use_phase,
            spatial_weight=spatial_weight,
            frequency_weight=frequency_weight,
            spectral_norm=spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.spatial_disc(x)

    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Discriminate real vs fake samples."""
        return self.forward(x)

    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract feature maps for feature matching."""
        # For now, return empty dict as feature extraction is not implemented # IMPL
        return {}

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "KSpaceAwareDiscriminator"

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        # Simplified calculation - actual shape depends on network architecture
        batch_size, channels, height, width = input_shape
        # Approximate output size after downsampling
        out_height = height // (2 ** self.spatial_disc.spatial_disc.__len__())
        out_width = width // (2 ** self.spatial_disc.spatial_disc.__len__())
        return (batch_size, 1, out_height, out_width)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_kspace_discriminator(
    in_channels: int = 1,
    base_channels: int = 64,
    **kwargs,
) -> KSpaceAwareDiscriminator:
    """Factory function for creating k-space aware discriminator.

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels for network
        **kwargs: Additional arguments passed to KSpaceAwareDiscriminator

    Returns:
        Configured k-space aware discriminator

    """
    return KSpaceAwareDiscriminator(
        in_channels=in_channels,
        base_channels=base_channels,
        **kwargs,
    )


def create_frequency_discriminator(
    in_channels: int = 1,
    base_channels: int = 64,
    **kwargs,
) -> FrequencyDomainDiscriminator:
    """Factory function for creating pure frequency-domain discriminator.

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels for network
        **kwargs: Additional arguments passed to FrequencyDomainDiscriminator

    Returns:
        Configured frequency-domain discriminator

    """
    return FrequencyDomainDiscriminator(
        in_channels=in_channels,
        base_channels=base_channels,
        **kwargs,
    )


if __name__ == "__main__":
    # Quick test
    disc = create_kspace_discriminator(in_channels=1, base_channels=64)

    # Test with sample input
    x = torch.randn(2, 1, 128, 128)
    out = disc(x)

    logger.debug(f"Input shape: {x.shape}")
    logger.debug(f"Output shape: {out.shape}")
    logger.debug(f"Parameter count: {disc.get_parameter_count()}")

    # Test frequency discriminator
    freq_disc = create_frequency_discriminator(in_channels=1, base_channels=32)
    freq_out = freq_disc(x)

    logger.debug(f"Frequency disc output shape: {freq_out.shape}")
    freq_params = sum(p.numel() for p in freq_disc.parameters())
    logger.debug(f"Frequency disc parameters: {freq_params}")
