import torch
from torch import nn

from spectramr.models.layers.kan.kan_convs.fast_kan_conv import FastKANConv2DLayer
from spectramr.models.layers.kan.kan_convs.kans.kan import KAN
from spectramr.models.registry import register_model


class MappingNetwork(nn.Module):
    """MappingNetwork class."""

    def __init__(self, z_dim, w_dim):
        """__init__.

        Args:
            z_dim (Any): Description.
            w_dim (Any): Description.
        """
        super().__init__()
        # Make the mapping network more flexible with dimension adaptation
        self.z_dim = z_dim
        self.w_dim = w_dim
        # The KAN mapping IS the registered mechanism of style_kan_gan;
        # construction failure must propagate, never degrade to an MLP
        # (pitfall #9/#16).
        self.mapping = KAN([z_dim] + [512] * 7 + [w_dim])

    def forward(self, x):
        # Ensure input tensor has the expected dimensions
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for MappingNetwork.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Check if input dimension matches expected z_dim
        if x.size(1) != self.z_dim:
            # print(f"Warning: Input dim {x.size(1)} doesn't match z_dim {self.z_dim}")
            if x.size(1) > self.z_dim:
                x = x[:, : self.z_dim]  # Truncate
            else:
                # Pad with zeros
                padding = self.z_dim - x.size(1)
                x = torch.nn.functional.pad(x, (0, padding))

        return self.mapping(x)


class AdaIN(nn.Module):
    """AdaIN class."""

    def __init__(self, channels, w_dim):
        """__init__.

        Args:
            channels (Any): Description.
            w_dim (Any): Description.
        """
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.channels = channels
        self.w_dim = w_dim
        # KAN style transforms are the mechanism; no nn.Linear fallback
        # (pitfall #9/#16).
        self.style_scale_transform = KAN([w_dim, channels])
        self.style_shift_transform = KAN([w_dim, channels])

    def forward(self, image, w):
        """forward.

        Args:
            image (Any): Description.
            w (Any): Description.
        Returns:
            Any: Description.

        forward method for AdaIN.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            w (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        normalized_image = self.instance_norm(image)

        # Ensure w has correct dimensions
        if w.dim() > 2:
            w = w.view(w.size(0), -1)
        if w.size(1) != self.w_dim:
            if w.size(1) > self.w_dim:
                w = w[:, : self.w_dim]
            else:
                padding = self.w_dim - w.size(1)
                w = torch.nn.functional.pad(w, (0, padding))

        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image


class SynthesisBlock(nn.Module):
    """SynthesisBlock class."""

    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        kernel_size=3,
        padding=1,
        upsample=True,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            w_dim (Any): Description.
            kernel_size (Any): Description.
            padding (Any): Description.
            upsample (Any): Description.
        """
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if upsample else None
        )
        self.conv = FastKANConv2DLayer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.adain = AdaIN(out_channels, w_dim)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        """forward.

        Args:
            x (Any): Description.
            w (Any): Description.
        Returns:
            Any: Description.

        forward method for SynthesisBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            w (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, w)
        x = self.lrelu(x)
        return x


@register_model(name="style_kan_gan", training_mode="gan")
class StyleKANGAN(nn.Module):
    """StyleKANGAN class."""

    def __init__(self, out_channels=1, z_dim=512, w_dim=512, img_size=256):
        """__init__.

        Args:
            out_channels (Any): Description.
            z_dim (Any): Description.
            w_dim (Any): Description.
            img_size (Any): Description.
        """
        super().__init__()
        self.z_dim = z_dim
        self.mapping_network = MappingNetwork(z_dim, w_dim)
        self.starting_constant = nn.Parameter(torch.randn(1, 512, 4, 4))

        self.blocks = nn.ModuleList()
        in_ch = 512

        # Initial block from constant
        self.blocks.append(SynthesisBlock(in_ch, in_ch, w_dim, upsample=False))

        # From 4x4 to 256x256
        resolutions = [4, 8, 16, 32, 64, 128, 256]
        ch_map = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64}

        for i in range(len(resolutions) - 1):
            if resolutions[i + 1] > img_size:
                break
            in_ch = ch_map[resolutions[i]]
            out_ch = ch_map[resolutions[i + 1]]
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim))
            self.blocks.append(SynthesisBlock(out_ch, out_ch, w_dim, upsample=False))

        self.to_rgb = FastKANConv2DLayer(
            ch_map[img_size],
            out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, z):
        # z can be a single noise vector or a list for style mixing
        """forward.

        Args:
            z (Any): Description.
        Returns:
            Any: Description.

        forward method for StyleKANGAN.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if not isinstance(z, list):
            z = [z]

        w = [self.mapping_network(noise) for noise in z]

        # For simplicity, we are not implementing style mixing here, just using
        # the first w
        w = w[0]

        x = self.starting_constant.repeat(z[0].shape[0], 1, 1, 1)
        for block in self.blocks:
            x = block(x, w)

        return self.to_rgb(x)


def get_generator(
    in_channels=1,
    out_channels=1,
    z_dim=512,
    w_dim=512,
    img_size=256,
    **kwargs,
):
    """get_generator.

    Args:
        in_channels (Any): Description.
        out_channels (Any): Description.
        z_dim (Any): Description.
        w_dim (Any): Description.
        img_size (Any): Description.
    Returns:
        Any: Description.
    """
    return StyleKANGAN(out_channels, z_dim, w_dim, img_size)


def get_discriminator(in_channels=1, **kwargs):
    """get_discriminator.

    Args:
        in_channels (Any): Description.
    Returns:
        Any: Description.
    """
    from ..patch_gan_discriminator import PatchGANDiscriminator

    return PatchGANDiscriminator(in_channels)


def get_stylegan_kan_pair(
    in_channels=1,
    out_channels=1,
    z_dim=512,
    w_dim=512,
    img_size=256,
    **kwargs,
):
    """Get both generator and discriminator for StyleGAN-KAN"""
    generator = get_generator(in_channels, out_channels, z_dim, w_dim, img_size)
    discriminator = get_discriminator(in_channels)
    return generator, discriminator


class StyleGANKANGenerator:
    """StyleGAN-KAN generator wrapper class."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        z_dim=512,
        w_dim=512,
        img_size=256,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            z_dim (Any): Description.
            w_dim (Any): Description.
            img_size (Any): Description.
        """
        self.model = get_generator(in_channels, out_channels, z_dim, w_dim, img_size)

    def __call__(self, z):
        """__call__.

        Args:
            z (Any): Description.
        Returns:
            Any: Description.
        """
        return self.model(z)

    def forward(self, z):
        """forward.

        Args:
            z (Any): Description.
        Returns:
            Any: Description.
        """
        return self.model(z)


class StyleGANKANDiscriminator:
    """StyleGAN-KAN discriminator wrapper class."""

    def __init__(self, in_channels=1, **kwargs):
        """__init__.

        Args:
            in_channels (Any): Description.
        """
        self.model = get_discriminator(in_channels, **kwargs)

    def __call__(self, x):
        """__call__.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        return self.model(x)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        return self.model(x)
