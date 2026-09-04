from torch import nn

from spectramr.models.layers.kan.kanu_net import KANU_Net
from spectramr.models.registry import register_model

# This file leverages the existing KANU_Net as the KAN GAN implementation.
# We just need to provide the standard getter functions.


def get_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    bilinear: bool = False,
    norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    opt: dict | None = None,
    **kwargs,
):
    """Gets the KAN-based U-Net (KANU_Net) generator."""
    return KANU_Net(
        in_channels=in_channels,
        out_channels=out_channels,
        bilinear=bilinear,
        norm_layer=norm_layer,
        opt=opt,
    )


def get_discriminator(in_channels=1, **kwargs):
    """Gets the PatchGAN discriminator.
    For a fair comparison, we can use the same discriminator across models.
    """
    try:
        from ..discriminators.patchgan_discriminator import PatchGANDiscriminator
    except ImportError:
        from ..discriminators.patchgan_discriminator import PatchGANDiscriminator

    return PatchGANDiscriminator(in_channels)


@register_model(name="kan_gan", training_mode="gan")
class KANGenerator:
    """KAN-based generator wrapper class."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        bilinear: bool = False,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        opt: dict | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bilinear (bool): Description.
            norm_layer (type[nn.Module]): Description.
            opt (dict | None): Description.
        """
        self.model = get_generator(
            in_channels=in_channels,
            out_channels=out_channels,
            bilinear=bilinear,
            norm_layer=norm_layer,
            opt=opt,
            **kwargs,
        )

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


@register_model(name="kan_discriminator", training_mode="gan")
class KANDiscriminator:
    """KAN-based discriminator wrapper class."""

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
