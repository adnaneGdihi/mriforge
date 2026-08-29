import torch.nn as nn


class SimpleDiscriminator(nn.Module):
    """A simple convolutional discriminator for lightweight GAN training."""

    def __init__(self, in_channels=1, base_filters=64):
        """__init__.

        Args:
            in_channels (Any): Description.
            base_filters (Any): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, base_filters, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_filters, base_filters * 2, 4, 2, 1)),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_filters * 2, base_filters * 4, 4, 2, 1)),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_filters * 4, 1, 4, 1, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for SimpleDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)
