import torch
from torch import nn
from torch.autograd import Function


class GradientReversalLayer(Function):
    """Gradient Reversal Layer (GRL) for Domain-Adversarial Training.
    This layer has no parameters and simply reverses the gradient
    during backpropagation.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """forward.

        Args:
            ctx (Any): Description.
            x (torch.Tensor): Description.
            alpha (float): Description.
        Returns:
            torch.Tensor: Description.
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backward.

        Args:
            ctx (Any): Description.
            grad_output (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, None]: Description.
        """
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Applies the Gradient Reversal Layer."""
    return GradientReversalLayer.apply(x, alpha)


class DomainClassifier(nn.Module):
    """A simple domain classifier for DANN.
    It takes feature maps from a backbone network and predicts the domain.
    """

    def __init__(self, in_channels: int, num_domains: int = 2):
        """__init__.

        Args:
            in_channels (int): Description.
            num_domains (int): Description.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_domains),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            alpha (float): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DomainClassifier.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            alpha (float): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = grad_reverse(x, alpha)
        domain_output = self.classifier(features)
        return domain_output


class nn_Flatten(nn.Module):
    """nn_Flatten class."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for nn_Flatten.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return x.view(x.size(0), -1)
