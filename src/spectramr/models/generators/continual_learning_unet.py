import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import StandardUNetGenerator
from spectramr.models.registry import register_model

__all__ = ["ContinualLearningUNet"]


@register_model(name="continual_learning_unet", training_mode="reconstruction")
class ContinualLearningUNet(nn.Module, IGenerator):
    """
    Continual Learning U-Net with Elastic Weight Consolidation (EWC) support (Experiment 108).

    Wraps a standard U-Net and adds methods to store Fisher Information Matrix
    and previous parameters for EWC loss calculation.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.generator = StandardUNetGenerator(in_channels, out_channels)

        # EWC storage
        self.register_buffer("ewc_lambda", torch.tensor(0.0))  # Strength of EWC
        self.fisher = {}  # Fisher Information Matrix (diagonal)
        self.optpar = {}  # Optimal parameters from previous task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ContinualLearningUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.generator(x)

    def store_fisher_and_params(self, dataloader, device, num_samples=100):
        """
        Compute and store Fisher Information Matrix and current parameters.
        Call this after finishing a task.
        """
        self.eval()
        fisher = {
            n: torch.zeros_like(p) for n, p in self.generator.named_parameters() if p.requires_grad
        }

        # 1. Compute Fisher (diagonal approximation)
        # F_i = E [ (grad_i log p(x))^2 ]
        # We approximate by accumulating squared gradients of the loss
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break

            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            self.zero_grad()
            outputs = self(inputs)
            loss = nn.MSELoss()(outputs, targets)  # Use task loss
            loss.backward()

            for n, p in self.generator.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data**2

            count += inputs.shape[0]

        # Normalize
        for n in fisher:
            fisher[n] /= count
            self.fisher[n] = fisher[n]

        # 2. Store current parameters
        self.optpar = {n: p.data.clone() for n, p in self.generator.named_parameters()}

    def ewc_loss(self):
        """
        Calculate EWC penalty loss: sum_i (F_i * (theta_i - theta_old_i)^2)
        """
        loss = 0
        if not self.fisher:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        for n, p in self.generator.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optpar[n]) ** 2).sum().item()

        return self.ewc_lambda * loss

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return self.generator.get_parameter_count()

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return self.generator.get_output_shape(input_shape)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "continual_learning_unet"
