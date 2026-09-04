"""Physics-Informed Momentum Networks (PIMN)
=========================================

Implementation of Experiment 3.2 from the Research Roadmap.
Treats reconstruction as a dynamic system with momentum-accelerated optimization.
"""

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.integration import create_physics_operator
from spectramr.infrastructure.physics.interfaces import IPhysicsOperator
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class PIMNLayer(nn.Module):
    """Single iteration of PIMN with momentum."""

    def __init__(
        self,
        operator: IPhysicsOperator,
        in_channels: int = 2,
        hidden_channels: int = 32,
        kernel_size: int = 3,
    ):
        """__init__.

        Args:
            operator (IPhysicsOperator): Description.
            in_channels (int): Description.
            hidden_channels (int): Description.
            kernel_size (int): Description.
        """
        super().__init__()
        self.operator = operator

        # Regularization Network (Gradient of R(x))
        # Simple CNN to learn the prior gradient
        self.reg_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size, padding=kernel_size // 2),
        )

        # Learnable step size and momentum
        self.epsilon = nn.Parameter(torch.tensor(0.1))
        self.mu = nn.Parameter(torch.tensor(0.9))

    def forward(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y: torch.Tensor,  # Measured k-space
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update rule:
        x_{k+1} = x_k + epsilon * (F^T(y - F x_k) - grad_R(x_k)) + mu * (x_k - x_{k-1})

        forward method for PIMNLayer.

        Executes PyTorch tensor operations.

        Args:
            x_curr (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_prev (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Data Consistency Gradient: F^T(y - F x_k)
        # Forward: x_k -> k-space

        # Handle Re/Im input (B, 2, H, W) -> make complex for operator
        if x_curr.shape[1] == 2:
            # (B, 2, H, W) -> (B, H, W, 2) -> (B, H, W) complex
            input_complex = torch.view_as_complex(x_curr.permute(0, 2, 3, 1).contiguous())
            k_pred = self.operator.forward(input_complex)
        else:
            k_pred = self.operator.forward(x_curr)

        # Residual in k-space (masked)
        # y is already masked usually, but let's ensure consistency
        # If y is full k-space, we apply mask. If y is undersampled, it's already masked.
        # We assume y is the acquired (undersampled) k-space.
        k_residual = y - k_pred * mask.unsqueeze(1)

        # Adjoint: k-space -> image
        grad_dc = self.operator.adjoint(k_residual)

        if grad_dc.is_complex():
            # Convert back to Real/Imag format matching x_curr
            grad_dc_real = torch.view_as_real(grad_dc)

            if grad_dc_real.dim() == 5:
                # (B, Coils, H, W, 2) -> (B, Coils, 2, H, W) -> (B, Coils*2, H, W)
                grad_dc = grad_dc_real.permute(0, 1, 4, 2, 3).contiguous()
                grad_dc = grad_dc.view(x_curr.shape[0], -1, x_curr.shape[-2], x_curr.shape[-1])
            else:
                # (B, H, W, 2) -> (B, 2, H, W)
                grad_dc = grad_dc_real.permute(0, 3, 1, 2).contiguous()

        # 2. Regularization Gradient: grad_R(x_k)
        grad_reg = self.reg_net(x_curr)

        # 3. Momentum Term
        momentum = x_curr - x_prev

        # Update
        x_next = x_curr + self.epsilon * (grad_dc - grad_reg) + self.mu * momentum

        return x_next


@register_model(name="pimn", training_mode="reconstruction")
class PIMN(nn.Module, IGenerator):
    """Physics-Informed Momentum Network."""

    def __init__(
        self,
        in_channels: int = 2,  # Real/Imag
        out_channels: int = 1,  # Magnitude
        num_iterations: int = 10,
        operator_type: str = "fft2d",
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_iterations (int): Description.
            operator_type (str): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_iterations = num_iterations

        self.operator = create_physics_operator(operator_type)

        # Shared or independent layers?
        # Research implies "Deep Unrolled", usually shared or independent.
        # We'll use independent layers for flexibility.
        self.layers = nn.ModuleList(
            [PIMNLayer(self.operator, in_channels) for _ in range(num_iterations)]
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "PIMN"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate output from initial estimate."""
        return self.forward(z, **kwargs)

    def forward(
        self,
        x: torch.Tensor,  # Initial estimate (e.g. zero-filled)
        ref_kspace: torch.Tensor | None = None,  # Renamed from kspace for consistency
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Initial image estimate [B, 2, H, W] (Real/Imag) or [B, 1, H, W]
            ref_kspace: Reference k-space data [B, C, H, W, 2] or [B, 2, H, W]
            mask: Sampling mask [B, H, W]

        forward method for PIMN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ref_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # If no k-space provided, we create it from the input estimate
        if ref_kspace is None:
            # If x is magnitude [B, 1, H, W], we need complex representation
            if x.shape[1] == 1:
                # Convert magnitude to complex (assume zero phase)
                x_complex = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device)
                x_complex[:, 0] = x[:, 0]  # Real channel
                x = x_complex

            # Create k-space from image via forward operator
            ref_kspace = self.operator.forward(x)

            # If no mask, assume fully sampled
            if mask is None:
                mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device)

        # Apply undersampling to k-space if mask provided but kspace was generated
        if ref_kspace is not None and mask is not None:
            ref_kspace = ref_kspace * mask.unsqueeze(1)

        x_curr = x
        x_prev = x  # x_{-1} = x_0

        for layer in self.layers:
            x_next = layer(x_curr, x_prev, ref_kspace, mask)
            x_prev = x_curr
            x_curr = x_next

        # Return magnitude
        if x_curr.shape[1] == 2:  # Complex representation
            magnitude = torch.sqrt(x_curr[:, 0] ** 2 + x_curr[:, 1] ** 2 + 1e-8)
            return magnitude.unsqueeze(1)
        return x_curr
