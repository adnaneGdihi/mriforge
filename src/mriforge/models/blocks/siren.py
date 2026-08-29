import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    """
    Sine-activated representation layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            is_first (bool): Description.
            omega_0 (float): Description.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        # Exact Initialization from [Sitzmann et al., 2020]
        # Crucial for convergence of high-frequency spatial gradients
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = torch.sqrt(torch.tensor(6.0 / in_features)) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SirenLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    Implicit Neural Representation (INR) using Periodic Activation Functions (SIREN).
    Parameterizes continuous physical fields (B0, GNL) from sparse boundaries.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        out_features: int = 2,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 1.0,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            hidden_features (int): Description.
            hidden_layers (int): Description.
            out_features (int): Description.
            first_omega_0 (float): Description.
            hidden_omega_0 (float): Description.
        """
        super().__init__()

        self.net = []
        self.net.append(
            SirenLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)
        )

        for _ in range(hidden_layers):
            self.net.append(
                SirenLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.final_linear = nn.Linear(hidden_features, out_features)

        with torch.no_grad():
            bound = torch.sqrt(torch.tensor(6.0 / hidden_features)) / hidden_omega_0
            self.final_linear.weight.uniform_(-bound, bound)

        self.net = nn.Sequential(*self.net, self.final_linear)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Densely or sparsely sampled coordinates [..., 2] or [..., 3]
        Returns:
            Mapped values: [..., out_features]

        forward method for SIREN.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure coordinates are floats and require gradients if PINN loss is active
        # The caller handles .requires_grad_(True)
        out = self.net(coords)
        return out
