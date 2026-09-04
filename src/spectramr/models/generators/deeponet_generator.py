import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class BranchNet(nn.Module):
    """Encodes the input function (e.g., undersampled MRI) into latent coefficients.

    Uses a CNN to process the input grid (image/k-space).
    feature_dim = basis_dim (p)
    Output shape: [B, basis_dim * channels_out]
    """

    def __init__(self, in_channels: int, basis_dim: int, out_channels: int, width: int = 64):
        """__init__.

        Args:
            in_channels (int): Description.
            basis_dim (int): Description.
            out_channels (int): Description.
            width (int): Description.
        """
        super().__init__()
        # Simple ResNet-like or CNN structure
        self.conv_in = nn.Conv2d(in_channels, width, 3, 1, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(width, width * 2, 4, 2, 1),  # /2
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1),  # /4
            nn.ReLU(),
            nn.Conv2d(width * 4, width * 8, 4, 2, 1),  # /8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Flatten spatial
        )
        self.fc = nn.Linear(width * 8, basis_dim * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for BranchNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.conv_in(x)
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class TrunkNet(nn.Module):
    """Encodes the query coordinates into basis functions.

    Input: [H*W, 2] (coordinates)
    Output: [H*W, basis_dim]
    """

    def __init__(self, input_dim: int = 2, basis_dim: int = 128, width: int = 128, depth: int = 3):
        """__init__.

        Args:
            input_dim (int): Description.
            basis_dim (int): Description.
            width (int): Description.
            depth (int): Description.
        """
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, basis_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TrunkNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


@register_model(name="deeponet", training_mode="reconstruction")
class DeepONetGenerator(nn.Module, IGenerator):
    """Deep Operator Network (DeepONet) for MRI.

    Learns the operator G: u -> G(u) where u is the input function (undersampled image)
    and G(u) is the reconstructed image function.

    G(u)(y) = Branch(u) • Trunk(y) + Bias

    Args:
        in_channels: Input channels (2 for complex k-space/image)
        out_channels: Output channels (2 for complex image)
        basis_dim: Number of basis functions (p)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        basis_dim: int = 128,
        hidden_dim: int = 64,
        img_size: tuple[int, int] = (64, 64),
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            basis_dim (int): Description.
            hidden_dim (int): Description.
            img_size (tuple[int, int]): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.img_size = img_size

        # Branch Net: [B, C, H, W] -> [B, basis_dim * out_channels]
        # We output coefficients for each output channel separately
        self.branch = BranchNet(in_channels, basis_dim, out_channels, width=hidden_dim)

        # Trunk Net: [N_points, 2] -> [N_points, basis_dim]
        # We assume 2D coordinates
        self.trunk = TrunkNet(input_dim=2, basis_dim=basis_dim, width=hidden_dim)

        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Coordinate cache
        self.grid_cache = {}

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "DeepONet"

    def get_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """get_grid.

        Args:
            height (int): Description.
            width (int): Description.
            device (torch.device): Description.
        Returns:
            torch.Tensor: Description.
        """
        key = (height, width, str(device))
        if key not in self.grid_cache:
            # Create meshgrid [-1, 1]
            y = torch.linspace(-1, 1, height, device=device)
            x = torch.linspace(-1, 1, width, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
            self.grid_cache[key] = grid.view(-1, 2)  # [H*W, 2]

        return self.grid_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, H, W]
        output: [B, out_channels, H, W]

        forward method for DeepONetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # 1. Branch: Encode input function u
        # coefficients: [B, basis_dim * out_channels]
        coeffs = self.branch(x)
        # Reshape to [B, out_channels, basis_dim]
        coeffs = coeffs.view(B, self.out_channels, self.basis_dim)

        # 2. Trunk: Encode query coordinates y
        # We assume output resolution matches input resolution (usually true for U-Net replacement)
        # coords: [H*W, 2]
        coords = self.get_grid(H, W, x.device)
        # basis: [H*W, basis_dim]
        basis = self.trunk(coords)

        # 3. Dot Product
        # G(u)(y) = sum_k b_k * t_k
        # coeffs: [B, C, p]
        # basis:  [N, p]
        # result: [B, C, N] = coeffs @ basis.T

        # [B, C, p] @ [p, N] -> [B, C, N]
        out_flat = torch.matmul(coeffs, basis.t())

        # Add bias
        # [B, C, N] + [C, 1]
        out_flat = out_flat + self.bias.view(1, -1, 1)

        # Reshape to image
        out = out_flat.view(B, self.out_channels, H, W)

        return out

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate output from input tensor x."""
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())
