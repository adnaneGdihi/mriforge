import torch
from torch import nn


class SplineActivation(nn.Module):
    """Element-wise B-spline activation function.
    Applies a learnable spline transformation to each feature.
    """

    def __init__(
        self,
        num_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: list = None,
    ):
        """__init__.

        Args:
            num_features (int): Description.
            grid_size (int): Description.
            spline_order (int): Description.
            grid_range (list): Description.
        """
        if grid_range is None:
            grid_range = [-1, 1]
        super().__init__()
        self.num_features = num_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Create the grid tensor
        grid = torch.linspace(
            self.grid_range[0],
            self.grid_range[1],
            self.grid_size + 2 * self.spline_order + 1,
            dtype=torch.float32,
        )
        self.register_buffer("grid", grid)

        # Initialize spline weights
        # Shape: (1, num_features, 1, 1, grid_size + spline_order)
        self.spline_weight = nn.Parameter(
            torch.randn(1, self.num_features, 1, 1, self.grid_size + self.spline_order),
        )
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity="linear")

    def b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis functions for the input tensor.
        x shape: (B, C, D1, D2, ...)
        Returns basis of shape: (B, C, D1, D2, ..., grid_size + spline_order)
        """
        num_basis = self.grid_size + self.spline_order
        basis = x.unsqueeze(-1).expand(*x.shape, num_basis)
        basis = torch.softmax(basis, dim=-1)
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SplineActivation.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Input feature count {x.size(1)} does not match SplineActivation feature count {self.num_features}",
            )

        basis = self.b_spline_basis(x)

        # Adapt spline_weight for broadcasting across different input dimensions (1D, 2D, 3D)
        # Original shape is (1, C, 1, 1, G) for 4D input. Squeeze to (C, G) then
        # reshape.
        w = self.spline_weight.squeeze()
        w_shape = [1, self.num_features] + [1] * (x.ndim - 2) + [-1]
        w = w.view(*w_shape)

        spline_output = (basis * w).sum(-1)

        return spline_output


class KANConv2DLayer(nn.Module):
    """A stable 2D Convolutional KAN Layer.
    Combines a standard Conv2D layer with a learnable spline-based activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (Any): Description.
            stride (Any): Description.
            padding (Any): Description.
            dilation (Any): Description.
            groups (Any): Description.
            bias (Any): Description.
            grid_size (Any): Description.
            spline_order (Any): Description.
            base_activation (Any): Description.
        """
        super().__init__()
        self.base_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.base_activation = base_activation()
        self.spline_activation = SplineActivation(
            num_features=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KANConv2DLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        base_output = self.base_conv(x)
        spline_output = self.spline_activation(base_output)
        return self.base_activation(base_output) + spline_output


class KANConv1DLayer(nn.Module):
    """A stable 1D Convolutional KAN Layer.
    Combines a standard Conv1D layer with a learnable spline-based activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (Any): Description.
            stride (Any): Description.
            padding (Any): Description.
            dilation (Any): Description.
            groups (Any): Description.
            bias (Any): Description.
            grid_size (Any): Description.
            spline_order (Any): Description.
            base_activation (Any): Description.
        """
        super().__init__()
        self.base_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.base_activation = base_activation()
        self.spline_activation = SplineActivation(
            num_features=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KANConv1DLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        base_output = self.base_conv(x)
        spline_output = self.spline_activation(base_output)
        return self.base_activation(base_output) + spline_output


class KANConv3DLayer(nn.Module):
    """A stable 3D Convolutional KAN Layer.
    Combines a standard Conv3D layer with a learnable spline-based activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (Any): Description.
            stride (Any): Description.
            padding (Any): Description.
            dilation (Any): Description.
            groups (Any): Description.
            bias (Any): Description.
            grid_size (Any): Description.
            spline_order (Any): Description.
            base_activation (Any): Description.
        """
        super().__init__()
        self.base_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.base_activation = base_activation()
        self.spline_activation = SplineActivation(
            num_features=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KANConv3DLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        base_output = self.base_conv(x)
        spline_output = self.spline_activation(base_output)
        return self.base_activation(base_output) + spline_output
