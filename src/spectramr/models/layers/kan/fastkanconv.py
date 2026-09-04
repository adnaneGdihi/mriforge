import torch
import torch.nn.functional as F
from torch import nn


class PolynomialFunction(nn.Module):
    """PolynomialFunction class."""

    def __init__(self, degree: int = 3):
        """__init__.

        Args:
            degree (int): Description.
        """
        super().__init__()
        self.degree = degree

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for PolynomialFunction.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.stack([x**i for i in range(self.degree)], dim=-1)

    @property
    def num_grids(self):
        """num_grids.

        Returns:
            Any: Description.
        """
        return self.degree


class BSplineFunction(nn.Module):
    """BSplineFunction class."""

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        degree: int = 3,
        num_basis: int = 8,
    ):
        """__init__.

        Args:
            grid_min (float): Description.
            grid_max (float): Description.
            degree (int): Description.
            num_basis (int): Description.
        """
        super().__init__()
        self.degree = degree
        self.num_basis = num_basis
        self.knots = torch.linspace(
            grid_min,
            grid_max,
            num_basis + degree + 1,
        )  # Uniform knots

    def basis_function(self, i, k, t):
        """basis_function.

        Args:
            i (Any): Description.
            k (Any): Description.
            t (Any): Description.
        Returns:
            Any: Description.
        """
        if k == 0:
            return ((self.knots[i] <= t) & (t < self.knots[i + 1])).float()
        left_num = (t - self.knots[i]) * self.basis_function(i, k - 1, t)
        left_den = self.knots[i + k] - self.knots[i]
        left = left_num / left_den if left_den != 0 else torch.zeros_like(left_num)

        right_num = (self.knots[i + k + 1] - t) * self.basis_function(
            i + 1,
            k - 1,
            t,
        )
        right_den = self.knots[i + k + 1] - self.knots[i + 1]
        right = right_num / right_den if right_den != 0 else torch.zeros_like(right_num)

        return left + right

    def forward(self, x):
        # x = x.squeeze()  # Removed to support broadcasting for arbitrary shapes
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for BSplineFunction.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        basis_functions = torch.stack(  # type: ignore
            [self.basis_function(i, self.degree, x) for i in range(self.num_basis)],
            dim=-1,
        )
        return basis_functions

    @property
    def num_grids(self):
        """num_grids.

        Returns:
            Any: Description.
        """
        return self.num_basis


class ChebyshevFunction(nn.Module):
    """ChebyshevFunction class."""

    def __init__(self, degree: int = 4):
        """__init__.

        Args:
            degree (int): Description.
        """
        super().__init__()
        self.degree = degree

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for ChebyshevFunction.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        chebyshev_polynomials = [torch.ones_like(x), x]
        for _n in range(2, self.degree):
            chebyshev_polynomials.append(
                2 * x * chebyshev_polynomials[-1] - chebyshev_polynomials[-2],
            )
        return torch.stack(chebyshev_polynomials, dim=-1)

    @property
    def num_grids(self):
        """num_grids.

        Returns:
            Any: Description.
        """
        return self.degree


class FourierBasisFunction(nn.Module):
    """FourierBasisFunction class."""

    def __init__(self, num_frequencies: int = 4, period: float = 1.0):
        """__init__.

        Args:
            num_frequencies (int): Description.
            period (float): Description.
        """
        super().__init__()
        assert num_frequencies % 2 == 0, "num_frequencies must be even"
        self.num_frequencies = num_frequencies
        self.period = nn.Parameter(torch.Tensor([period]), requires_grad=False)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for FourierBasisFunction.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        frequencies = torch.arange(1, self.num_frequencies // 2 + 1, device=x.device)
        sin_components = torch.sin(
            2 * torch.pi * frequencies * x[..., None] / self.period,
        )
        cos_components = torch.cos(
            2 * torch.pi * frequencies * x[..., None] / self.period,
        )
        basis_functions = torch.cat(  # type: ignore
            [sin_components, cos_components],
            dim=-1,
        )
        return basis_functions

    @property
    def num_grids(self):
        """num_grids.

        Returns:
            Any: Description.
        """
        return self.num_frequencies


class RadialBasisFunction(nn.Module):
    """RadialBasisFunction class."""

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 4,
        denominator: float = None,
    ):
        """__init__.

        Args:
            grid_min (float): Description.
            grid_max (float): Description.
            num_grids (int): Description.
            denominator (float): Description.
        """
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for RadialBasisFunction.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))

    @property
    def num_grids(self):
        """num_grids.

        Returns:
            Any: Description.
        """
        return len(self.grid)


class SplineConv2D(nn.Conv2d):
    """SplineConv2D class."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        init_scale: float = 0.1,
        padding_mode: str = "zeros",
        **kw,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (Union[int, tuple[int, int]]): Description.
            stride (Union[int, tuple[int, int]]): Description.
            padding (Union[int, tuple[int, int]]): Description.
            dilation (Union[int, tuple[int, int]]): Description.
            groups (int): Description.
            bias (bool): Description.
            init_scale (float): Description.
            padding_mode (str): Description.
        """
        self.init_scale = init_scale
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            **kw,
        )

    def reset_parameters(self) -> None:
        # nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)  #
        # Commented out - WeightInitializationManager will handle this
        """reset_parameters."""
        if self.bias is not None:
            # nn.init.zeros_(self.bias)  # Commented out - WeightInitializationManager
            # will handle this
            pass  # Keep structure but no initialization


class FastKANConvLayer(nn.Module):
    """FastKANConvLayer class."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = "same",
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 4,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
        padding_mode: str = "zeros",
        kan_type: str = "RBF",
        dropout=0.0,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (Union[int, tuple[int, int]]): Description.
            stride (Union[int, tuple[int, int]]): Description.
            padding (Union[int, tuple[int, int], str]): Description.
            dilation (Union[int, tuple[int, int]]): Description.
            groups (int): Description.
            bias (bool): Description.
            grid_min (float): Description.
            grid_max (float): Description.
            num_grids (int): Description.
            use_base_update (bool): Description.
            base_activation (Any): Description.
            spline_weight_init_scale (float): Description.
            padding_mode (str): Description.
            kan_type (str): Description.
            dropout (Any): Description.
        """
        super().__init__()

        # Handle "same" padding
        if isinstance(padding, str) and padding == "same":
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(k // 2 for k in kernel_size)

        if kan_type == "RBF":
            self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        elif kan_type == "Fourier":
            self.rbf = FourierBasisFunction(num_grids)
        elif kan_type == "Poly":
            self.rbf = PolynomialFunction(num_grids)
        elif kan_type == "Chebyshev":
            self.rbf = ChebyshevFunction(num_grids)
        elif kan_type == "BSpline":
            self.rbf = BSplineFunction(grid_min, grid_max, 3, num_grids)
        else:
            raise ValueError(
                f"Invalid kan_type: {kan_type}. Must be one of: RBF, Fourier, Poly, Chebyshev, BSpline"
            )

        self.spline_conv = SplineConv2D(
            in_channels * num_grids,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            spline_weight_init_scale,
            padding_mode,
        )

        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
            )

        # Add layer normalization for input normalization before RBF
        self.layer_norm = nn.InstanceNorm2d(in_channels)

        # Initialize weights only for layers that exist
        conv_layers = [self.spline_conv]
        if use_base_update:
            conv_layers.append(self.base_conv)

        for conv_layer in conv_layers:
            # Initialize weights properly
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity="linear")
            if conv_layer.bias is not None:
                nn.init.zeros_(conv_layer.bias)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        # nn.init.kaiming_normal_(self.base_conv.weight, nonlinearity='linear')  # Commented out - WeightInitializationManager will handle this
        # nn.init.kaiming_normal_(self.spline_conv.weight, nonlinearity='linear')
        # # Commented out - WeightInitializationManager will handle this

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for FastKANConvLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle 5D multi-coil/volumetric input: [B, C, D, H, W] → [B*D, C, H, W]
        reshaped_5d = False
        original_shape = x.shape
        if x.ndim == 5:
            reshaped_5d = True
            B, C, D, H, W = x.shape
            x = x.reshape(B * D, C, H, W)

        batch_size, channels, height, width = x.shape

        # Apply dropout first if specified
        if self.dropout is not None:
            x = self.dropout(x)

        # Apply layer normalization to normalize input before RBF
        x_norm = self.layer_norm(x)

        # Apply RBF globally using broadcasting (Vectorized)
        # Input: (B, C, H, W)
        # Output: (B, C, H, W, num_grids)
        x_rbf = self.rbf(x_norm)

        # Permute to: (batch, channels, num_grids, height, width)
        x_rbf = x_rbf.permute(0, 1, 4, 2, 3)

        # Flatten grids and channels: (batch, channels * num_grids, height, width)
        x_rbf = x_rbf.contiguous().view(batch_size, -1, height, width)

        # Apply spline convolution
        ret = self.spline_conv(x_rbf)

        if self.use_base_update:
            base = self.base_conv(self.base_activation(x))
            ret = ret + base

        # Restore 5D shape if input was 5D
        if reshaped_5d:
            out_C = ret.shape[1]
            ret = ret.reshape(B, D, out_C, height, width).permute(0, 2, 1, 3, 4)

        return ret

    def update_grid_size(self, new_num_grids: int):
        """
        Update the grid size of the KAN layer (Spectral Continuation).
        Re-initializes the RBF/Basis functions and the Spline Convolution with new dimensions.
        """
        if self.rbf.num_grids == new_num_grids:
            return

        # 1. Update RBF/Basis Function
        if isinstance(self.rbf, RadialBasisFunction):
            # Preserve grid range
            grid_min = self.rbf.grid[0].item()
            grid_max = self.rbf.grid[-1].item()
            self.rbf = RadialBasisFunction(grid_min, grid_max, new_num_grids)
        elif isinstance(self.rbf, FourierBasisFunction):
            self.rbf = FourierBasisFunction(new_num_grids)
        elif isinstance(self.rbf, PolynomialFunction):
            self.rbf = PolynomialFunction(new_num_grids)
        elif isinstance(self.rbf, ChebyshevFunction):
            self.rbf = ChebyshevFunction(new_num_grids)
        elif isinstance(self.rbf, BSplineFunction):
            grid_min = self.rbf.knots[0].item()
            grid_max = self.rbf.knots[-1].item()
            self.rbf = BSplineFunction(grid_min, grid_max, self.rbf.degree, new_num_grids)

        # 2. Update Spline Convolution
        # Input channels for spline conv = in_channels * num_grids
        in_channels = (
            self.base_conv.in_channels if self.use_base_update else self.layer_norm.num_features
        )
        out_channels = self.spline_conv.out_channels

        new_in_channels_spline = in_channels * new_num_grids

        # Create new convolution provided it matches the old configuration
        new_spline_conv = SplineConv2D(
            new_in_channels_spline,
            out_channels,
            self.spline_conv.kernel_size,
            self.spline_conv.stride,
            self.spline_conv.padding,
            self.spline_conv.dilation,
            self.spline_conv.groups,
            self.spline_conv.bias is not None,
            self.spline_conv.init_scale,
            self.spline_conv.padding_mode,
        ).to(self.spline_conv.weight.device)

        # Initialize
        nn.init.kaiming_normal_(new_spline_conv.weight, nonlinearity="linear")
        if new_spline_conv.bias is not None:
            nn.init.zeros_(new_spline_conv.bias)

        self.spline_conv = new_spline_conv
