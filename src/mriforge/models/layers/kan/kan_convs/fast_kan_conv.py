import inspect

import torch
from torch import nn

from .kans import RadialBasisFunction


class FastKANConvNDLayer(nn.Module):
    """FastKANConvNDLayer class."""

    def __init__(
        self,
        conv_class,
        norm_class,
        input_dim,
        output_dim,
        kernel_size,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        ndim: int = 2,
        grid_size=8,
        base_activation=nn.SiLU,
        grid_range=None,
        dropout=0.0,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            conv_class (Any): Description.
            norm_class (Any): Description.
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            ndim (int): Description.
            grid_size (Any): Description.
            base_activation (Any): Description.
            grid_range (Any): Description.
            dropout (Any): Description.
        """
        if grid_range is None:
            grid_range = [-2, 2]
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        if isinstance(base_activation, str):
            if base_activation.lower() == "silu":
                self.base_activation = nn.SiLU()
            elif base_activation.lower() == "relu":
                self.base_activation = nn.ReLU()
            elif base_activation.lower() == "gelu":
                self.base_activation = nn.GELU()
            elif base_activation.lower() == "identity":
                self.base_activation = nn.Identity()
            else:
                raise ValueError(f"Unsupported activation string: {base_activation}")
        else:
            self.base_activation = base_activation()
        self.grid_range = grid_range
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if input_dim % groups != 0:
            raise ValueError("input_dim must be divisible by groups")
        if output_dim % groups != 0:
            raise ValueError("output_dim must be divisible by groups")

        if norm_class is not None:
            # Drop unexpected arbitrary kwargs
            valid_params = inspect.signature(norm_class).parameters
            norm_kw = {k: v for k, v in norm_kwargs.items() if k in valid_params}

            # Handle GroupNorm specially since it has different argument order
            if norm_class == nn.GroupNorm:
                num_groups = norm_kwargs.get(
                    "num_groups", groups
                )  # get original groups not filtered
                if input_dim % num_groups != 0:
                    num_groups = 1  # Fallback to 1 group
                self.layer_norm = norm_class(num_groups, input_dim, **norm_kw)
            else:
                self.layer_norm = norm_class(input_dim, **norm_kw)
        else:
            self.layer_norm = None

        # 1. Replace ModuleList with single Grouped Convolution
        # This fuses G small kernels into 1 large kernel call
        self.base_conv = conv_class(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,  # Native Grouping
            bias=False,
        )

        # 2. Spline Conv also becomes monolithic
        # Input to spline conv is flattened grid basis: [B, C*Grid, ...]
        # We need to ensure groups align correctly.
        # Original: groups=1 for each of G splits.
        # New: groups=G, but input channels are C*Grid.
        # Each group should process (C/G)*Grid input channels and produce Output/G output channels.
        self.spline_conv = conv_class(
            input_dim * grid_size,
            output_dim,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,  # Native Grouping
            bias=False,
        )

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        # Initialize weights using Kaiming uniform distribution for better
        # training start
        nn.init.kaiming_uniform_(self.base_conv.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.spline_conv.weight, nonlinearity="linear")

    def forward(self, x):
        # No splits, no loops, no clones, no cats.

        # 1. Base Path
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for FastKANConvNDLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_base = self.base_activation(x)
        base_output = self.base_conv(x_base)

        # 2. Spline Path
        if self.dropout is not None:
            x = self.dropout(x)

        if self.layer_norm:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x

        # Apply RBF (Vectorized) [B, C, ...] -> [B, C, ..., G]
        spline_basis = self.rbf(x_norm)

        # Reshape for Spline Conv: Stack grid basis into channels
        # [B, C, ..., G] -> [B, C, G, ...] -> [B, C*G, ...]
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)

        spline_output = self.spline_conv(spline_basis)

        # 3. Fusion
        return base_output + spline_output


class FastKANConv3DLayer(FastKANConvNDLayer):
    """FastKANConv3DLayer class."""

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        grid_size=8,
        base_activation=nn.SiLU,
        grid_range=None,
        dropout=0.0,
        norm_layer=nn.InstanceNorm3d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            grid_size (Any): Description.
            base_activation (Any): Description.
            grid_range (Any): Description.
            dropout (Any): Description.
            norm_layer (Any): Description.
        """
        if grid_range is None:
            grid_range = [-2, 2]
        super().__init__(
            nn.Conv3d,
            norm_layer,
            input_dim,
            output_dim,
            kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=3,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            **norm_kwargs,
        )


class FastKANConv2DLayer(FastKANConvNDLayer):
    """FastKANConv2DLayer class."""

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        grid_size=8,
        base_activation=nn.SiLU,
        grid_range=None,
        dropout=0.0,
        norm_layer=nn.InstanceNorm2d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            grid_size (Any): Description.
            base_activation (Any): Description.
            grid_range (Any): Description.
            dropout (Any): Description.
            norm_layer (Any): Description.
        """
        if grid_range is None:
            grid_range = [-2, 2]
        super().__init__(
            nn.Conv2d,
            norm_layer,
            input_dim,
            output_dim,
            kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=2,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            **norm_kwargs,
        )


class FastKANConv1DLayer(FastKANConvNDLayer):
    """FastKANConv1DLayer class."""

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        grid_size=8,
        base_activation=nn.SiLU,
        grid_range=None,
        dropout=0.0,
        norm_layer=nn.InstanceNorm1d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            grid_size (Any): Description.
            base_activation (Any): Description.
            grid_range (Any): Description.
            dropout (Any): Description.
            norm_layer (Any): Description.
        """
        if grid_range is None:
            grid_range = [-2, 2]
        super().__init__(
            nn.Conv1d,
            norm_layer,
            input_dim,
            output_dim,
            kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=1,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            **norm_kwargs,
        )


class FourierKANConvNDLayer(nn.Module):
    """
    Fourier KAN Convolution Layer.
    Learns activation functions in the Frequency Domain (Spectral KAN).
    """

    def __init__(
        self,
        conv_class,
        norm_class,
        input_dim,
        output_dim,
        kernel_size,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        ndim: int = 2,
        grid_size=8,
        base_activation=nn.SiLU,
        grid_range=None,
        dropout=0.0,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            conv_class (Any): Description.
            norm_class (Any): Description.
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            ndim (int): Description.
            grid_size (Any): Description.
            base_activation (Any): Description.
            grid_range (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()

        # Base Linear Transform (Spatial Convolution)
        self.base_conv = conv_class(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            stride,  # typo in my thought? no, keep args order
            dilation,
            groups=groups,
            bias=False,
        )
        # Fix args: kernel, stride, padding, dilation
        # Previous call was: kernel_size, stride, padding, dilation
        # Let's match typical conv signature or just use kwargs map if unsure.
        # But I need to be precise.
        # fast_kan_conv init: kernel_size, stride, padding, dilation...
        self.base_conv = conv_class(
            input_dim,
            output_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        if norm_class:
            valid_params = inspect.signature(norm_class).parameters
            norm_kw = {k: v for k, v in norm_kwargs.items() if k in valid_params}
            self.layer_norm = norm_class(input_dim, **norm_kw)
        else:
            self.layer_norm = None

        nn.init.kaiming_uniform_(self.base_conv.weight, nonlinearity="linear")


class FourierKANConv2DLayer(FourierKANConvNDLayer):
    """FourierKANConv2DLayer class."""

    def __init__(self, input_dim, output_dim, kernel_size, **kwargs):
        """__init__.

        Args:
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
        """
        super().__init__(
            nn.Conv2d,
            nn.InstanceNorm2d,
            input_dim,
            output_dim,
            kernel_size,
            ndim=2,
            **kwargs,
        )

        # FNO Weights: [Out, In, Modes1, Modes2] Complex
        scale = 1 / (input_dim * output_dim)
        self.weights1 = nn.Parameter(
            scale * torch.randn(input_dim, output_dim, self.grid_size, self.grid_size, 2)
        )

    def forward(self, x):
        # 1. Base
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        base_out = self.base_conv(self.base_activation(x))

        # 2. Spectral (FNO)
        # x: [B, C, H, W]
        if self.layer_norm:
            x = self.layer_norm(x)

        x_fft = torch.fft.rfft2(x, norm="ortho")

        # Modes to keep
        modes1 = min(self.grid_size, x_fft.shape[-2])
        modes2 = min(self.grid_size, x_fft.shape[-1])

        # Complex multiplication on low freq corners
        # x_fft_corner: [B, In, M1, M2]
        x_fft_corner = x_fft[..., :modes1, :modes2]

        # Weights: [In, Out, M1, M2] (Complex view)
        w_complex = torch.view_as_complex(self.weights1)  # [In, Out, M1, M2]
        # Resize if grid size > actual size
        w_used = w_complex[..., :modes1, :modes2]

        # Einstein Summation: B I X Y, I O X Y -> B O X Y
        out_fft_corner = torch.einsum("bixy,ioxy->boxy", x_fft_corner, w_used)

        # Pad back to full size
        out_fft = torch.zeros(
            x.shape[0],
            self.outdim,
            x_fft.shape[-2],
            x_fft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        out_fft[..., :modes1, :modes2] = out_fft_corner

        out_spectral = torch.fft.irfft2(out_fft, s=(x.shape[-2], x.shape[-1]), norm="ortho")

        # Adjust dimensions if padding changed size slightly
        if out_spectral.shape != base_out.shape:
            out_spectral = out_spectral[..., : base_out.shape[-2], : base_out.shape[-1]]

        return base_out + out_spectral
