import torch
from torch import nn
from torch.nn.functional import conv1d, conv2d, conv3d


class KABNConvNDLayer(nn.Module):
    """KABNConvNDLayer class."""

    def __init__(
        self,
        conv_class,
        norm_class,
        conv_w_fun,
        input_dim,
        output_dim,
        degree,
        kernel_size,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        ndim: int = 2,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            conv_class (Any): Description.
            norm_class (Any): Description.
            conv_w_fun (Any): Description.
            input_dim (Any): Description.
            output_dim (Any): Description.
            degree (Any): Description.
            kernel_size (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            ndim (int): Description.
        """
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if input_dim % groups != 0:
            raise ValueError("input_dim must be divisible by groups")
        if output_dim % groups != 0:
            raise ValueError("output_dim must be divisible by groups")

        self.base_conv = nn.ModuleList(
            [
                conv_class(
                    input_dim // groups,
                    output_dim // groups,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups=1,
                    bias=False,
                )
                for _ in range(groups)
            ],
        )

        self.layer_norm = nn.ModuleList(
            [norm_class(output_dim // groups, output_dim, **norm_kwargs) for _ in range(groups)],
        )

        poly_shape = (
            groups,
            output_dim // groups,
            (input_dim // groups) * (degree + 1),
        ) + tuple(kernel_size for _ in range(ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))

        # Initialize weights using Kaiming uniform distribution for better
        # training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity="linear")

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity="linear")

    # @lru_cache(maxsize=128) # This is commented out as it can cause errors with tensor inputs
    def bernstein_poly(self, x, degree):
        """bernstein_poly.

        Args:
            x (Any): Description.
            degree (Any): Description.
        Returns:
            Any: Description.
        """
        bernsteins = torch.ones(
            x.shape + (self.degree + 1,),
            dtype=x.dtype,
            device=x.device,
        )
        for j in range(1, degree + 1):
            for k in range(degree + 1 - j):
                bernsteins[..., k] = bernsteins[..., k] * (1 - x) + bernsteins[..., k + 1] * x

        bernsteins = bernsteins.moveaxis(-1, 2)
        bernsteins = bernsteins.flatten(1, 2)

        return bernsteins

    def forward_kab(self, x, group_index):
        # Apply base activation to input and then linear transform with base
        # weights
        """forward_kab.

        Args:
            x (Any): Description.
            group_index (Any): Description.
        Returns:
            Any: Description.
        """
        base_output = self.base_conv[group_index](x)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial
        # computation
        x_normalized = torch.sigmoid(x)

        if self.dropout is not None:
            x_normalized = self.dropout(x_normalized)

        # Compute Legendre polynomials for the normalized x
        bernstein_basis = self.bernstein_poly(x_normalized, self.degree)
        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        # Compute polynomial output using polynomial weights
        poly_output = self.conv_w_fun(
            bernstein_basis,
            self.poly_weights[group_index],
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=1,
        )

        # poly_output = poly_output.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], self.outdim // self.groups)
        # Combine base and polynomial outputs, normalize, and activate
        x = base_output + poly_output
        if isinstance(self.layer_norm[group_index], nn.LayerNorm):
            orig_shape = x.shape
            x = self.layer_norm[group_index](x.view(orig_shape[0], -1)).view(orig_shape)
        else:
            x = self.layer_norm[group_index](x)
        x = self.base_activation(x)

        return x

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KABNConvNDLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kab(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KABNConv3DLayer(KABNConvNDLayer):
    """KABNConv3DLayer class."""

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        degree=3,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.InstanceNorm3d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            nn.Conv3d,
            norm_layer,
            conv3d,
            input_dim,
            output_dim,
            degree,
            kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=3,
            dropout=dropout,
            **norm_kwargs,
        )


class KABNConv2DLayer(KABNConvNDLayer):
    """KABNConv2DLayer class."""

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        degree=3,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.InstanceNorm2d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            nn.Conv2d,
            norm_layer,
            conv2d,
            input_dim,
            output_dim,
            degree,
            kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=2,
            dropout=dropout,
            **norm_kwargs,
        )


class KABNConv1DLayer(KABNConvNDLayer):
    """KABNConv1DLayer class."""

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        degree=3,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.InstanceNorm1d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            output_dim (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            nn.Conv1d,
            norm_layer,
            conv1d,
            input_dim,
            output_dim,
            degree,
            kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=1,
            dropout=dropout,
            **norm_kwargs,
        )
