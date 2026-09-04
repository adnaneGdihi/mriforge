import torch
from torch import nn


class KACNConvNDLayer(nn.Module):
    """KACNConvNDLayer class."""

    def __init__(
        self,
        conv_class,
        norm_class,
        input_dim,
        output_dim,
        degree,
        kernel_size,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        ndim: int = 2,
        dropout=0.0,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            conv_class (Any): Description.
            norm_class (Any): Description.
            input_dim (Any): Description.
            output_dim (Any): Description.
            degree (Any): Description.
            kernel_size (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            ndim (int): Description.
            dropout (Any): Description.
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
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.epsilon = 1e-7
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

        self.layer_norm = nn.ModuleList(
            [norm_class(output_dim // groups, output_dim, **norm_kwargs) for _ in range(groups)],
        )

        self.poly_conv = nn.ModuleList(
            [
                conv_class(
                    (degree + 1) * input_dim // groups,
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
        arange_buffer_size = (
            1,
            1,
            -1,
        ) + tuple(1 for _ in range(ndim))
        self.register_buffer(
            "arange",
            torch.arange(0, degree + 1, 1).view(*arange_buffer_size),
        )
        # Initialize weights using Kaiming uniform distribution for better
        # training start
        for conv_layer in self.poly_conv:
            nn.init.normal_(
                conv_layer.weight,
                mean=0.0,
                std=1 / (input_dim * (degree + 1) * kernel_size**ndim),
            )

    def forward_kacn(self, x, group_index):
        # Apply base activation to input and then linear transform with base
        # weights
        """forward_kacn.

        Args:
            x (Any): Description.
            group_index (Any): Description.
        Returns:
            Any: Description.
        """
        x = torch.tanh(x)
        x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon)).unsqueeze(2)
        x = (x * self.arange).flatten(1, 2)
        x = x.cos()
        x = self.poly_conv[group_index](x)
        x = self.layer_norm[group_index](x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KACNConvNDLayer.

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
            y = self.forward_kacn(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KACNConv3DLayer(KACNConvNDLayer):
    """KACNConv3DLayer class."""

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
        dropout=0.0,
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
            dropout (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            nn.Conv3d,
            norm_layer,
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


class KACNConv2DLayer(KACNConvNDLayer):
    """KACNConv2DLayer class."""

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
        dropout=0.0,
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
            dropout (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            nn.Conv2d,
            norm_layer,
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


class KACNConv1DLayer(KACNConvNDLayer):
    """KACNConv1DLayer class."""

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
        dropout=0.0,
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
            dropout (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            nn.Conv1d,
            norm_layer,
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
