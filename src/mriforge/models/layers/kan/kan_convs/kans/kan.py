# taken from and based on https://github.com/1ssb/torchkan/blob/main/torchkan.py
# and https://github.com/1ssb/torchkan/blob/main/KALnet.py
# and https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# and https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# and https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
# and https://github.com/zavareh1/Wav-KAN
# and https://github.com/quiqi/relu_kan/issues/2

from torch import nn

from .layers import (
    BernsteinKANLayer,
    BottleNeckGRAMLayer,
    ChebyKANLayer,
    FastKANLayer,
    GRAMLayer,
    JacobiKANLayer,
    KALNLayer,
    KANLayer,
    ReLUKANLayer,
    WavKANLayer,
)


class L1(nn.Module):
    """L1 regularization wrapper for layers."""

    def __init__(self, layer: nn.Module, l1_decay: float):
        """__init__.

        Args:
            layer (nn.Module): Description.
            l1_decay (float): Description.
        """
        super().__init__()
        self.layer = layer
        self.l1_decay = l1_decay

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for L1.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.layer(x)


class KAN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    """KAN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        grid_size=5,
        spline_order=3,
        base_activation=nn.GELU,
        grid_range: list = None,
        l1_decay: float = 0.0,
        first_dropout: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            grid_size (Any): Description.
            spline_order (Any): Description.
            base_activation (Any): Description.
            grid_range (list): Description.
            l1_decay (float): Description.
            first_dropout (bool): Description.
        """
        if grid_range is None:
            grid_range = [-1, 1]
        super().__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are
        # calculated
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation
        self.grid_range = grid_range

        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            layer = KANLayer(
                in_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                base_activation=base_activation,
                grid_range=grid_range,
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class KALN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    """KALN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        degree=3,
        base_activation=nn.SiLU,
        first_dropout: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            degree (Any): Description.
            base_activation (Any): Description.
            first_dropout (bool): Description.
        """
        super().__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are
        # calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = KALNLayer(
                in_features,
                out_features,
                degree,
                base_activation=base_activation,
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KALN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class FastKAN(nn.Module):
    """FastKAN class."""

    def __init__(
        self,
        layers_hidden: list[int],
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        grid_range: list[float] = None,
        grid_size: int = 8,
        use_base_update: bool = True,
        base_activation=nn.SiLU,
        spline_weight_init_scale: float = 0.1,
        first_dropout: bool = True,
        **kwargs,
    ) -> None:
        """__init__.

        Args:
            layers_hidden (list[int]): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            grid_range (list[float]): Description.
            grid_size (int): Description.
            use_base_update (bool): Description.
            base_activation (Any): Description.
            spline_weight_init_scale (float): Description.
            first_dropout (bool): Description.
        """
        if grid_range is None:
            grid_range = [-2, 2]
        super().__init__()
        self.layers_hidden = layers_hidden
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.num_layers = len(layers_hidden[:-1])

        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = FastKANLayer(
                in_features,
                out_features,
                grid_min=self.grid_min,
                grid_max=self.grid_max,
                num_grids=grid_size,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for FastKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class KACN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    """KACN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        degree=3,
        first_dropout: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            degree (Any): Description.
            first_dropout (bool): Description.
        """
        super().__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are
        # calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = ChebyKANLayer(in_features, out_features, degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KACN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class KAGN(nn.Module):
    """KAGN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        degree=3,
        base_activation=nn.SiLU,
        first_dropout: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            degree (Any): Description.
            base_activation (Any): Description.
            first_dropout (bool): Description.
        """
        super().__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are
        # calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = GRAMLayer(in_features, out_features, degree, act=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KAGN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class BottleNeckKAGN(nn.Module):
    """BottleNeckKAGN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        degree=3,
        base_activation=nn.SiLU,
        first_dropout: bool = True,
        dim_reduction: float = 8,
        min_internal: int = 16,
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            degree (Any): Description.
            base_activation (Any): Description.
            first_dropout (bool): Description.
            dim_reduction (float): Description.
            min_internal (int): Description.
        """
        super().__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are
        # calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = BottleNeckGRAMLayer(
                in_features,
                out_features,
                degree,
                act=base_activation,
                dim_reduction=dim_reduction,
                min_internal=min_internal,
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for BottleNeckKAGN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class KABN(nn.Module):
    """KABN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        degree=3,
        base_activation=nn.SiLU,
        first_dropout: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            degree (Any): Description.
            base_activation (Any): Description.
            first_dropout (bool): Description.
        """
        super().__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are
        # calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = BernsteinKANLayer(
                in_features,
                out_features,
                degree,
                act=base_activation,
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KABN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class KAJN(nn.Module):
    """KAJN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        degree=3,
        a: float = 1,
        b: float = 1,
        base_activation=nn.SiLU,
        first_dropout: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            degree (Any): Description.
            a (float): Description.
            b (float): Description.
            base_activation (Any): Description.
            first_dropout (bool): Description.
        """
        super().__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are
        # calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = JacobiKANLayer(
                in_features,
                out_features,
                degree,
                a=a,
                b=b,
                act=base_activation,
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KAJN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class WavKAN(nn.Module):
    """WavKAN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        first_dropout: bool = True,
        wavelet_type: str = "mexican_hat",
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            first_dropout (bool): Description.
            wavelet_type (str): Description.
        """
        super().__init__()  # Initialize the parent nn.Module class

        assert wavelet_type in [
            "mexican_hat",
            "morlet",
            "dog",
            "meyer",
            "shannon",
        ], ValueError(f"Unsupported wavelet type: {wavelet_type}")
        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are
        # calculated
        self.wavelet_type = wavelet_type
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = WavKANLayer(in_features, out_features, wavelet_type=wavelet_type)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for WavKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


class ReLUKAN(nn.Module):
    """ReLUKAN class."""

    def __init__(
        self,
        layers_hidden,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        g: int = 1,
        k: int = 1,
        train_ab: bool = True,
        first_dropout: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            layers_hidden (Any): Description.
            dropout (float): Description.
            l1_decay (float): Description.
            g (int): Description.
            k (int): Description.
            train_ab (bool): Description.
            first_dropout (bool): Description.
        """
        super().__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in
        # each layer
        self.layers_hidden = layers_hidden
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:], strict=False),
        ):
            # Base weight for linear transformation in each layer
            layer = ReLUKANLayer(in_features, g, k, out_features, train_ab=train_ab)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for ReLUKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x


def mlp_kan(
    layers_hidden,
    dropout: float = 0.0,
    grid_size: int = 5,
    spline_order: int = 3,
    base_activation: nn.Module = nn.GELU,
    grid_range: list = None,
    l1_decay: float = 0.0,
) -> KAN:
    """mlp_kan.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        grid_size (int): Description.
        spline_order (int): Description.
        base_activation (nn.Module): Description.
        grid_range (list): Description.
        l1_decay (float): Description.
    Returns:
        KAN: Description.
    """
    if grid_range is None:
        grid_range = [-1, 1]
    return KAN(
        layers_hidden,
        dropout=dropout,
        grid_size=grid_size,
        spline_order=spline_order,
        base_activation=base_activation,
        grid_range=grid_range,
        l1_decay=l1_decay,
    )


def mlp_fastkan(
    layers_hidden,
    dropout: float = 0.0,
    grid_size: int = 5,
    base_activation: nn.Module = nn.GELU,
    grid_range: list = None,
    l1_decay: float = 0.0,
) -> FastKAN:
    """mlp_fastkan.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        grid_size (int): Description.
        base_activation (nn.Module): Description.
        grid_range (list): Description.
        l1_decay (float): Description.
    Returns:
        FastKAN: Description.
    """
    if grid_range is None:
        grid_range = [-1, 1]
    return FastKAN(
        layers_hidden,
        dropout=dropout,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        l1_decay=l1_decay,
    )


def mlp_kaln(
    layers_hidden,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: nn.Module = nn.GELU,
    l1_decay: float = 0.0,
) -> KALN:
    """mlp_kaln.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        degree (int): Description.
        base_activation (nn.Module): Description.
        l1_decay (float): Description.
    Returns:
        KALN: Description.
    """
    return KALN(
        layers_hidden,
        dropout=dropout,
        base_activation=base_activation,
        degree=degree,
        l1_decay=l1_decay,
    )


def mlp_kabn(
    layers_hidden,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: nn.Module = nn.GELU,
    l1_decay: float = 0.0,
) -> KABN:
    """mlp_kabn.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        degree (int): Description.
        base_activation (nn.Module): Description.
        l1_decay (float): Description.
    Returns:
        KABN: Description.
    """
    return KABN(
        layers_hidden,
        dropout=dropout,
        base_activation=base_activation,
        degree=degree,
        l1_decay=l1_decay,
    )


def mlp_kacn(
    layers_hidden,
    dropout: float = 0.0,
    degree: int = 3,
    l1_decay: float = 0.0,
) -> KACN:
    """mlp_kacn.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        degree (int): Description.
        l1_decay (float): Description.
    Returns:
        KACN: Description.
    """
    return KACN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay)


def mlp_kajn(
    layers_hidden,
    dropout: float = 0.0,
    degree: int = 3,
    l1_decay: float = 0.0,
    a: float = 1.0,
    b: float = 1.0,
) -> KAJN:
    """mlp_kajn.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        degree (int): Description.
        l1_decay (float): Description.
        a (float): Description.
        b (float): Description.
    Returns:
        KAJN: Description.
    """
    return KAJN(
        layers_hidden,
        dropout=dropout,
        degree=degree,
        l1_decay=l1_decay,
        a=a,
        b=b,
    )


def mlp_kagn(
    layers_hidden,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: nn.Module = nn.GELU,
    l1_decay: float = 0.0,
) -> KAGN:
    """mlp_kagn.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        degree (int): Description.
        base_activation (nn.Module): Description.
        l1_decay (float): Description.
    Returns:
        KAGN: Description.
    """
    return KAGN(
        layers_hidden,
        dropout=dropout,
        base_activation=base_activation,
        degree=degree,
        l1_decay=l1_decay,
    )


def mlp_bottleneck_kagn(
    layers_hidden,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: nn.Module = nn.GELU,
    l1_decay: float = 0.0,
    dim_reduction: float = 8,
    min_internal: int = 16,
) -> BottleNeckKAGN:
    """mlp_bottleneck_kagn.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        degree (int): Description.
        base_activation (nn.Module): Description.
        l1_decay (float): Description.
        dim_reduction (float): Description.
        min_internal (int): Description.
    Returns:
        BottleNeckKAGN: Description.
    """
    return BottleNeckKAGN(
        layers_hidden,
        dropout=dropout,
        base_activation=base_activation,
        degree=degree,
        l1_decay=l1_decay,
        dim_reduction=dim_reduction,
        min_internal=min_internal,
    )


def mlp_relukan(
    layers_hidden,
    dropout: float = 0.0,
    a: int = 1,
    b: int = 1,
    train_ab: bool = True,
    l1_decay: float = 0.0,
) -> ReLUKAN:
    """mlp_relukan.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        a (int): Description.
        b (int): Description.
        train_ab (bool): Description.
        l1_decay (float): Description.
    Returns:
        ReLUKAN: Description.
    """
    return ReLUKAN(
        layers_hidden,
        dropout=dropout,
        a=a,
        b=b,
        train_ab=train_ab,
        l1_decay=l1_decay,
    )


def mlp_wav_kan(
    layers_hidden,
    dropout: float = 0.0,
    wavelet_type: str = "mexican_hat",
    l1_decay: float = 0.0,
) -> WavKAN:
    """mlp_wav_kan.

    Args:
        layers_hidden (Any): Description.
        dropout (float): Description.
        wavelet_type (str): Description.
        l1_decay (float): Description.
    Returns:
        WavKAN: Description.
    """
    return WavKAN(
        layers_hidden,
        dropout=dropout,
        wavelet_type=wavelet_type,
        l1_decay=l1_decay,
    )
