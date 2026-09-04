"""Complex KAN Convolution Layer.
==============================

Implementation of Complex-Valued Convolution using FastKAN (Kolmogorov-Arnold Network) layers.
Mimics complex arithmetic: (x + iy) * (u + iv) = (xu - yv) + i(xv + yu)
where u and v are learnable non-linear functions (KANs) acting on the input.

This effectively replaces the linear weights in ComplexConv2d with KAN functions.
"""

import torch
import torch.nn as nn

from spectramr.models.layers.kan.fastkanconv import FastKANConvLayer


class ComplexFastKANConvLayer(nn.Module):
    """
    Complex-Valued FastKAN Convolution.

    Uses two real-valued FastKANConvLayers to approximate complex operations.
    Let input Z = X + iY.
    Let learnable KAN functions be U(.) and V(.).

    If we treat U and V as the "complex weights" equivalent in KAN:
    Real(Out) = U(X) - V(Y)
    Imag(Out) = V(X) + U(Y)

    However, KANs are non-linear. The above is a direct analog of multiplication.
    Another variation is to have 4 distinct KANs for full flexibility (like unconstrained complex weights):
    Real(Out) = KAN_rr(X) - KAN_ri(Y)
    Imag(Out) = KAN_ir(X) + KAN_ii(Y)

    We will implement the 4-KAN version for maximum expressivity ("split-complex KAN"),
    as standard 'weights' U, V being shared across X and Y might be too restrictive given KANs are input-dependent.

    Actually, to keep parameter count reasonable and closer to "Complex Conv" logic where weights are shared:
    We'll use the "tied" version (2 KANs) if `independent_components=False`,
    and 4 KANs if `independent_components=True`.
    """

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
        kan_type: str = "RBF",
        dropout: float = 0.0,
        independent_components: bool = True,
    ):
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
            kan_type (str): Description.
            dropout (float): Description.
            independent_components (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.independent_components = independent_components

        # Helper to build a KAN layer
        def build_kan():
            """build_kan.

            Returns:
                Any: Description.
            """
            return FastKANConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                kan_type=kan_type,
                dropout=dropout,
            )

        if independent_components:
            # 4 Independent KANs
            self.kan_rr = build_kan()  # Real input -> Real output contrib
            self.kan_ri = build_kan()  # Imag input -> Real output contrib
            self.kan_ir = build_kan()  # Real input -> Imag output contrib
            self.kan_ii = build_kan()  # Imag input -> Imag output contrib
        else:
            # 2 Tied KANs (Mimic shared complex weights)
            self.kan_real = build_kan()
            self.kan_imag = build_kan()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex input [B, 2*Cin, H, W] (Stacked Real/Imag)
               or [B, Cin, H, W] (Complex64/128)

        Returns:
            [B, 2*Cout, H, W] (Stacked Real/Imag)

        forward method for ComplexFastKANConvLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Parse Input
        if torch.is_complex(x):
            x_real = x.real
            x_imag = x.imag
        else:
            # Assume [B, 2*C, H, W]
            channels = x.shape[1] // 2
            x_real = x[:, :channels, ...]
            x_imag = x[:, channels:, ...]

        if self.independent_components:
            # Real(Out) = KAN_rr(Xr) - KAN_ri(Xi)
            # Imag(Out) = KAN_ir(Xr) + KAN_ii(Xi)
            # Note: The sign of the mixing terms is convention.
            # Standard complex mult: (xr + i xi)(wr + i wi) = (xr wr - xi wi) + i(xr wi + xi wr)
            # Mapping: wr ~ rr/ii, wi ~ ir/ri.
            # We allow them to be learned freely.

            out_real = self.kan_rr(x_real) - self.kan_ri(x_imag)
            out_imag = self.kan_ir(x_real) + self.kan_ii(x_imag)

        else:
            # Tied version (strict complex multiplication analogy)
            # Weights: W = Wr + i Wi
            # Input: X = Xr + i Xi
            # Out = (Xr Wr - Xi Wi) + i (Xr Wi + Xi Wr)
            # Here KANs replace W multiplication.

            # We need to reuse the SAME KAN layer on different inputs?
            # FastKANConvLayer has internal state (RBF grids don't change, but spline weights are fixed).
            # Yes, standard nn.Module reuse is fine.

            xr_wr = self.kan_real(x_real)
            xi_wi = self.kan_imag(x_imag)
            xr_wi = self.kan_imag(x_real)
            xi_wr = self.kan_real(x_imag)

            out_real = xr_wr - xi_wi
            out_imag = xr_wi + xi_wr

        # Return stacked [B, 2*Cout, H, W]
        return torch.cat([out_real, out_imag], dim=1)
