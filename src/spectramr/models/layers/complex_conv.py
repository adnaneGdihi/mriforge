"""
Complex Convolution Layers.

Provides complex-valued 2D convolution operations optimized for performance using
fused kernels and block-matrix operations. Essential for processing MRI k-space data.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D Convolution optimized for arithmetic intensity.

    Performs convolution on complex inputs (real, imag) using a single fused kernel
    launch via block-matrix convolution:

    [Y_r]   [W_r  -W_i]   [X_r]
    [   ] = [         ] * [   ]
    [Y_i]   [W_i   W_r]   [X_i]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
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
            padding_mode (str): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = (
            padding if padding_mode == "zeros" else 0
        )  # Use manual padding for non-zero modes
        self.manual_padding = padding if padding_mode != "zeros" else None
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        # Define weights manually for block construction
        self.weight_real = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.weight_imag = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize as complex weights
        # Standard Kaiming init for complex numbers: scale by 1/sqrt(2)
        """reset_parameters.

        Returns:
            Any: Description.
        """
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        with torch.no_grad():
            self.weight_real.div_(math.sqrt(2))
            self.weight_imag.div_(math.sqrt(2))

        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass using fused kernel.

        forward method for ComplexConv2d.

        Executes PyTorch tensor operations.

        Args:
            input_real (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            input_imag (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Prepare Input: [B, 2*Cin, H, W]
        # 1. Prepare Input: [B, 2*Cin, H, W]
        # [FIX] Handle Complex Input automatically
        if torch.is_complex(input_real):
            # Input is [B, C, H, W] complex
            # We want [B, 2*C, H, W] real (Concatenated Real|Imag)
            x_real = input_real.real
            x_imag = input_real.imag
            cat_input = torch.cat([x_real, x_imag], dim=1)
            # Re-assign input_imag to None to trigger correct return path
            input_imag = None

        elif input_imag is None:
            # Assume input is [B, 2*C, H, W] interleaved: [R1, I1, R2, I2, ...]
            # Must convert to stacked for fused kernel compatibility
            channels = input_real.shape[1] // 2
            x_real = input_real[:, 0::2, ...]
            x_imag = input_real[:, 1::2, ...]
            cat_input = torch.cat([x_real, x_imag], dim=1)
        else:
            x_real = input_real
            x_imag = input_imag
            cat_input = torch.cat([x_real, x_imag], dim=1)

        # Apply manual padding for non-zero mode (circular, reflect, etc.)
        if self.manual_padding is not None and self.manual_padding > 0:
            pad_amount = self.manual_padding
            if isinstance(pad_amount, int):
                pad_amount = (pad_amount, pad_amount, pad_amount, pad_amount)
            else:
                # Convert tuple (pad_h, pad_w) to (pad_left, pad_right, pad_top, pad_bottom)
                pad_amount = (
                    pad_amount[1],
                    pad_amount[1],
                    pad_amount[0],
                    pad_amount[0],
                )
            cat_input = F.pad(cat_input, pad_amount, mode=self.padding_mode)

        # 2. Construct Block Weight Matrix: [2*Cout, 2*Cin, K, K]
        # [W_r, -W_i]
        # [W_i,  W_r]
        w_row1 = torch.cat([self.weight_real, -self.weight_imag], dim=1)
        w_row2 = torch.cat([self.weight_imag, self.weight_real], dim=1)
        cat_weights = torch.cat([w_row1, w_row2], dim=0)

        # 3. Construct Bias: [2*Cout]
        if self.bias_real is not None:
            cat_bias = torch.cat([self.bias_real, self.bias_imag], dim=0)
        else:
            cat_bias = None

        # 4. Single Fused Convolution (use padding=0 if manual padding applied)
        out = F.conv2d(
            cat_input,
            cat_weights,
            cat_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # 5. Return
        if input_imag is None:
            # Return interleaved for combined output path
            out_real = out[:, : self.out_channels, ...]
            out_imag = out[:, self.out_channels :, ...]
            # Use flatten(1, 2) for TorchScript compatibility instead of unpacking shape
            return torch.stack([out_real, out_imag], dim=2).flatten(1, 2)
        else:
            channels = out.shape[1] // 2
            return out[:, :channels, ...], out[:, channels:, ...]


class ComplexConvTranspose2d(nn.Module):
    """
    Complex-valued 2D Transposed Convolution.

    Performs transposed convolution (upsampling) on complex inputs.
    Uses block-matrix construction similar to ComplexConv2d but adapted for
    transposed convolution weight layout [In, Out, K, K].

    y_r = x_r *^T w_r - x_i *^T w_i
    y_i = x_r *^T w_i + x_i *^T w_r
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (Union[int, tuple[int, int]]): Description.
            stride (Union[int, tuple[int, int]]): Description.
            padding (Union[int, tuple[int, int]]): Description.
            output_padding (Union[int, tuple[int, int]]): Description.
            groups (int): Description.
            bias (bool): Description.
            dilation (Union[int, tuple[int, int]]): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        # Weights for TransposeConv are [In, Out/groups, K, K]
        # We declare them this way to match PyTorch convention
        self.weight_real = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, *self.kernel_size)
        )
        self.weight_imag = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, *self.kernel_size)
        )

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Scale for complex initialization
        """reset_parameters.

        Returns:
            Any: Description.
        """
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        with torch.no_grad():
            self.weight_real.div_(math.sqrt(2))
            self.weight_imag.div_(math.sqrt(2))

        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # 1. Prepare Input: [B, 2*Cin, H, W]
        """forward.

        Args:
            input_real (torch.Tensor): Description.
            input_imag (torch.Tensor): Description.
        Returns:
            Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Description.

        forward method for ComplexConvTranspose2d.

        Executes PyTorch tensor operations.

        Args:
            input_real (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            input_imag (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if torch.is_complex(input_real):
            x_real = input_real.real
            x_imag = input_real.imag
            cat_input = torch.cat([x_real, x_imag], dim=1)
            input_imag = None  # Signal to return combined
        elif input_imag is None:
            # Assume input is [B, 2*C, H, W] interleaved: [R1, I1, R2, I2, ...]
            # Must convert to stacked for fused kernel compatibility
            channels = input_real.shape[1] // 2
            x_real = input_real[:, 0::2, ...]
            x_imag = input_real[:, 1::2, ...]
            cat_input = torch.cat([x_real, x_imag], dim=1)
        else:
            x_real = input_real
            x_imag = input_imag
            cat_input = torch.cat([x_real, x_imag], dim=1)

        # 2. Construct Block Weight Matrix for Transpose Conv: [2*Cin, 2*Cout, K, K]
        # Input Real (Row 1): Maps to Out Real (Wr) and Out Imag (Wi) -> [Wr, Wi]
        # Input Imag (Row 2): Maps to Out Real (-Wi) and Out Imag (Wr) -> [-Wi, Wr]

        # Note: dim=1 is Output Channel dimension in TransposeConv weights [In, Out, K, K]

        w_input_real = torch.cat([self.weight_real, self.weight_imag], dim=1)  # [Cin, 2*Cout]
        w_input_imag = torch.cat([-self.weight_imag, self.weight_real], dim=1)  # [Cin, 2*Cout]

        # Concatenate along dim=0 (Input Channel dimension)
        cat_weights = torch.cat([w_input_real, w_input_imag], dim=0)  # [2*Cin, 2*Cout]

        # 3. Construct Bias
        if self.bias_real is not None:
            cat_bias = torch.cat([self.bias_real, self.bias_imag], dim=0)
        else:
            cat_bias = None

        # 4. Convolution
        out = F.conv_transpose2d(
            cat_input,
            cat_weights,
            cat_bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

        # 5. Return
        if input_imag is None:
            # Return interleaved for combined output path
            out_real = out[:, : self.out_channels, ...]
            out_imag = out[:, self.out_channels :, ...]
            # Use flatten(1, 2) for TorchScript compatibility instead of unpacking shape
            return torch.stack([out_real, out_imag], dim=2).flatten(1, 2)
        else:
            channels = out.shape[1] // 2
            return out[:, :channels, ...], out[:, channels:, ...]


class ComplexConv2dReflectPad(nn.Module):
    """
    Complex-valued 2D convolution with reflect padding to eliminate boundary ringing.

    Wraps ComplexConv2d while applying torch.nn.functional.pad with reflect mode
    before convolution. Eliminates Gibbs ringing artifacts that occur with zero-
    padding, which creates box artifacts in the Fourier domain.

    Reflect padding mirrors boundary values:
        Zero-pad (BAD):  [x, 0, 0, y] -> Fourier -> box artifact
        Reflect-pad:     [x, y, x, y] -> Fourier -> natural boundary

    Args:
        Same as ComplexConv2d, see that class for details.
        Note: padding parameter is interpreted as the amount of reflect padding
              to apply before convolution, NOT as Conv2d's padding parameter.

    Shape:
        - Input: (B, 2*C_in, H, W) complex stacked as real/imaginary
        - Output: (B, 2*C_out, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
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
        """
        super().__init__()
        self.reflect_padding = padding
        self.conv = ComplexConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # No padding in underlying conv; we apply padding explicitly
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass with reflect padding.

        Args:
            input_real (torch.Tensor): Real component [B, C_in, H, W] or stacked [B, 2*C_in, H, W].
            input_imag (torch.Tensor): Imaginary component [B, C_in, H, W], or None if stacked.

        Returns:
            Same as ComplexConv2d.forward().

        forward method for ComplexConv2dReflectPad.

        Executes PyTorch tensor operations.

        Args:
            input_real (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            input_imag (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Convert padding to tuple if int
        if isinstance(self.reflect_padding, int):
            pad_tuple = (
                self.reflect_padding,
                self.reflect_padding,
                self.reflect_padding,
                self.reflect_padding,
            )
        else:
            # Assume (pad_h, pad_w) tuple
            pad_h, pad_w = self.reflect_padding
            pad_tuple = (pad_w, pad_w, pad_h, pad_h)  # (left, right, top, bottom)

        # Apply reflect padding
        padded_real = F.pad(input_real, pad_tuple, mode="reflect")
        if input_imag is not None:
            padded_imag = F.pad(input_imag, pad_tuple, mode="reflect")
        else:
            padded_imag = None

        # Convolution without additional padding
        return self.conv(padded_real, padded_imag)


class ComplexRepetitionFusion(nn.Module):
    """
    Learns to dynamically fuse multiple repetitions (NEX) in complex k-space
    without naively averaging them (which destroys independent phase distributions and noise realizations).
    """

    def __init__(self, num_physical_coils=4, num_repetitions=4):
        """__init__.

        Args:
            num_physical_coils (Any): Description.
            num_repetitions (Any): Description.
        """
        super().__init__()
        # 1x1 Complex Convolution across the Rep * Coil dimension
        # Maps [Reps * Coils] -> [Coils] without touching spatial kx, ky
        self.fusion = ComplexConv2d(
            in_channels=num_physical_coils * num_repetitions,
            out_channels=num_physical_coils,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x_reps):
        """
        x_reps: [B_flat, Reps, Coils, Ky, Kx]

        forward method for ComplexRepetitionFusion.

        Executes PyTorch tensor operations.

        Args:
            x_reps (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, R, C, Ky, Kx = x_reps.shape

        # 1. Flatten Repetitions and Coils into a single channel dimension
        # Shape: [B_flat, Reps * Coils, Ky, Kx]
        x_flat = x_reps.view(B, R * C, Ky, Kx)

        # 2. Learnable Temporal/NEX Fusion natively in k-space
        # Shape: [B_flat, Coils, Ky, Kx]
        x_optimal = self.fusion(x_flat)

        return x_optimal
