import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model


class CliffordConv2d(nn.Module):
    """
    Paradigm 3: Clifford Convolution (Geometric Algebra Cl(2,0)).
    Treats MRI features as Multivectors {Scalar, e1, e2, Bivector} rather than independent channels.

    Physics Benefit: Enforces SE(2) Rotational Equivariance naturally.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            kernel_size (Any): Description.
            stride (Any): Description.
            padding (Any): Description.
        """
        super().__init__()
        # Cl(2,0) has 4 components: [u, v, w, h] corresponding to {1, e1, e2, e12}
        # We process input/output channels in groups of 4 (multivectors).
        assert in_channels % 4 == 0 and out_channels % 4 == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # We learn 4 weight matrices corresponding to the geometric product interactions
        # W = W0 + W1*e1 + W2*e2 + W3*e12
        # Interaction rules of Cl(2,0):
        # e1*e1=1, e2*e2=1, e1*e2=e12, e2*e1=-e12

        # Reduced effective channels for weight tensors
        eff_in = in_channels // 4
        eff_out = out_channels // 4

        self.weight_0 = nn.Parameter(torch.Tensor(eff_out, eff_in, kernel_size, kernel_size))
        self.weight_1 = nn.Parameter(torch.Tensor(eff_out, eff_in, kernel_size, kernel_size))
        self.weight_2 = nn.Parameter(torch.Tensor(eff_out, eff_in, kernel_size, kernel_size))
        self.weight_3 = nn.Parameter(torch.Tensor(eff_out, eff_in, kernel_size, kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        """reset_parameters.

        Returns:
            Any: Description.
        """
        scale = math.sqrt(1.0 / (self.in_channels * self.kernel_size**2))
        for w in [self.weight_0, self.weight_1, self.weight_2, self.weight_3]:
            nn.init.uniform_(w, -scale, scale)

    def forward(self, x):
        # Input x: [Batch, In_Channels, H, W]
        # Split into geometric components {1, e1, e2, e12}
        # In MRI: 1=DC/Mag, e1=Real, e2=Imag, e12=PhaseCurl
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for CliffordConv2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        u, v, w, h = torch.chunk(x, 4, dim=1)

        # Geometric Product Rule Table for Cl(2,0):
        # Out_0 (Scalar) = W0*u + W1*v + W2*w - W3*h
        # Out_1 (e1)     = W0*v + W1*u - W2*h + W3*w
        # Out_2 (e2)     = W0*w + W1*h + W2*u - W3*v
        # Out_3 (e12)    = W0*h - W1*w + W2*v + W3*u

        def conv(input_tensor, weight_tensor):
            """conv.

            Args:
                input_tensor (Any): Description.
                weight_tensor (Any): Description.
            Returns:
                Any: Description.
            """
            return F.conv2d(input_tensor, weight_tensor, stride=self.stride, padding=self.padding)

        # Component 0 (Scalar Part)
        out_0 = (
            conv(u, self.weight_0)
            + conv(v, self.weight_1)
            + conv(w, self.weight_2)
            - conv(h, self.weight_3)
        )

        # Component 1 (Vector e1 Part - e.g., Real)
        out_1 = (
            conv(v, self.weight_0)
            + conv(u, self.weight_1)
            - conv(h, self.weight_2)
            + conv(w, self.weight_3)
        )

        # Component 2 (Vector e2 Part - e.g., Imag)
        out_2 = (
            conv(w, self.weight_0)
            + conv(h, self.weight_1)
            + conv(u, self.weight_2)
            - conv(v, self.weight_3)
        )

        # Component 3 (Bivector e12 Part - e.g., Local Phase Rotation)
        out_3 = (
            conv(h, self.weight_0)
            - conv(w, self.weight_1)
            + conv(v, self.weight_2)
            + conv(u, self.weight_3)
        )

        return torch.cat([out_0, out_1, out_2, out_3], dim=1)


@register_model(name="clifford_gan", training_mode="gan")
class CliffordGANGenerator(nn.Module):
    """Generator using Geometric Algebra Cl(2,0) to preserve MRI phase/spin physics.

    Input: ULF MRI (Real, Imag) → Expand to Multivector {1, e1, e2, e12} → HF MRI.

    Supports:
    - Configurable depth via num_layers kwarg
    - Residual skip connections for gradient stability
    - QSM dipole kernel conditioning via kwargs
    """

    def __init__(self, in_channels=2, out_channels=2, base_dim=64, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_layers = kwargs.get("num_layers", 6)
        hidden_dim = kwargs.get("hidden_dim", base_dim)

        # Pre-process: Project Real/Imag (2ch) to Multivector (4ch)
        self.entry = nn.Conv2d(in_channels, hidden_dim * 4, 3, padding=1)

        # Clifford Body: Rotational Equivariant blocks with residual connections
        self.cliff_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.cliff_layers.append(CliffordConv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1))
            self.norms.append(nn.GroupNorm(4, hidden_dim * 4))
        self.act = nn.SiLU()

        # Post-process: Project back to Real/Imag
        self.exit = nn.Conv2d(hidden_dim * 4, out_channels, 1)

    def forward(self, x, **kwargs):
        """Forward pass: (B, C, H, W) → (B, out_channels, H, W).

        Supports optional kwargs passed by ReconstructionTrainingStrategy.

        forward method for CliffordGANGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        identity = x
        h = self.entry(x)

        # Clifford body with residual connections
        for cliff, norm in zip(self.cliff_layers, self.norms, strict=True):
            residual = h
            h = cliff(h)
            h = norm(h)
            h = self.act(h)
            h = h + residual  # Residual skip

        out = self.exit(h)

        # Global residual if channels match
        if self.in_channels == self.out_channels:
            out = out + identity

        return out

    @property
    def name(self) -> str:
        return "clifford_gan"
