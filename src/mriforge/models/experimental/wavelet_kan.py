import math

import torch
import torch.nn as nn


class WaveletKANLinear(nn.Module):
    """
    Paradigm 7 Refinement: Wavelet KAN (Wav-KAN).
    Uses continuous wavelet transforms on edges instead of B-Splines.
    Better at capturing high-frequency details (edges, textures) in MRI.
    """

    def __init__(self, in_features, out_features, wavelet_type="mexican_hat"):
        """__init__.

        Args:
            in_features (Any): Description.
            out_features (Any): Description.
            wavelet_type (Any): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Learnable parameters for translation (b) and dilation (s)
        # For each input-output pair, we learn a wavelet function
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.dilation = nn.Parameter(torch.ones(out_features, in_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        """reset_parameters.

        Returns:
            Any: Description.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.translation, -1, 1)
        nn.init.uniform_(self.dilation, 0.5, 1.5)

    def mexican_hat(self, x):
        # (1 - x^2) * exp(-0.5 * x^2)
        """mexican_hat.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        return (1 - x**2) * torch.exp(-0.5 * x**2)

    def morlet(self, x):
        # cos(5x) * exp(-0.5 * x^2)
        """morlet.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        return torch.cos(5 * x) * torch.exp(-0.5 * x**2)

    def forward(self, x):
        # x: [Batch, In]
        # We need to compute wavelet( (x - b) / s ) * w
        # This requires expanding x to [Batch, Out, In] to handle pairwise params

        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for WaveletKANLinear.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, I = x.shape
        O = self.out_features

        # [Batch, 1, In]
        x_expanded = x.unsqueeze(1)

        # Compute normalized input: (x - t) / d
        # t, d are [Out, In]
        x_norm = (x_expanded - self.translation) / self.dilation

        # Apply Wavelet
        if self.wavelet_type == "mexican_hat":
            basis = self.mexican_hat(x_norm)
        elif self.wavelet_type == "morlet":
            basis = self.morlet(x_norm)
        else:
            raise ValueError(f"Unknown wavelet: {self.wavelet_type}")

        # Weighted sum over inputs: sum(basis * weight, dim=2)
        # basis: [Batch, Out, In], weight: [Out, In]
        out = (basis * self.weight).sum(dim=2)

        return out
