import torch
import torch.nn as nn


class SpectralEnergyAlignment(nn.Module):
    """
    Spectral Energy Alignment (SEA) layer.

    Physically forces the generated k-space to reside on the same numerical
    manifold as the target data. This isolates the generator's task to predicting
    phase and relative structural magnitude, rather than guessing absolute global scale.
    """

    def __init__(self, epsilon=1e-8):
        """__init__.

        Args:
            epsilon (Any): Description.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, k_pred, k_masked_input):
        """
        k_pred: Complex generator output [B, C, H, W]
        k_masked_input: The conditioned masked k-space [B, C, H, W]

        forward method for SpectralEnergyAlignment.

        Executes PyTorch tensor operations.

        Args:
            k_pred (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            k_masked_input (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Compute Parseval-compliant energy (Frobenius norm over spatial dims)
        # We calculate the norm over the last two dimensions (H, W).
        pred_energy = torch.linalg.norm(k_pred, dim=(-2, -1), keepdim=True)
        target_energy = torch.linalg.norm(k_masked_input, dim=(-2, -1), keepdim=True)

        # 2. Compute scalar. CRITICAL: Detach the scale factor so the
        # optimizer doesn't try to minimize loss by shrinking the scale tensor.
        scale = (target_energy / (pred_energy + self.epsilon)).detach()

        # 3. Align the predicted manifold to the measurement manifold
        return k_pred * scale
