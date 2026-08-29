"""Self-Supervised Adversarial Diffusion (SSAD-MRI) training logic.

Based on cluster.md Section 3.3 (Deep Dive):
"SSAD-MRI trains a diffusion model only on undersampled data.
Partition undersampled acquisition Omega into Theta (Input) and Lambda (Loss)."
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from mriforge.infrastructure.physics.fft_ops import fft2c


class SSADSplitter(nn.Module):
    """Splits available k-space mask into Input and Loss masks."""

    def __init__(self, split_ratio: float = 0.4):
        """
        Args:
            split_ratio: Fraction of available points to use for LOSS (Lambda).
                         The rest (1 - split_ratio) are used for INPUT (Theta).
        """
        super().__init__()
        self.split_ratio = split_ratio

    def forward(
        self, mask: Int[Tensor, "B 1 H W"]
    ) -> tuple[Int[Tensor, "B 1 H W"], Int[Tensor, "B 1 H W"]]:
        """
        Returns:
            mask_input (Theta): Subset used for reconstruction input.
            mask_loss (Lambda): Subset used for self-supervision loss.

        forward method for SSADSplitter.

        Executes PyTorch tensor operations.

        Args:
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[Int[Tensor, 'B 1 H W'], Int[Tensor, 'B 1 H W']]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Identify indices where mask is 1
        B, C, H, W = mask.shape
        flat_mask = mask.view(B, -1)  # [B, H*W]

        mask_input = torch.zeros_like(flat_mask)
        mask_loss = torch.zeros_like(flat_mask)

        # Determine number of points to split per batch item
        # We can optimize this with vectorization if ratio is fixed,
        # but exact count might vary if initial masks vary.

        # Simple random splitting logic
        # Ideally we want to keep the "Auto-Calibration Signal (ACS)" in both?
        # Or split everything pure randomly. The paper implies random splitting.

        rand_tensor = torch.rand_like(flat_mask.float())

        # Only split where mask is 1
        # If mask[i] == 1:
        #   if rand < ratio: assigned to LOSS
        #   else: assigned to INPUT
        # If mask[i] == 0: assigned to neither (ignored)

        is_active = flat_mask > 0.5
        is_loss = (rand_tensor < self.split_ratio) & is_active
        is_input = (~is_loss) & is_active

        mask_loss[is_loss] = 1.0
        mask_input[is_input] = 1.0

        return mask_input.view(B, C, H, W), mask_loss.view(B, C, H, W)


class SSADInterface(nn.Module):
    """Wraps a diffusion/reconstruction model for SSAD training."""

    def __init__(self, reconstruction_model: nn.Module, split_ratio: float = 0.4):
        """__init__.

        Args:
            reconstruction_model (nn.Module): Description.
            split_ratio (float): Description.
        """
        super().__init__()
        self.model = reconstruction_model
        self.splitter = SSADSplitter(split_ratio)

    def forward(
        self,
        kspace: Float[Tensor, "B 2 H W"],  # 2 chans for complex (real, imag)
        mask_omega: Int[Tensor, "B 1 H W"],
    ) -> dict[str, Tensor]:
        """
        Args:
            kspace: The acquired undersampled k-space (zeros where not sampled).
            mask_omega: The acquisition mask (binary).

        forward method for SSADInterface.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask_omega (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Split Mask
        mask_theta, mask_lambda = self.splitter(mask_omega)

        # 2. Prepare Input
        # y_theta = y_omega * mask_theta
        input_kspace = kspace * mask_theta

        # 3. Reconstruct
        # Model expects input kspace and mask
        # It should output reconstructed image or kspace.
        # Assuming model outputs IMAGE `x_recon`.
        x_recon = self.model(input_kspace, mask_theta)

        # 4. Forward Project to K-Space
        # FFT2 centered
        k_recon = self._fft2(x_recon)

        # 5. Compute Self-Supervised Loss
        # Loss = || k_recon[Lambda] - k_gt[Lambda] ||
        # Note: k_gt here is the original 'kspace' input, which contains data at Lambda.

        diff = (k_recon - kspace) * mask_lambda
        loss = torch.norm(diff, p=2) ** 2 / (torch.sum(mask_lambda) + 1e-8)

        return {
            "loss": loss,
            "x_recon": x_recon,
            "mask_input": mask_theta,
            "mask_loss": mask_lambda,
        }

    def _fft2(self, x: Tensor) -> Tensor:
        # x is complex (B, 2, H, W) or real?
        # Assuming x is (B, 2, H, W) representing complex image
        # OR (B, 1, H, W) magnitude?
        # Reconstruction usually outputs complex image for MRI.
        # Impl: complex -> FFT -> complex

        """_fft2.

        Args:
            x (Tensor): Description.
        Returns:
            Tensor: Description.
        """
        if x.shape[1] == 2:
            x_complex = torch.complex(x[:, 0], x[:, 1])
        else:
            x_complex = x.squeeze(1).complex()  # Assuming purely real image if 1 chan?

        # ``fft2c`` already centers the output (DC at H/2, W/2). The
        # spurious ``torch.fft.fftshift`` that lived here previously
        # corner-shifted the DC, producing k-space in the OPPOSITE
        # convention to what the SSAD loss compares it against
        # (``kspace`` arrives centered from the data pipeline). The
        # diff was then mixing conventions — CLAUDE.md pitfall #2.
        # See TODO/backlog_ssot_and_layering_cleanup.md Phase 3.
        k = fft2c(x_complex)

        k_real = torch.view_as_real(k).permute(0, 3, 1, 2)  # [B, 2, H, W]
        return k_real
