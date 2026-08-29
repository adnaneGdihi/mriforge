import torch
import torch.nn as nn

from mriforge.infrastructure.physics.fft_ops import ifft2c


class MRIEnvironmentWrapper:
    """
    Simulates the MRI Scanner environment for the RL Agent.
    State: Current Zero-Filled Reconstruction + Current Mask
    Action: Toggle a k-space line (0 -> 1)
    Reward: Improvement in PSNR/SSIM relative to ground truth (or uncertainty reduction).
    """

    def __init__(self, ground_truth_kspace):
        """__init__.

        Args:
            ground_truth_kspace (Any): Description.
        """
        self.gt_kspace = ground_truth_kspace  # (B, C, H, W)
        self.mask = torch.zeros_like(ground_truth_kspace[:, 0:1, :, :])  # Binary mask (B, 1, H, W)
        self.current_kspace = torch.zeros_like(ground_truth_kspace)

    def step(self, action_mask):
        """
        Apply new sampling lines.
        action_mask: Binary mask of *new* lines to acquire.
        """
        # 1. Update Mask (Logical OR to keep history)
        new_mask = torch.max(self.mask, action_mask)

        # 2. Acquire Data (Physics Simulation)
        # In a real scanner, this triggers hardware. Here, we reveal GT data.
        self.current_kspace = self.gt_kspace * new_mask
        self.mask = new_mask

        # 3. Reconstruct (Simple IFFT for state observation)
        # Handle complex conversion if input is 2-channel real
        if self.current_kspace.shape[1] == 2:
            kspace_complex = torch.complex(self.current_kspace[:, 0], self.current_kspace[:, 1])
            recon = ifft2c(kspace_complex).abs()  # Use physics module for proper centering
            # Add channel dim back if needed for consistency, usually image is (B, 1, H, W)
            recon = recon.unsqueeze(1)
        else:
            recon = ifft2c(self.current_kspace).abs()  # Use physics module for proper centering
            if recon.ndim == 3:
                recon = recon.unsqueeze(1)

        return recon, self.mask


class DeepQScannerAgent(nn.Module):
    """
    RL Agent (DQN/Policy Gradient) that decides WHERE to scan next.
    Input: Current Reconstruction (Aliased) + Current Mask
    Output: Probability map for next k-space lines.
    """

    def __init__(self, img_size=256, in_channels=2, action_space=None):
        """__init__.

        Args:
            img_size (Any): Description.
            in_channels (Any): Description.
            action_space (Any): Description.
        """
        super().__init__()
        # Lightweight CNN to estimate uncertainty/saliency
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, 32, 3, padding=1),  # +1 for Mask channel
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global context
        )

        # Policy Head: Maps global context back to a 1D line-selection vector
        # (Assuming Cartesian Phase Encoding lines along H)
        output_dim = action_space if action_space is not None else img_size

        self.policy_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # Logits for each phase encoding line
            nn.Sigmoid(),
        )

    def forward(self, current_recon, current_mask) -> torch.Tensor:
        """
        Decides which lines to scan next based on current image quality.

        forward method for DeepQScannerAgent.

        Executes PyTorch tensor operations.

        Args:
            current_recon (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            current_mask (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Input: [Batch, 2, H, W] (Recon) + [Batch, 1, H, W] (Mask)
        # Ensure recon has correct channels. If recon is (B, 1, H, W) and in_channels=2, we might simulate Re/Im or just dup.
        # But usually recon is magnitude (1 ch).
        # Adjust encoder input if needed.
        # For now, blindly cat.
        if current_recon.shape[1] != 2 and current_recon.shape[1] == 1:
            # Duplicate to match expected in_channels=2 (Real/Imag placeholder) # IMPL
            current_recon = current_recon.repeat(1, 2, 1, 1)

        x = torch.cat([current_recon, current_mask], dim=1)

        features = self.encoder(x).view(x.size(0), -1)
        line_probs = self.policy_head(features)  # [Batch, H]

        return line_probs

    def select_action(self, line_probs, budget=10):
        """
        Selects top-k most uncertain lines to acquire.
        """
        # Greedy selection for demo; in RL use categorical sampling
        _, top_indices = torch.topk(line_probs, k=budget, dim=1)

        # Convert indices to full 2D mask
        B, H = line_probs.shape
        W = H  # Assuming square
        action_mask = torch.zeros(B, 1, H, W, device=line_probs.device)

        for b in range(B):
            action_mask[b, :, top_indices[b], :] = 1.0

        return action_mask
