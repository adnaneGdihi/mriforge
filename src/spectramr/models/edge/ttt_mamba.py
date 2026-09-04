"""
Test-Time Training (TTT) Mamba.

Implementation of a Mamba layer that adapts its weights during inference
to capture instance-specific features.

Reference: "Test-Time Training with Self-Supervision for Generalization under Shifts" (Sun et al.)
Adapted for SSMs: We update a specific 'adapter' matrix using a self-supervised proxy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model


@register_model(name="ttt_mamba", training_mode="reconstruction")
class TTTMambaBlock(nn.Module):
    """
    Mamba block with Test-Time Training capabilities.

    Structure:
    - Standard Mamba/SSM backbone.
    - TTT Adapter: A small learnable matrix W_ttt.
    - Proxy Loss: Self-reconstruction of masked input tokens.

    During Inference:
    1. Receive input x.
    2. TTT Step:
       - Generate mask.
       - Predict masked tokens using CURRENT weights.
       - Compute gradient of Loss w.r.t W_ttt.
       - Update W_ttt: W_ttt = W_ttt - lr * grad.
    3. Forward Step:
       - Process x using UPDATED W_ttt.
    """

    def __init__(self, dim: int, d_state: int = 16, ttt_lr: float = 0.01, ttt_steps: int = 1):
        """__init__.

        Args:
            dim (int): Description.
            d_state (int): Description.
            ttt_lr (float): Description.
            ttt_steps (int): Description.
        """
        super().__init__()
        self.dim = dim
        self.ttt_lr = ttt_lr
        self.ttt_steps = ttt_steps

        # Standard projections (Simulated Mamba for prototype)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        # TTT Adapter (The weight we update at test time)
        # For simplicity, let's say this adapter transforms the hidden state
        self.adapter = nn.Linear(dim, dim, bias=False)

        # We need a copy of initial weights to reset after inference (if stateful)
        # But for standard TTT, we reset per sample.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, L, D]

        forward method for TTTMambaBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.training:
            # Standard training forward
            return self._forward_impl(x, self.adapter.weight)
        else:
            # Inference with TTT
            # We process ONE sample at a time usually for TTT or Batch adaptation.
            # Let's assume batch adaptation.

            # 1. Clone weights to avoid permanent mutation
            # (In a real optimized kernel we'd use fast weights)
            w_ttt = self.adapter.weight.clone()

            # 2. TTT Loop
            # Enable gradients specifically for w_ttt only
            with torch.enable_grad():
                w_ttt.requires_grad_(True)

                for _ in range(self.ttt_steps):
                    # Create proxy task: Reconstruct masked input
                    # Mask 20% of tokens
                    # x is [B, L, D]
                    mask = (torch.rand(x.shape[:2], device=x.device) < 0.2).float().unsqueeze(-1)
                    x_masked = x * (1 - mask)

                    # Forward with current w_ttt
                    # We need to use functional call to use w_ttt tensor
                    out_proxy = self._forward_functional(x_masked, w_ttt)

                    # Loss: Reconstruction of original x
                    # Simple MSE on masked regions
                    loss = F.mse_loss(out_proxy * mask, x * mask)

                    # Gradient
                    grad = torch.autograd.grad(loss, w_ttt, create_graph=False)[0]

                    # SGD Update
                    w_ttt = w_ttt - self.ttt_lr * grad

            # 3. Final Prediction with adapted weights
            # Detach to stop gradient tracking
            out = self._forward_functional(x, w_ttt.detach())
            return out

    def _forward_impl(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Standard module forward using registered parameter."""
        # x: [B, L, D]
        x1, x2 = self.in_proj(x).chunk(2, dim=-1)

        # Logic: x1 * Silu(Adapter(x2))
        # Use functional linear for adapter
        x2_adapt = F.linear(x2, weight)
        x_act = x1 * F.silu(x2_adapt)

        return self.out_proj(x_act)

    def _forward_functional(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Forward pass using an explicit weight tensor for the adapter."""
        x1, x2 = self.in_proj(x).chunk(2, dim=-1)
        x2_adapt = F.linear(x2, weight)  # Explicit weight
        x_act = x1 * F.silu(x2_adapt)
        return self.out_proj(x_act)


def create_ttt_mamba(**kwargs):
    """create_ttt_mamba.

    Returns:
        Any: Description.
    """
    return TTTMambaBlock(**kwargs)
