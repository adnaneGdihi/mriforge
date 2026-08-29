"""Sparse Autoencoder for interpretability (PR-24 / Part-I F).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-24.

Trained on top of a frozen reconstruction model's intermediate
features. The SAE expands the feature dim by a factor (typically 8x)
and adds an L1 sparsity penalty so each "feature" is only active for a
small fraction of inputs — making the bottleneck interpretable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """L1-sparse top-K autoencoder for interpretability.

    Args:
        input_dim: feature dim of the frozen backbone.
        expansion_factor: how much wider the SAE bottleneck is.
        l1_lambda: L1 weight on the bottleneck activations.
        top_k: optional ``k`` for top-k sparsity (Anthropic-style); set
               to ``None`` to use plain L1 penalty.
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 8,
        l1_lambda: float = 1e-3,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        hidden = input_dim * expansion_factor
        self.encoder = nn.Linear(input_dim, hidden)
        self.decoder = nn.Linear(hidden, input_dim)
        self.l1_lambda = l1_lambda
        self.top_k = top_k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.encoder(x))
        if self.top_k is not None:
            # top-K activation: keep top_k largest, zero the rest per sample.
            topk_vals, topk_idx = z.topk(self.top_k, dim=-1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(-1, topk_idx, topk_vals)
            return z_sparse
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_hat, z = self.forward(x)
        recon = (x_hat - x).pow(2).mean()
        sparsity = z.abs().mean()
        total = recon + self.l1_lambda * sparsity
        return total, {
            "loss_total": total,
            "loss_recon": recon.detach(),
            "loss_sparsity": sparsity.detach(),
            "active_fraction": (z > 0).float().mean().detach(),
        }


__all__ = ["SparseAutoencoder"]
