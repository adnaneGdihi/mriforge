r"""Top-1 sparse mixture-of-experts router.

Implements the gating + load-balance loss of Shazeer *et al.* [1]. A
linear gate maps a latent to a probability simplex over ``K`` experts;
top-1 routing dispatches each example to its arg-max expert. The
load-balance auxiliary loss

.. math::

    \mathcal{L}_{\text{aux}} = K \sum_{k=1}^{K} f_k\, P_k,

with :math:`f_k` the empirical routing fraction and :math:`P_k` the mean
gate probability, is minimised (value 1) when both are uniform —
preventing collapse onto a single expert.

Reference:
    [1] N. Shazeer *et al.*, "Outrageously large neural networks: the
        sparsely-gated mixture-of-experts layer," ICLR 2017.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Top1Router(nn.Module):
    """Linear gate with top-1 routing and a load-balance auxiliary loss.

    Args:
        in_dim: Dimensionality of the latent fed to the gate.
        num_experts: Number of experts ``K`` (``>= 2``).
    """

    def __init__(self, in_dim: int, num_experts: int) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"Top1Router requires num_experts >= 2, got {num_experts}")
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route a batch of latents.

        Args:
            z: Latent vectors ``(B, in_dim)``.

        Returns:
            ``(k_star, gate_val, probs)`` where ``k_star`` is the chosen
            expert index ``(B,)``, ``gate_val`` is the (differentiable)
            top-1 probability ``(B,)``, and ``probs`` is the full gate
            distribution ``(B, K)``.
        """
        logits = self.gate(z)
        probs = F.softmax(logits, dim=-1)
        gate_val, k_star = probs.max(dim=-1)
        return k_star, gate_val, probs

    def load_balance_loss(self, k_star: Tensor, probs: Tensor) -> Tensor:
        r"""Compute :math:`K \sum_k f_k P_k`.

        Args:
            k_star: Selected expert indices ``(B,)``.
            probs: Gate distribution ``(B, K)``.

        Returns:
            Scalar load-balance loss; minimum value 1.0 at uniform load.
        """
        K = self.num_experts
        # f_k: fraction of the batch routed to expert k.
        one_hot = F.one_hot(k_star, num_classes=K).to(probs.dtype)
        f = one_hot.mean(dim=0)  # (K,)
        # P_k: mean gate probability for expert k.
        P = probs.mean(dim=0)  # (K,)
        return K * torch.sum(f * P)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, num_experts={self.num_experts}"


__all__ = ["Top1Router"]
