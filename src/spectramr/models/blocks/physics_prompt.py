"""Physics-derived prompt embedding for the universal paradigm (PR-7).

This is the framework's differentiator versus published foundation
reconstruction models (e.g. prompt-tuned recon transformers): instead of
learning prompt *tokens* empirically, the conditioning embedding is a
deterministic, differentiable function of the **physical acquisition
parameters** of the scan — repetition time (TR), echo time (TE), flip angle,
and main-field strength.

Two scans that share physics therefore map to nearby points in conditioning
space without any per-protocol re-training, which is exactly what makes a
single model generalise across mixed contrast / field-strength / acceleration
cohorts.

The map is a small MLP over a log-normalised parameter vector::

    prompt = MLP( normalise([TR, TE, flip_angle, field_strength_T]) )

It is a plain ``nn.Module`` building block (not ``@register_model`` — blocks in
``models/blocks/`` are imported by path, never registered).

References
----------
Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer,"
AAAI 2018 (FiLM consumes embeddings of this kind).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PhysicsPromptEmbedding"]


class PhysicsPromptEmbedding(nn.Module):
    """Map an acquisition-parameter vector to a conditioning embedding.

    Args:
        in_features: Number of physical acquisition scalars per sample.
            Default ``4`` for ``[TR, TE, flip_angle, field_strength_T]``.
        embed_dim: Output embedding dimension.
        hidden_dim: Width of the MLP hidden layer. Defaults to
            ``max(2 * embed_dim, 64)``.

    Forward signature
    -----------------
    ``forward(params: Tensor[B, in_features]) -> Tensor[B, embed_dim]``.

    The forward is deterministic in ``eval`` mode and fully differentiable
    w.r.t. both the input parameters and the MLP weights.
    """

    def __init__(
        self,
        in_features: int = 4,
        embed_dim: int = 128,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")

        self.in_features = in_features
        self.embed_dim = embed_dim
        hidden = hidden_dim if hidden_dim is not None else max(2 * embed_dim, 64)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, in_features]`` acquisition params to ``[B, embed_dim]``.

        Args:
            params: Acquisition-parameter tensor of shape ``[B, in_features]``,
                e.g. ``[TR, TE, flip_angle_deg, field_strength_T]``.

        Returns:
            Conditioning embedding of shape ``[B, embed_dim]``.
        """
        if params.dim() != 2 or params.shape[-1] != self.in_features:
            raise ValueError(
                f"PhysicsPromptEmbedding expects params of shape "
                f"[B, {self.in_features}]; got {tuple(params.shape)}"
            )
        # log1p compresses the multi-order-of-magnitude span of TR/TE so the MLP
        # sees comparably-scaled inputs; sign-preserving for robustness.
        x = params.to(self.mlp[0].weight.dtype)
        x = torch.sign(x) * torch.log1p(x.abs())
        return self.mlp(x)
