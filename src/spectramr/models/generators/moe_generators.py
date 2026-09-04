"""Mixture of Experts Generators
=================================

Sparse routing and dense MoE architectures for MRI reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 1. Mixture of Experts — dense gating with multiple expert sub-networks
# ---------------------------------------------------------------------------


@register_model(name="mixture_of_experts", training_mode="reconstruction")
class MixtureOfExpertsGenerator(nn.Module, IGenerator):
    """Dense MoE: all experts contribute, weighted by a learned gating network."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_experts: int = 4,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.shared_enc = nn.Sequential(
            _ConvBN(in_channels, bf), nn.MaxPool2d(2), _ConvBN(bf, bf * 2), nn.MaxPool2d(2)
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    _ConvBN(bf * 2, bf * 4),
                    nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
                    nn.GELU(),
                    nn.Conv2d(bf, out_channels, 1),
                )
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bf * 2, n_experts), nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.shared_enc(x)
        weights = self.gate(feat)  # (B, n_experts)
        expert_outs = torch.stack(
            [
                F.interpolate(e(feat), size=(H, W), mode="bilinear", align_corners=False)
                for e in self.experts
            ]
        )
        return (weights.T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * expert_outs).sum(0)


# ---------------------------------------------------------------------------
# 2. MoE Sparse Routing — top-k expert selection
# ---------------------------------------------------------------------------


@register_model(name="moe_sparse_routing", training_mode="reconstruction")
class MoESparseRoutingGenerator(nn.Module, IGenerator):
    """Sparse MoE: only top-k experts activated per sample (Switch Transformer style)."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_experts: int = 8,
        top_k: int = 2,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.top_k = top_k
        self.shared_enc = nn.Sequential(
            _ConvBN(in_channels, bf), nn.MaxPool2d(2), _ConvBN(bf, bf * 2), nn.MaxPool2d(2)
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    _ConvBN(bf * 2, bf * 4),
                    nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
                    nn.GELU(),
                    nn.Conv2d(bf, out_channels, 1),
                )
                for _ in range(n_experts)
            ]
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bf * 2, n_experts)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.shared_enc(x)
        logits = self.router(feat)  # (B, n_experts)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)

        result = torch.zeros(B, self.experts[0][-1].out_channels, H, W, device=x.device)
        for b in range(B):
            for k in range(self.top_k):
                expert_idx = topk_idx[b, k].item()
                out = F.interpolate(
                    self.experts[expert_idx](feat[b : b + 1]),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                result[b] += weights[b, k] * out[0]
        return result
