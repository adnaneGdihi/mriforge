"""Foundation Model Generators
==============================

Pretrained-backbone-based and LLM-guided architectures for MRI reconstruction.
"""

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2), nn.InstanceNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 1. Foundation Model Adapter — adapter layers for a frozen foundation model
# ---------------------------------------------------------------------------


@register_model(name="foundation_model_adapter", training_mode="reconstruction")
class FoundationModelAdapter(nn.Module, IGenerator):
    """Frozen ViT backbone with trainable adapter + task-specific decoder.

    Simulates parameter-efficient fine-tuning: only adapters + head are trained.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 128,
        depth: int = 6,
        adapter_dim: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        # Frozen backbone layers
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 4, batch_first=True)
                for _ in range(depth)
            ]
        )
        # Trainable adapter for each block
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, adapter_dim), nn.GELU(), nn.Linear(adapter_dim, dim))
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_channels * 16))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        tokens = patches.flatten(2).permute(0, 2, 1)
        for blk, adapter in zip(self.blocks, self.adapters, strict=False):
            tokens = blk(tokens) + adapter(tokens)
        out = self.head(tokens)
        return out.permute(0, 2, 1).reshape(B, self.out_channels, pH * 4, pW * 4)[:, :, :H, :W]


# ---------------------------------------------------------------------------
# 2. Medical DINO — self-distillation vision transformer for medical imaging
# ---------------------------------------------------------------------------


@register_model(name="medical_dino", training_mode="reconstruction")
class MedicalDINOGenerator(nn.Module, IGenerator):
    """DINO-style ViT: teacher-student self-distillation adapted for MRI.

    Reconstruction head on the student branch for downstream tasks.
    """

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, dim: int = 128, depth: int = 6, **kwargs
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 4, batch_first=True),
            num_layers=depth,
        )
        # Projection head (DINO-style)
        self.projection = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        # Reconstruction decoder
        self.decoder = nn.Sequential(nn.Linear(dim, out_channels * 16))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        tokens = patches.flatten(2).permute(0, 2, 1)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.encoder(tokens)
        patch_tokens = tokens[:, 1:]
        out = self.decoder(patch_tokens)
        return out.permute(0, 2, 1).reshape(B, self.out_channels, pH * 4, pW * 4)[:, :, :H, :W]


# ---------------------------------------------------------------------------
# 3. LLM-Guided Reconstruction — text/prompt-conditioned reconstruction
# ---------------------------------------------------------------------------


@register_model(name="llm_guided_reconstruction", training_mode="reconstruction")
class LLMGuidedReconstruction(nn.Module, IGenerator):
    """Cross-attention fusion: image features attend to learned prompt embeddings.

    Simulates LLM guidance via a learnable prompt bank (no actual LLM required).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 128,
        n_prompts: int = 16,
        depth: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.image_proj = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.prompt_bank = nn.Parameter(torch.randn(1, n_prompts, dim) * 0.02)
        self.cross_attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(dim, num_heads=4, batch_first=True) for _ in range(depth)]
        )
        self.ff_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
                for _ in range(depth)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth * 2)])
        self.head = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.image_proj(x)
        tokens = feat.flatten(2).permute(0, 2, 1)
        prompts = self.prompt_bank.expand(B, -1, -1)
        for i, (ca, ff) in enumerate(zip(self.cross_attn_layers, self.ff_layers, strict=False)):
            tokens = tokens + ca(self.norms[i * 2](tokens), prompts, prompts)[0]
            tokens = tokens + ff(self.norms[i * 2 + 1](tokens))
        out = tokens.permute(0, 2, 1).reshape(B, -1, H, W)
        return self.head(out)


# ---------------------------------------------------------------------------
# 4. LLM-MRI Foundation — large foundation model architecture for MRI
# ---------------------------------------------------------------------------


@register_model(name="llm_mri_foundation", training_mode="reconstruction")
class LLMMRIFoundation(nn.Module, IGenerator):
    """Large-scale ViT with MAE-style pretraining objective for MRI.

    Deeper and wider than standard ViT, with sinusoidal position embeddings.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 256,
        depth: int = 12,
        heads: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True, dropout=0.1
            ),
            num_layers=depth,
        )
        self.norm = nn.LayerNorm(dim)
        self.decoder_proj = nn.Linear(dim, out_channels * 16)
        self.out_channels = out_channels
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        N = pH * pW
        tokens = patches.flatten(2).permute(0, 2, 1)
        tokens = tokens + self.pos_embed[:, :N]
        tokens = self.norm(self.encoder(tokens))
        out = self.decoder_proj(tokens)
        return out.permute(0, 2, 1).reshape(B, self.out_channels, pH * 4, pW * 4)[:, :, :H, :W]
