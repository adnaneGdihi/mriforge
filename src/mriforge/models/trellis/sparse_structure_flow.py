"""Sparse Structure Flow model adapted for MRIForge integration.

Migrated from src/implementations/trellis/models/sparse_structure_flow.py
during consolidation.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mriforge.models.blocks.trellis_spatial import patchify, unpatchify
from mriforge.models.blocks.trellis_transformer import AbsolutePositionEmbedder
from mriforge.models.blocks.trellis_utils import convert_module_to_f16, convert_module_to_f32


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into a vector representation."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        """__init__.

        Args:
            hidden_size (int): Description.
            frequency_embedding_size (int): Description.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        # Precompute frequencies
        dim = frequency_embedding_size
        max_period = 10_000
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(float(max_period)))
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / max(half, 1),
        )
        self.register_buffer("freqs", freqs)

    def timestep_embedding(
        self,
        timesteps: torch.Tensor,
        dim: int,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        # Use cached freqs if available and matching dimension
        """timestep_embedding.

        Args:
            timesteps (torch.Tensor): Description.
            dim (int): Description.
            max_period (int): Description.
        Returns:
            torch.Tensor: Description.
        """
        if hasattr(self, "freqs") and self.freqs.shape[0] == dim // 2:
            freqs = self.freqs
        else:
            half = dim // 2
            freqs = torch.exp(
                -torch.log(torch.tensor(float(max_period)))
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / max(half, 1),
            ).to(device=timesteps.device)

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """forward.

        Args:
            timesteps (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TimestepEmbedder.

        Executes PyTorch tensor operations.

        Args:
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        embedding = self.timestep_embedding(
            timesteps,
            self.frequency_embedding_size,
        )
        return self.mlp(embedding)


class SparseStructureFlowModel(nn.Module):
    """Lightweight port of TRELLIS sparse structure flow."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: int | None = None,
        num_head_channels: int | None = 64,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ) -> None:
        """__init__.

        Args:
            resolution (int): Description.
            in_channels (int): Description.
            model_channels (int): Description.
            cond_channels (int): Description.
            out_channels (int): Description.
            num_blocks (int): Description.
            num_heads (int | None): Description.
            num_head_channels (int | None): Description.
            mlp_ratio (float): Description.
            patch_size (int): Description.
            pe_mode (Literal['ape', 'rope']): Description.
            use_fp16 (bool): Description.
            use_checkpoint (bool): Description.
            share_mod (bool): Description.
            qk_rms_norm (bool): Description.
            qk_rms_norm_cross (bool): Description.
        """
        super().__init__()
        if pe_mode not in ["ape", "rope"]:
            raise ValueError(f"Unsupported position embedding mode: {pe_mode}")

        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or max(
            model_channels // max(num_head_channels, 1),
            1,
        )
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation: nn.Sequential | None = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )
        else:
            self.adaLN_modulation = None

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(
                *[torch.arange(resolution // patch_size, device=self.device) for _ in range(3)],
                indexing="ij",
            )
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb, persistent=False)
        else:
            self.pos_emb = None

        self.input_layer = nn.Linear(
            in_channels * patch_size**3,
            model_channels,
        )

        # Import here to avoid circular imports
        from mriforge.models.blocks.trellis_transformer import ModulatedTransformerCrossBlock

        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_rope=(pe_mode == "rope"),
                )
                for _ in range(num_blocks)
            ],
        )

        self.out_layer = nn.Linear(
            model_channels,
            out_channels * patch_size**3,
        )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """device.

        Returns:
            torch.device: Description.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """convert_to_fp16."""
        self.blocks.apply(convert_module_to_f16)
        self.dtype = torch.float16

    def convert_to_fp32(self) -> None:
        """convert_to_fp32."""
        self.blocks.apply(convert_module_to_f32)
        self.dtype = torch.float32

    def initialize_weights(self) -> None:
        """initialize_weights."""

        def _basic_init(module: nn.Module) -> None:
            """_basic_init.

            Args:
                module (nn.Module): Description.
            """
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.share_mod and self.adaLN_modulation is not None:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """forward.

        Args:
            x (torch.Tensor): Description.
            timesteps (torch.Tensor): Description.
            cond (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SparseStructureFlowModel.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        expected = [
            x.shape[0],
            self.in_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        ]
        if list(x.shape) != expected:
            actual = list(x.shape)
            message = f"Input shape mismatch, expected {expected}, got {actual}"
            raise ValueError(message)

        x = x.to(self.dtype)
        cond = cond.to(self.dtype)

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h.permute(0, 2, 1)  # (batch, seq_len, model_channels)
        if self.pe_mode == "ape":
            pos_emb = self.pos_emb
            h = h + pos_emb.unsqueeze(0)
        t_emb = self.t_embedder(timesteps)
        if self.adaLN_modulation is not None:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.to(self.dtype)
        h = h.to(self.dtype)
        for block in self.blocks:
            if self.use_checkpoint:
                h = checkpoint(block, h, t_emb, cond, use_reentrant=False)
            else:
                h = block(h, t_emb, cond)
        h = h.to(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        spatial = self.resolution // self.patch_size
        h = h.view(
            h.shape[0],
            spatial,
            spatial,
            spatial,
            h.shape[-1],
        )
        return unpatchify(h, self.patch_size).contiguous()


__all__ = ["SparseStructureFlowModel", "TimestepEmbedder"]
