"""Hilbert-Mamba Generators for ULF-to-HF MRI Reconstruction.

Ten novel generator architectures that combine Hilbert space-filling curve
linearization with Mamba (Structured State Space Models) for paired
Ultra-Low Field to High Field MRI super-resolution.

Key idea: A Hilbert curve maps 2D/3D grids into locality-preserving 1D
sequences.  Mamba processes these sequences in O(L) time vs O(L²) for
Transformers, enabling tractable volumetric processing.

Methods:
    1. CT-Mamba    — Continuous-time discretization resampling
    2. FE-Mamba    — Fractal expansion Kronecker upsampling
    3. CRM-Mamba   — Cross-resolution modulated SSM
    4. HW-Mamba    — Hilbert-wavelet spectral detail prediction
    5. MM-Mamba    — Macro-micro nested hierarchical processing
    6. MDI-Mamba   — Multi-directional isotropic fusion
    7. 2.5D-Mamba  — Orthogonal slice-to-volume cross-plane
    8. HLD-Mamba   — Hilbert latent diffusion noise predictor
    9. INR-Mamba   — Implicit neural representation continuous SR
   10. Hilbert-Mamba U-Net — Combined baseline (FE + MDI)

Reference:
    Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces." arXiv:2312.00752.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.blocks.mamba_block import MambaBlock
from mriforge.models.blocks.topology_linearizer import (
    ImageTopologyLinearizer,
    resolution_agnostic_mode,
)
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


def _squeeze_volume_to_2d(
    x: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    """Squeeze 5D ``[B, C, D, H, W]`` volumes to 4D ``[B, C, H, W]``.

    TorchIO validation :class:`~torchio.SubjectsLoader` returns full 3D
    volumes instead of 2D patches.  Generators built with :class:`Conv2d`
    stems crash on the 5D input.  This helper transparently handles:

    * ``[B, C, H, W, 1]`` — squeeze last dim (TorchIO 2D patch)
    * ``[B, C, D, H, W]`` where ``D > 1`` — take middle slice
    * ``[B, C, H, W]`` — no-op

    Returns:
        Tuple of (squeezed 4D tensor, original shape or ``None`` if no-op).
    """
    if x.ndim == 5 and x.shape[-1] == 1:
        return x.squeeze(-1), x.shape
    if x.ndim == 5:
        D = x.shape[2]
        mid = D // 2
        return x[:, :, mid].contiguous(), x.shape
    return x, None


def _unsqueeze_to_volume(out: torch.Tensor, original_shape: tuple[int, ...] | None) -> torch.Tensor:
    """Restore 5D shape if the input was squeezed by :func:`_squeeze_volume_to_2d`."""
    if original_shape is None:
        return out
    if original_shape[-1] == 1:
        return out.unsqueeze(-1)
    # Middle-slice case: expand prediction back to full volume depth
    D = original_shape[2]
    return out.unsqueeze(2).expand(-1, -1, D, -1, -1)


# ═══════════════════════════════════════════════════════════════════════
# Shared building blocks
# ═══════════════════════════════════════════════════════════════════════


class _HilbertLinearize(nn.Module):
    """Lazy Hilbert linearizer that builds its permutation on first call.

    Supports both 2D and 3D modes and caches per spatial resolution.
    The ``mode`` parameter controls which linearization scheme is used,
    allowing the user to select among all modes registered in
    ``ImageTopologyLinearizer`` (hilbert_2d, hilbert_3d, morton_2d, etc.).
    """

    def __init__(self, mode: str = "hilbert_2d") -> None:
        super().__init__()
        self.mode = mode
        self._linearizer: ImageTopologyLinearizer | None = None
        self._cached_shape: tuple[int, ...] | None = None

    def _get_linearizer(
        self, spatial_shape: tuple[int, ...], device: torch.device
    ) -> ImageTopologyLinearizer:
        if self._cached_shape != spatial_shape:
            self._linearizer = ImageTopologyLinearizer(
                spatial_shape, mode=resolution_agnostic_mode(self.mode)
            )
            self._linearizer = self._linearizer.to(device)
            self._cached_shape = spatial_shape
        return self._linearizer  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten spatial dims into Hilbert-ordered 1D sequence.

        Args:
            x: ``[B, C, H, W]`` or ``[B, C, D, H, W]``.

        Returns:
            Tuple of (sequence ``[B, C, L]``, original spatial shape).

        forward method for _HilbertLinearize.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, tuple[int, ...]]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        spatial = x.shape[2:]
        lin = self._get_linearizer(spatial, x.device)
        return lin(x), spatial

    def reverse(self, seq: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        """Restore spatial dims from Hilbert-ordered 1D sequence."""
        lin = self._get_linearizer(spatial_shape, seq.device)
        return lin.reverse(seq)


class _MambaEncoder(nn.Module):
    """Stack of Mamba blocks operating on [B, L, D] sequences."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process [B, L, D] sequence through Mamba stack with residuals.

        forward method for _MambaEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)


class _ConvStem(nn.Module):
    """Patch-embed or channel projection for Hilbert-Mamba generators."""

    def __init__(self, in_ch: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(min(8, d_model), d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ConvStem.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.proj(x)


class _ConvHead(nn.Module):
    """Output head: project back to target channels."""

    def __init__(self, d_model: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ConvHead.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════
# Method 1: CT-Mamba — Continuous-Time Discretization Resampling
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="ct_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class CTMamba(nn.Module):
    """Continuous-Time Mamba with asymmetric encode/decode step sizes.

    Encodes the ULF sequence at a coarse discretization step Δ_U and
    decodes the HF output at a finer step Δ_H, exploiting the SSM's
    continuous-time ODE foundation for natural multi-rate processing.

    Math:
        ḣ(t) = A h(t) + B x(t),  y(t) = C h(t)
        Encode ULF with Δ_U = 1/L_U, decode HF with Δ_H = 1/L_H.

    Args:
        in_channels: Input channels (ULF).
        out_channels: Output channels (HF).
        d_model: Mamba model dimension.
        num_layers: Number of Mamba blocks.
        d_state: SSM state dimension.
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.encoder = _MambaEncoder(d_model, num_layers, d_state)

        # Learnable step-size parameters for asymmetric discretization
        self.delta_encoder = nn.Parameter(torch.ones(d_model) * 1.0)
        self.delta_decoder = nn.Parameter(torch.ones(d_model) * 0.5)

        # Decoder: Mamba stack at fine resolution
        self.decoder = _MambaEncoder(d_model, num_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

        logger.info(
            "[CTMamba] d_model=%d, layers=%d, mode=%s",
            d_model,
            num_layers,
            linearization_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF [B, C_in, H, W] → HF [B, C_out, H, W].

        forward method for CTMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]

        # Encode with coarse step modulation (clamp sigmoid to avoid vanishing)
        seq = seq * torch.sigmoid(self.delta_encoder).clamp(min=0.01, max=0.99)
        latent = self.encoder(seq)

        # Decode with fine step modulation
        latent = latent * torch.sigmoid(self.delta_decoder).clamp(min=0.01, max=0.99)
        decoded = self.decoder(latent)

        decoded = decoded.permute(0, 2, 1)  # [B, D, L]
        out = self.linearize.reverse(decoded, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 2: FE-Mamba — Fractal Expansion Kronecker
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="fe_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class FEMamba(nn.Module):
    """Fractal Expansion Mamba with Kronecker upsampling.

    Exploits Hilbert's fractal self-similarity: one ULF voxel perfectly
    maps to s^D contiguous HF sub-voxels.  A linear expansion matrix W
    generates initial sub-voxel estimates, then Mamba smoothes boundaries.

    Math:
        z_{i·s^D + j} = W_j · x_i + b_j,  ∀ j ∈ {0…s^D-1}
        ŷ = Mamba_dec(z)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Number of Mamba blocks.
        expansion_factor: Sub-pixel expansion factor per dimension.
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        expansion_factor: int = 2,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.expansion_factor = expansion_factor
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        # Separate linearizer for upscaled output to avoid cache pollution
        self._up_linearize = _HilbertLinearize(mode=linearization_mode)

        # Fractal expansion: each token → expansion_factor^2 tokens
        expand_count = expansion_factor**2
        self.expand_proj = nn.Linear(d_model, d_model * expand_count)

        self.smoother = _MambaEncoder(d_model, num_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

        logger.info(
            "[FEMamba] d_model=%d, expand=%d, mode=%s",
            d_model,
            expansion_factor,
            linearization_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF [B, C_in, H, W] → HF [B, C_out, sH, sW].

        forward method for FEMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        B, _, H, W = x.shape
        s = self.expansion_factor
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]

        # Fractal expansion: [B, L, D] → [B, L, D*s²] → [B, L*s², D]
        expanded = self.expand_proj(seq)  # [B, L, D*s²]
        expanded = expanded.reshape(B, -1, seq.shape[-1])  # [B, L*s², D]

        # Smooth boundaries with Mamba
        smoothed = self.smoother(expanded)
        smoothed = smoothed.permute(0, 2, 1)  # [B, D, L*s²]

        # Reverse at upscaled resolution (separate linearizer to avoid cache pollution)
        up_shape = (H * s, W * s)
        up_lin = self._up_linearize._get_linearizer(up_shape, x.device)
        out = up_lin.reverse(smoothed)
        out = self.head(out)

        # Residual: bilinear upsample input to match output resolution
        out_ch = self.head.proj[-1].out_channels
        residual = F.interpolate(x[:, :out_ch], size=up_shape, mode="bilinear", align_corners=False)
        out = out + residual
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 3: CRM-Mamba — Cross-Resolution Modulated
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="crm_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class CRMMamba(nn.Module):
    """Cross-Resolution Modulated Mamba.

    Generates HF sequences while dynamically conditioning the SSM
    transition matrices on ULF structural context.  The discretization
    step Δ_j at each HF position is modulated by its ULF parent state.

    Math:
        H^U = Mamba_enc(x)
        Δ_j = Softplus(Linear(H^U_{⌊j/s^D⌋} + y_{j-1}))
        h_j = Ā_j h_{j-1} + B̄_j x_j,  where Ā uses Δ_j

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Number of Mamba blocks per encoder/decoder.
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.encoder = _MambaEncoder(d_model, num_layers, d_state)

        # Cross-resolution modulation network
        self.delta_modulator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.softplus = nn.Softplus()

        self.decoder = _MambaEncoder(d_model, num_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF [B, C_in, H, W] → HF [B, C_out, H, W].

        forward method for CRMMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)
        seq = seq.permute(0, 2, 1)  # [B, L, D]

        # Encode ULF context
        ulf_context = self.encoder(seq)  # [B, L, D]

        # Cross-resolution modulation: concatenate context with input
        combined = torch.cat([ulf_context, seq], dim=-1)  # [B, L, 2D]
        delta_mod = self.softplus(self.delta_modulator(combined))  # [B, L, D]
        delta_mod = delta_mod.clamp(max=10.0)  # Prevent unbounded multiplicative explosion

        # Modulate and decode
        modulated = seq * delta_mod
        decoded = self.decoder(modulated)

        decoded = decoded.permute(0, 2, 1)
        out = self.linearize.reverse(decoded, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 4: HW-Mamba — Hilbert-Wavelet Spectral
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="hw_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class HWMamba(nn.Module):
    """Hilbert-Wavelet Spectral Mamba.

    Instead of predicting raw pixels, uses 1D DWT on the Hilbert-ordered
    sequence.  Mamba predicts only the sparse high-frequency detail
    coefficients, while the ULF directly provides low-frequency approx.

    Math:
        c_A, c_D = DWT(y)     # target decomposition
        c_A ≈ x               # ULF approximates low-freq
        ĉ_D = Mamba(x)        # predict detail coefficients
        ŷ = IDWT(x, ĉ_D)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Number of Mamba blocks.
        wavelet_levels: Number of DWT decomposition levels (unused at
            inference, included for config symmetry).
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        wavelet_levels: int = 3,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)

        # Detail predictor: Mamba predicts wavelet detail coefficients
        self.detail_predictor = _MambaEncoder(d_model, num_layers, d_state)

        # Learned wavelet-like decomposition (trainable 1D conv filters)
        # Instead of fixed wavelet, we learn analysis/synthesis filters
        self.analysis_low = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.analysis_high = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.synthesis = nn.ConvTranspose1d(
            d_model * 2, d_model, kernel_size=4, stride=2, padding=1
        )

        self.head = _ConvHead(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF [B, C_in, H, W] → HF [B, C_out, H, W].

        forward method for HWMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]

        # Analysis: decompose into approximate + detail
        approx = self.analysis_low(seq)  # [B, D, L//2]
        detail = self.analysis_high(seq)  # [B, D, L//2]

        # Predict refined detail coefficients with Mamba
        detail_input = detail.permute(0, 2, 1)  # [B, L//2, D]
        refined_detail = self.detail_predictor(detail_input)
        refined_detail = refined_detail.permute(0, 2, 1)  # [B, D, L//2]

        # Synthesis: reconstruct with refined details
        combined = torch.cat([approx, refined_detail], dim=1)  # [B, 2D, L//2]
        reconstructed = self.synthesis(combined, output_size=seq.shape)  # [B, D, L]

        out = self.linearize.reverse(reconstructed, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 5: MM-Mamba — Macro-Micro Nested
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="mm_mamba",
    training_mode="reconstruction",
    # Macro-Micro Nested Mamba: explicitly designed for tractable 3D
    # volumetric processing. Doc: divides Hilbert curve into local
    # blocks. Accepts both 2D (degenerate single-block) and 3D.
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class MMMamba(nn.Module):
    """Macro-Micro Nested Mamba for tractable 3D volumetric processing.

    Divides the Hilbert curve into local blocks (micro-curves), processes
    each independently, then runs a macro-level Mamba over block summaries.
    Global context is fused back into local features for HF generation.

    Math:
        h_micro^(i) = MaxPool(Mamba_micro(P_i))
        Z_global = Mamba_macro([h_micro^(1), ..., h_micro^(K)])
        output_i = Mamba_dec(P_i + context_i)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        micro_layers: Mamba blocks per micro encoder.
        macro_layers: Mamba blocks for macro encoder.
        block_size: Tokens per micro block.
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        micro_layers: int = 2,
        macro_layers: int = 2,
        block_size: int = 1024,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)

        self.micro_encoder = _MambaEncoder(d_model, micro_layers, d_state)
        self.macro_encoder = _MambaEncoder(d_model, macro_layers, d_state)

        # Global context injection
        self.context_proj = nn.Linear(d_model, d_model)
        self.decoder = _MambaEncoder(d_model, micro_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF → HF with macro-micro hierarchy.

        forward method for MMMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]
        B, L, D = seq.shape

        # Pad to multiple of block_size
        pad_len = (self.block_size - L % self.block_size) % self.block_size
        if pad_len > 0:
            seq = F.pad(seq, (0, 0, 0, pad_len))

        num_blocks = seq.shape[1] // self.block_size

        # Micro: process each block independently
        blocks = seq.reshape(B * num_blocks, self.block_size, D)
        micro_out = self.micro_encoder(blocks)  # [B*K, block, D]

        # Macro: pool each block → summary, run global Mamba
        block_summaries = micro_out.mean(dim=1)  # [B*K, D]
        block_summaries = block_summaries.reshape(B, num_blocks, D)
        global_ctx = self.macro_encoder(block_summaries)  # [B, K, D]

        # Fuse global context back into local features
        ctx_expanded = self.context_proj(global_ctx)  # [B, K, D]
        ctx_expanded = ctx_expanded.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        ctx_expanded = ctx_expanded.reshape(B, -1, D)  # [B, K*block, D]

        fused = micro_out.reshape(B, -1, D) + ctx_expanded
        decoded = self.decoder(fused)

        # Remove padding and restore spatial
        decoded = decoded[:, :L, :].permute(0, 2, 1)
        out = self.linearize.reverse(decoded, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 6: MDI-Mamba — Multi-Directional Isotropic
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="mdi_mamba",
    training_mode="reconstruction",
    # Multi-Directional Isotropic Mamba: K parallel scan directions
    # (Hilbert / Morton / Snake / Zigzag — all 2D variants per the
    # DEFAULT_MODES). Acts as a multi-head Mamba via scan-direction
    # diversity. The 2D restriction is from the default mode list.
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class MDIMamba(nn.Module):
    """Multi-Directional Isotropic Mamba.

    Processes the input through K different Hilbert curve orientations
    (or other linearization modes), each with its own Mamba branch,
    then averages in spatial domain to compensate for boundary tears.

    Math:
        V̂_HF = (1/K) Σ_k H_k⁻¹(Mamba_k(H_k(V^ULF)))

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Mamba blocks per branch.
        num_directions: Number of scan directions (K).
        linearization_modes: List of linearization modes to use. If not
            provided, defaults to ``["hilbert_2d", "morton_2d",
            "snake_2d", "zigzag_2d"]`` (first K entries).
    """

    # Default modes ordered by locality quality
    DEFAULT_MODES = [
        "hilbert_2d",
        "morton_2d",
        "snake_2d",
        "zigzag_2d",
        "radial_inside_out",
        "radial_outside_in",
    ]

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        num_directions: int = 4,
        d_state: int = 16,
        linearization_modes: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Resolve modes: user-specified or first K defaults
        if linearization_modes is not None:
            self.modes = linearization_modes[:num_directions]
        else:
            self.modes = self.DEFAULT_MODES[:num_directions]

        self.num_directions = len(self.modes)
        self.stem = _ConvStem(in_channels, d_model)

        # Per-direction Mamba branches + linearizers
        self.linearizers = nn.ModuleList([_HilbertLinearize(mode=m) for m in self.modes])
        self.branches = nn.ModuleList(
            [_MambaEncoder(d_model, num_layers, d_state) for _ in self.modes]
        )

        self.head = _ConvHead(d_model, out_channels)

        logger.info("[MDIMamba] modes=%s, d_model=%d", self.modes, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: multi-directional average in spatial domain.

        forward method for MDIMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        accum = torch.zeros_like(feat)

        for lin, branch in zip(self.linearizers, self.branches, strict=False):
            seq, shape = lin(feat)  # [B, D, L]
            seq = seq.permute(0, 2, 1)  # [B, L, D]
            processed = branch(seq)  # [B, L, D]
            processed = processed.permute(0, 2, 1)
            restored = lin.reverse(processed, shape)
            accum = accum + restored

        averaged = accum / self.num_directions
        out = self.head(averaged) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 7: 2.5D-Mamba — Orthogonal Slice-to-Volume
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="two_half_d_mamba",
    training_mode="reconstruction",
    # 2.5D: input is 3D volume; processes 2D in-plane slices then a
    # cross-plane Mamba pass. Accepts 3D input only (the 2.5D pipeline
    # requires a depth axis to fuse).
    spatial_dims=(3,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class TwoHalfDMamba(nn.Module):
    """2.5D Orthogonal Slice-to-Volume Mamba.

    Processes 2D slices in-plane with Hilbert-linearized Mamba, then
    runs a secondary Mamba cross-plane along the Z-axis to restore
    volumetric coherence.

    For 2D inputs (single slices), acts as a standard Hilbert-Mamba.
    For 3D inputs, the cross-plane pass enables inter-slice context.

    Math:
        f_z = Mamba_in_plane(H_2D(slice_z))  for each slice z
        y_{z,i} = Mamba_z_axis(F[:, i, :])   cross-plane at position i

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        in_plane_layers: Mamba blocks for in-plane processing.
        cross_plane_layers: Mamba blocks for Z-axis processing.
        linearization_mode: Linearization mode for 2D in-plane Hilbert.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        in_plane_layers: int = 4,
        cross_plane_layers: int = 2,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.in_plane_mamba = _MambaEncoder(d_model, in_plane_layers, d_state)
        self.cross_plane_mamba = _MambaEncoder(d_model, cross_plane_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: 2D or 3D input → enhanced output.

        For 2D: [B, C, H, W] → in-plane only.
        For 3D: [B, C, H, W, D] (TorchIO convention) → in-plane + cross-plane.

        Note: Unlike other Hilbert generators, TwoHalfDMamba natively
        supports 5D input so no squeeze is needed for the 3D path.
        The 5D path handles full volumes with cross-plane Mamba.

        forward method for TwoHalfDMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, H, W, D)):
                TorchIO convention with depth as the last axis. Earlier
                revisions of this file unpacked ``[B, C, D, H, W]`` and
                produced ``Hilbert 2D requires square power-of-2 dims,
                got (H, D)`` on rectangular volumes like the 2026-05-10
                ``exp_hm_07_25d_mamba`` smoke crash (``patch_size:
                [256, 256, 16]``).

        Returns:
            torch.Tensor: Output tensor in the same shape convention as the input.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        is_3d = x.dim() == 5

        if is_3d:
            # TorchIO yields [B, C, H, W, D] for volumetric patches.
            B, C, H, W, D = x.shape
            # Process each slice in-plane: fold depth into batch.
            x_2d = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
            feat = self.stem(x_2d)  # [B*D, d, H, W]
            seq, shape = self.linearize(feat)  # [B*D, d, L]
            seq = seq.permute(0, 2, 1)  # [B*D, L, d]
            in_plane = self.in_plane_mamba(seq)  # [B*D, L, d]

            # Reshape for cross-plane: [B, D, L, d]
            d_model = in_plane.shape[-1]
            L = in_plane.shape[1]
            cross_input = in_plane.reshape(B, D, L, d_model)

            # Cross-plane Mamba at each spatial position
            cross_input = cross_input.permute(0, 2, 1, 3)  # [B, L, D, d]
            cross_input = cross_input.reshape(B * L, D, d_model)
            cross_out = self.cross_plane_mamba(cross_input)  # [B*L, D, d]
            cross_out = cross_out.reshape(B, L, D, d_model)
            cross_out = cross_out.permute(0, 2, 1, 3)  # [B, D, L, d]
            cross_out = cross_out.reshape(B * D, L, d_model)

            # Restore spatial
            cross_out = cross_out.permute(0, 2, 1)  # [B*D, d, L]
            restored = self.linearize.reverse(cross_out, shape)  # [B*D, d, H, W]
            out = self.head(restored)  # [B*D, C_out, H, W]
            # Unfold back to TorchIO 5-D layout ``[B, C_out, H, W, D]``.
            return out.reshape(B, D, -1, H, W).permute(0, 2, 3, 4, 1)
        else:
            # Standard 2D: in-plane only
            feat = self.stem(x)
            seq, shape = self.linearize(feat)
            seq = seq.permute(0, 2, 1)
            processed = self.in_plane_mamba(seq)
            processed = processed.permute(0, 2, 1)
            out = self.linearize.reverse(processed, shape)
            return self.head(out) + x[:, : self.head.proj[-1].out_channels]


# ═══════════════════════════════════════════════════════════════════════
# Method 8 (Loss): Soft-DTW — see soft_dtw_loss.py
# The generator for Method 8 is HilbertMambaUNet with the DTW loss.
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Method 9: HLD-Mamba — Hilbert Latent Diffusion
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="hld_mamba",
    training_mode="diffusion",
    # Hilbert Latent Diffusion Mamba: 1D noise predictor in a latent
    # token sequence. The latent comes from a separately-trained VAE
    # and is itself spatial (2D), so spatial_dims is image-rank.
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class HLDMamba(nn.Module):
    """Hilbert Latent Diffusion Mamba — 1D noise predictor for diffusion.

    Replaces the heavy 2D/3D U-Net noise predictor in diffusion models
    with a 1D Mamba block operating on Hilbert-linearized sequences.
    The ULF input is concatenated with the noisy HF target for conditioning.

    Math:
        y_t = √ᾱ_t · y + √(1-ᾱ_t) · ε       # forward diffusion
        ε_θ = Mamba(y_t ∥ x↑ ∥ Emb(t))        # noise prediction

    Args:
        in_channels: Input channels (noisy HF + ULF conditioning).
        out_channels: Output channels (predicted noise, same as HF).
        d_model: Mamba model dimension.
        num_layers: Number of Mamba blocks.
        time_embed_dim: Timestep embedding dimension.
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        d_model: int = 128,
        num_layers: int = 6,
        d_state: int = 16,
        time_embed_dim: int = 256,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, d_model),
        )

        self.encoder = _MambaEncoder(d_model, num_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict noise ε_θ from noisy input + timestep.

        Args:
            x: Noisy input [B, C, H, W] (concatenated noisy_hf + ulf).
            timesteps: Diffusion timesteps [B] (normalized 0→1).

        Returns:
            Predicted noise [B, C_out, H, W].

        forward method for HLDMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]

        # Inject timestep embedding (normalize to [0,1] to prevent large activations)
        if timesteps is not None:
            t_norm = timesteps.float() / 1000.0  # Normalize assuming ~1000 max timesteps
            t_emb = self.time_mlp(t_norm.unsqueeze(-1))  # [B, D]
            seq = seq + t_emb.unsqueeze(1)  # broadcast across L

        decoded = self.encoder(seq)
        decoded = decoded.permute(0, 2, 1)
        out = self.linearize.reverse(decoded, shape)
        out = self.head(out)
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 10: INR-Mamba — Implicit Neural Representation
# ═══════════════════════════════════════════════════════════════════════


class _FourierPosEnc(nn.Module):
    """Fourier positional encoding for continuous coordinates."""

    def __init__(self, d_model: int, num_freqs: int = 32) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        # Output dim: 1 (raw) + 2*num_freqs (sin+cos) = 1 + 2*num_freqs
        self.out_dim = 1 + 2 * num_freqs
        # Cap max exponent at 10 to prevent gradient overflow in sin/cos backprop.
        # With num_freqs=32, linspace(0, 10, 32) gives max freq = 2^10 = 1024,
        # so max |grad| through sin is ~1024*pi ≈ 3217, safe for float32.
        max_exp = min(num_freqs - 1, 10)
        freqs = 2.0 ** torch.linspace(0, max_exp, num_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode scalar coordinates with Fourier features.

        Args:
            t: Coordinate tensor [...].

        Returns:
            Fourier features [..., 1 + 2*num_freqs].

        forward method for _FourierPosEnc.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        t_unsq = t.unsqueeze(-1)  # [..., 1]
        angles = t_unsq * self.freqs * math.pi  # [..., num_freqs]
        return torch.cat([t_unsq, torch.sin(angles), torch.cos(angles)], dim=-1)


@register_model(
    name="inr_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class INRMamba(nn.Module):
    """Implicit Neural Representation Mamba for continuous SR.

    Mamba extracts a latent contextual sequence, then an MLP decodes
    arbitrary continuous coordinates to intensities.  Allows any scaling
    factor (1.4×, 2×, 4×, etc.) at inference without retraining.

    Math:
        Z = Mamba(x) ∈ R^{L_U × d}
        For any coordinate p = (x,y,z):
            t_p = H(p) ∈ [0, 1]  (continuous Hilbert scalar)
            z_p = Interpolate(Z, t_p)
            Intensity(p) = MLP(PosEnc(t_p), z_p)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Mamba blocks for context extraction.
        num_freqs: Fourier positional encoding frequencies.
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        num_freqs: int = 32,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.context_encoder = _MambaEncoder(d_model, num_layers, d_state)

        self.pos_enc = _FourierPosEnc(d_model, num_freqs)
        pos_dim = self.pos_enc.out_dim

        # MLP: (z_p ∥ pos_enc(t_p)) → intensity
        self.inr_mlp = nn.Sequential(
            nn.Linear(d_model + pos_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: reconstruct at same resolution as input.

        For arbitrary-scale SR, use ``query()`` directly.

        forward method for INRMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]

        # Extract context
        context = self.context_encoder(seq)  # [B, L, D]

        # Query at same positions (identity mapping for training)
        B, L, D = context.shape
        t = torch.linspace(0, 1, L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_features = self.pos_enc(t)  # [B, L, pos_dim]

        combined = torch.cat([context, pos_features], dim=-1)  # [B, L, D+pos]
        intensities = self.inr_mlp(combined)  # [B, L, C_out]
        intensities = intensities.permute(0, 2, 1)  # [B, C_out, L]

        out = self.linearize.reverse(intensities, shape)

        # Residual: anchor untrained output near the input
        out_ch = out.shape[1]
        out = out + x[:, :out_ch]
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Combined Baseline: Hilbert-Mamba UNet
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="hilbert_mamba_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class HilbertMambaUNet(nn.Module):
    """Hilbert-Mamba U-Net — encoder-decoder with skip connections.

    A full U-Net architecture where each encoder/decoder level uses
    Hilbert-linearized Mamba blocks.  Combines the core ideas of fractal
    linearization (Method 2) with multi-directional averaging (Method 6)
    in a single practical architecture.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        features: Feature dimensions per encoder level.
        num_mamba_layers: Mamba blocks per level.
        linearization_mode: Primary linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.linearize = _HilbertLinearize(mode=linearization_mode)

        # Encoder
        self.enc_convs = nn.ModuleList()
        self.enc_mambas = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for feat in features:
            self.enc_convs.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, feat, 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, feat), feat),
                    nn.Conv2d(feat, feat, 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, feat), feat),
                )
            )
            self.enc_mambas.append(_MambaEncoder(feat, num_mamba_layers, d_state))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = feat

        # Bottleneck
        self.bottleneck_mamba = _MambaEncoder(features[-1], num_mamba_layers * 2, d_state)

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_mambas = nn.ModuleList()

        for i in range(len(features) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(features[i], features[i - 1], 2, stride=2))
            self.dec_convs.append(
                nn.Sequential(
                    nn.Conv2d(features[i - 1] * 2, features[i - 1], 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, features[i - 1]), features[i - 1]),
                    nn.Conv2d(features[i - 1], features[i - 1], 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, features[i - 1]), features[i - 1]),
                )
            )
            self.dec_mambas.append(_MambaEncoder(features[i - 1], num_mamba_layers, d_state))

        self.final_up = nn.ConvTranspose2d(features[0], features[0], 2, stride=2)
        self.head = _ConvHead(features[0], out_channels)

    def _mamba_pass(self, feat: torch.Tensor, mamba: _MambaEncoder) -> torch.Tensor:
        """Hilbert-linearize → Mamba → restore spatial."""
        seq, shape = self.linearize(feat)
        seq = seq.permute(0, 2, 1)
        processed = mamba(seq)
        processed = processed.permute(0, 2, 1)
        return self.linearize.reverse(processed, shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: U-Net with Hilbert-Mamba at every level.

        forward method for HilbertMambaUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        skips = []

        # Encoder
        h = x
        for conv, mamba, pool in zip(self.enc_convs, self.enc_mambas, self.pools, strict=False):
            h = conv(h)
            h = h + self._mamba_pass(h, mamba)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        h = h + self._mamba_pass(h, self.bottleneck_mamba)

        # Decoder
        for up, conv, mamba, skip in zip(
            self.ups,
            self.dec_convs,
            self.dec_mambas,
            reversed(skips[:-1]),
            strict=False,
        ):
            h = up(h)
            # Handle potential size mismatch from pooling
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = conv(h)
            h = h + self._mamba_pass(h, mamba)

        h = self.final_up(h)
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)

        out = self.head(h)

        # Residual: anchor untrained output near the input
        out_ch = out.shape[1]
        out = out + x[:, :out_ch]
        return _unsqueeze_to_volume(out, vol_shape)


__all__ = [
    "CRMMamba",
    "CTMamba",
    "FEMamba",
    "HLDMamba",
    "HWMamba",
    "HilbertMambaUNet",
    "INRMamba",
    "MDIMamba",
    "MMMamba",
    "TwoHalfDMamba",
]
