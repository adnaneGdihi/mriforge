"""Dual Domain Processing Blocks.

Processes features in both K-space (Frequency) and Image-space (Spatial) domains.

Feature-domain contract (2026-07-03): the block no longer hard-assumes its
input is k-space. The required ``feature_domain`` kwarg ("kspace" | "image",
derived from ``force_pure_kspace`` by ``FourierBridgeNetwork``) dictates how
the two views are derived at entry and conjugates the output back so that
**output domain == input domain**. ``feature_domain="kspace"`` is numerically
identical to the historical behavior.
"""

import logging

import torch
from torch import nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved,
    interleaved_to_complex,
    validate_feature_domain,
)
from spectramr.models.blocks.normalization import ComplexAdaptiveGroupNorm
from spectramr.models.layers.complex_conv import ComplexConv2d, ComplexConv2dReflectPad
from spectramr.models.layers.complex_layers import ComplexBatchNorm2d

logger = logging.getLogger(__name__)


class DualDomainBlock(nn.Module):
    """
    Processes features in both K-space (Frequency) and Image-space (Spatial).

    Path A: K-space Convolution (Local Frequency consistency)
    Path B: Image-space Convolution (Local Spatial consistency)
    """

    def __init__(
        self,
        channels: int,
        activation: nn.Module | None = None,
        use_complex_conv: bool = False,
        time_embedding_dim: int | None = None,
        *,
        feature_domain: str,
    ):
        """__init__.

        Args:
            channels (int): Description.
            activation (nn.Module | None): Description.
            use_complex_conv (bool): Description.
            time_embedding_dim (int | None): Description.
            feature_domain (str): Domain of the input feature map,
                ``"kspace"`` or ``"image"``. Derived from
                ``force_pure_kspace`` by the caller — raises on anything else.
        """
        super().__init__()
        self.use_complex_conv = use_complex_conv
        self.channels = channels
        self.feature_domain = validate_feature_domain(feature_domain)

        # Default activation if none provided
        self.activation = activation if activation is not None else nn.ReLU(inplace=True)

        C = channels // 2 if channels % 2 == 0 else channels
        num_groups = C  # Instance Norm behavior for complex

        if use_complex_conv:
            # K-space path
            self.k_conv_1 = ComplexConv2d(C, C, 3, padding=1, padding_mode="circular", bias=False)
            self.k_conv_2 = ComplexConv2d(C, C, 3, padding=1, padding_mode="circular", bias=False)

            # [STABILIZATION] No normalization in k-space path
            self.k_gn = nn.Identity()

            # Image-space path
            self.i_conv_1 = ComplexConv2dReflectPad(C, C, 3, padding=1, bias=False)
            self.i_conv_2 = ComplexConv2dReflectPad(C, C, 3, padding=1, bias=False)

            # Image-space normalization is beneficial for stability
            if time_embedding_dim is not None:
                self.i_gn = ComplexAdaptiveGroupNorm(channels, num_groups, time_embedding_dim)
            else:
                self.i_gn = ComplexBatchNorm2d(C)

            # FUSION PROJECTION: Project 2*C complex channels back to C complex channels
            # k_feat (C complex) + i_feat (C complex) -> cat (2*C complex) -> project (C complex)
            # bias=False: k-space constant term creates spatial Dirac delta singularity
            self.fusion_proj = ComplexConv2d(channels, C, 1, bias=False)
        else:
            # Standard real convolutions path
            self.k_conv_1 = nn.Conv2d(
                channels, channels, 3, padding=1, padding_mode="circular", bias=False
            )
            self.k_conv_2 = nn.Conv2d(
                channels, channels, 3, padding=1, padding_mode="circular", bias=False
            )
            self.k_gn = nn.Identity()

            self.i_conv_1 = nn.Conv2d(
                channels, channels, 3, padding=1, padding_mode="reflect", bias=False
            )
            self.i_conv_2 = nn.Conv2d(
                channels, channels, 3, padding=1, padding_mode="reflect", bias=False
            )
            self.i_gn = (
                nn.GroupNorm(num_groups, channels) if time_embedding_dim is None else nn.Identity()
            )

            self.fusion_proj = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        # x is Interleaved [R1, I1, R2, I2, ...] in ``self.feature_domain``
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DualDomainBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor, same domain as the input.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Derive the two views from the declared input domain. Under "kspace"
        # this is numerically identical to the historical path (x IS k-space).
        h = interleaved_to_complex(x)
        if self.feature_domain == "kspace":
            k_tensor = x
            img_tensor = complex_to_interleaved(ifft2c(h))
        else:
            k_tensor = complex_to_interleaved(fft2c(h))
            img_tensor = x

        # 1. K-space path (k view)
        k = self.k_conv_1(k_tensor)
        k = self.k_gn(k)
        k = self.activation(k)
        k_feat = self.k_conv_2(k)

        # 2. Image-space path (image view)
        i = self.i_conv_1(img_tensor)
        if isinstance(self.i_gn, ComplexAdaptiveGroupNorm):
            i = self.i_gn(i, t_emb)
        else:
            i = self.i_gn(i)
        i = self.activation(i)
        i_feat_spatial = self.i_conv_2(i)

        # Transform the image branch back to the k frame for fusion. The
        # fusion MUST stay in the k frame in both modes: the real-conv
        # ``fusion_proj`` variant is R-linear but not C-linear, so it does not
        # commute with the FFT — fusing in the image frame would change the
        # learned function class, not just the representation.
        i_feat = complex_to_interleaved(fft2c(interleaved_to_complex(i_feat_spatial)))

        # 3. Fusion and Projection (k frame)
        combined = torch.cat([k_feat, i_feat], dim=1)
        fused_k = self.fusion_proj(combined)

        # Conjugate back so output domain == input domain
        if self.feature_domain == "kspace":
            return fused_k
        return complex_to_interleaved(ifft2c(interleaved_to_complex(fused_k)))
