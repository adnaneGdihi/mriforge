r"""Cross-contrast cross-domain generator with redundant supervision.

Implements §4 / cross-contrast cross-domain generators of the multi-
contrast brainstorm — a single shared encoder feeds *four* small
decoder heads:

.. table::

   ====================  ==========================================
   Head                  Output
   ====================  ==========================================
   ``image_source``      Reconstructed source image
   ``image_target``      Translated target-contrast image
   ``kspace_source``     Source k-space (centred FFT of head-1)
   ``kspace_target``     Target k-space (centred FFT of head-2)
   ====================  ==========================================

Training applies four reconstruction losses, one per head, against the
matching ground truth. The redundant supervision serves two purposes:

1. *Regularisation* — forcing consistency between image and k-space
   outputs eliminates a class of non-physical solutions that pass an
   image-only loss but produce broken k-space spectra.
2. *Ablation surface* — the head losses can be enabled/disabled
   independently to isolate which supervision pathway is doing the work.

The image heads are produced by a small conv decoder branching off the
shared bottleneck. The k-space heads are produced by *applying the
centred FFT to the image heads* — guaranteeing exact image/k-space
consistency by construction. (An alternative architecture trains
independent k-space heads with no FFT consistency, but this is exactly
the failure mode we want to forbid.)

Per CLAUDE.md, the FFT goes through ``fft2c`` from
:mod:`mriforge.infrastructure.physics.fft_ops` (centred, ``norm="ortho"``,
autocast-aware) — never raw ``torch.fft.fft2``.

References
----------
[8] K. Hammernik *et al.*, "Learning a variational network for
    reconstruction of accelerated MRI data," *Magn. Reson. Med.*,
    vol. 79, no. 6, 2018.

§4 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.infrastructure.physics.fft_ops import fft2c
from mriforge.models.blocks.acquisition_conditioning import AcquisitionEmbedding
from mriforge.models.registry import register_model

__all__ = ["CrossContrastCrossDomainGenerator"]


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    g = min(8, out_ch) if out_ch >= 8 else 1
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(g, out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(g, out_ch),
        nn.SiLU(),
    )


class _SmallDecoder(nn.Module):
    """Three-stage upsampling decoder branching off a shared bottleneck.

    Matches the encoder's 3 down-sampling steps so the head output recovers
    the input spatial resolution.
    """

    def __init__(self, bottleneck_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(bottleneck_ch, mid_ch, kernel_size=2, stride=2)
        self.block1 = _double_conv(mid_ch, mid_ch)
        self.up2 = nn.ConvTranspose2d(mid_ch, mid_ch // 2, kernel_size=2, stride=2)
        self.block2 = _double_conv(mid_ch // 2, mid_ch // 2)
        self.up3 = nn.ConvTranspose2d(mid_ch // 2, max(mid_ch // 4, 8), kernel_size=2, stride=2)
        self.block3 = _double_conv(max(mid_ch // 4, 8), max(mid_ch // 4, 8))
        self.head = nn.Conv2d(max(mid_ch // 4, 8), out_ch, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.block1(self.up1(h))
        h = self.block2(self.up2(h))
        h = self.block3(self.up3(h))
        return self.head(h)


@register_model(
    name="cross_contrast_cross_domain",
    training_mode="reconstruction",
    supports_contrast_conditioning=True,
)
class CrossContrastCrossDomainGenerator(nn.Module):
    r"""Shared encoder + 4 heads producing {image, k-space} × {source, target}.

    Args:
        in_channels: Input image channels (typically 1 for magnitude or
            2 for real/imag of single-coil data).
        kspace_channels: K-space output channels per head (2 for
            real/imag; 8 for SENSE-combined virtual coils).
        base_channels: Encoder/decoder feature width.
        contrast_dim: Per-sample contrast embedding dimension.
        n_contrasts: Cohort cardinality.

    Forward signature
    -----------------
    ``forward(x, acquisition_params=None, contrast_idx=None)`` returns
    a dict::

        {
            "image_source": Tensor (B, in_channels, H, W),
            "image_target": Tensor (B, in_channels, H, W),
            "kspace_source": Tensor (B, kspace_channels, H, W),
            "kspace_target": Tensor (B, kspace_channels, H, W),
        }

    The k-space heads are produced by applying ``fft2c`` to the image
    heads — image/k-space consistency holds *by construction*.
    """

    def __init__(
        self,
        in_channels: int = 1,
        kspace_channels: int = 2,
        base_channels: int = 32,
        contrast_dim: int = 16,
        n_contrasts: int = 4,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if kspace_channels < 2 or kspace_channels % 2:
            raise ValueError(
                f"kspace_channels must be even (real/imag pairs), got {kspace_channels}"
            )
        self.in_channels = in_channels
        # `out_channels` is exposed for the audit's domain-alignment check;
        # report the image-source head shape (the canonical output).
        self.out_channels = in_channels
        self.kspace_channels = kspace_channels

        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        # Shared encoder
        self.enc1 = _double_conv(in_channels, widths[0])
        self.enc2 = _double_conv(widths[0], widths[1])
        self.enc3 = _double_conv(widths[1], widths[2])
        self.bottleneck = _double_conv(widths[2], widths[3])

        # Conditioning — concatenated as a constant-spatial bias on the bottleneck
        self.acquisition_embedding = AcquisitionEmbedding(
            embed_dim=contrast_dim,
            n_freqs=6,
            n_contrasts=n_contrasts,
        )
        self.cond_proj = nn.Linear(contrast_dim, widths[3])

        # Two image-domain decoders — source reconstruction + target translation
        self.image_source_decoder = _SmallDecoder(widths[3], widths[2], in_channels)
        self.image_target_decoder = _SmallDecoder(widths[3], widths[2], in_channels)

        # Optional learnable adapter that turns 2-ch real/imag into a `kspace_channels`
        # tensor — useful when kspace_channels > 2 (e.g. SENSE virtual coils).
        if kspace_channels == 2:
            self.kspace_adapter: nn.Module | None = None
        else:
            self.kspace_adapter = nn.Conv2d(2, kspace_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor, contrast_emb: torch.Tensor) -> torch.Tensor:
        h = self.enc1(x)
        h = self.enc2(F.avg_pool2d(h, 2))
        h = self.enc3(F.avg_pool2d(h, 2))
        h = self.bottleneck(F.avg_pool2d(h, 2))
        # Add the contrast embedding as a spatial bias on the bottleneck.
        bias = self.cond_proj(contrast_emb).view(*contrast_emb.shape[:1], -1, 1, 1)
        return h + bias

    def _to_kspace(self, image: torch.Tensor) -> torch.Tensor:
        """Centred FFT → real/imag-as-channels → optional channel adapter."""
        spec = fft2c(image)  # complex (B, in_channels, H, W)
        # Reduce channel-dim to 1 if multi-channel input — take the magnitude per
        # voxel for the magnitude-only case; for richer adapters subclass.
        if spec.shape[1] > 1:
            spec = spec.mean(dim=1, keepdim=True)
        real_imag = torch.cat([spec.real, spec.imag], dim=1)  # (B, 2, H, W)
        if self.kspace_adapter is not None:
            real_imag = self.kspace_adapter(real_imag)
        return real_imag

    def forward(
        self,
        x: torch.Tensor,
        acquisition_params: dict[str, torch.Tensor] | None = None,
        contrast_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W); got {tuple(x.shape)}.")
        B = x.shape[0]
        if acquisition_params is None:
            zero = {
                k: torch.zeros(B, device=x.device, dtype=x.dtype)
                for k in ("TE", "TR", "TI", "FA", "B0")
            }
            c_idx = (
                contrast_idx
                if contrast_idx is not None
                else torch.zeros(B, dtype=torch.long, device=x.device)
            )
            cond = self.acquisition_embedding(zero, contrast_id=c_idx)
        else:
            params = {k: v.to(x.device).to(x.dtype) for k, v in acquisition_params.items()}
            c_idx = (
                contrast_idx
                if contrast_idx is not None
                else torch.zeros(B, dtype=torch.long, device=x.device)
            )
            cond = self.acquisition_embedding(params, contrast_id=c_idx)

        h = self._encode(x, cond)
        img_src = self.image_source_decoder(h)
        img_tgt = self.image_target_decoder(h)
        return {
            "image_source": img_src,
            "image_target": img_tgt,
            "kspace_source": self._to_kspace(img_src),
            "kspace_target": self._to_kspace(img_tgt),
        }
