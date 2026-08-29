"""Lightweight image encoder used by LGDR.

The Latent Geodesic Discrepancy ranker needs *some* embedding into a fixed
``d``-dimensional vector space so it can build a manifold graph. In practice
users will swap this out for DINOv2 / RadImageNet / a project-specific MAE,
but having a self-contained baseline that runs on CPU keeps the framework
testable and the smoke tests fast.

The default ``SimpleAEEncoder`` is a 4-stage strided-conv encoder followed
by global-avg-pool. It is **untrained** by default — for meaningful manifold
geometry the user is expected to either:

* call ``fit_random_features`` to install random-feature weights (which give
  surprisingly informative distances on natural images, akin to RBF random
  features),
* or replace the encoder via ``MetaEvaluationPipeline(encoder=...)``.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleAEEncoder(nn.Module):
    """4-stage strided-conv encoder with GAP + linear head.

    The output dimension is ``embed_dim`` (default 64). Designed to run on
    CPU at modest cost — typical batch of 256 slices encodes in <1 s.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.head = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            x = x.abs()
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        # Reduce to 1-channel magnitude for the encoder if multi-coil.
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        feats = self.features(x.float())
        gap = feats.mean(dim=(-2, -1))
        return self.head(gap)


def fit_random_features(encoder: SimpleAEEncoder, seed: int = 0) -> SimpleAEEncoder:
    """Install scaled-Gaussian random weights.

    Random conv features yield non-trivial distances even without training —
    enough for the LGDR diffusion-distance machinery to be informative on
    synthetic data and to pass the unit tests. For production use, swap to
    a pretrained encoder.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in encoder.parameters():
            if p.ndim >= 2:
                fan = float(p[0].numel())
                p.copy_(torch.randn(p.shape, generator=gen) / max(fan, 1) ** 0.5)
            else:
                p.zero_()
    encoder.eval()
    return encoder


def embed_batch(
    encoder: nn.Module,
    images: list[torch.Tensor],
    device: torch.device | str = "cpu",
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode a list of (variable-shape) images into [N, d]. Pads to the max."""
    encoder.eval()
    if not images:
        return torch.zeros((0, getattr(encoder, "embed_dim", 1)))
    # Find a common shape via center crop / pad.
    h = max(int(im.shape[-2]) for im in images)
    w = max(int(im.shape[-1]) for im in images)
    # Round to multiples of 16 so the 4-stride encoder doesn't crash.
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            tensors = []
            for im in chunk:
                if torch.is_complex(im):
                    im = im.abs()
                if im.ndim == 2:
                    im = im.unsqueeze(0)
                if im.ndim == 3 and im.shape[0] != 1:
                    im = im.mean(dim=0, keepdim=True)
                if im.ndim == 4:
                    im = im.mean(dim=0).mean(dim=0, keepdim=True)
                pad_h = h - im.shape[-2]
                pad_w = w - im.shape[-1]
                im = nn.functional.pad(im, (0, max(pad_w, 0), 0, max(pad_h, 0)))
                tensors.append(im[..., :h, :w].float())
            batch = torch.stack(tensors, dim=0).to(device)
            out_chunks.append(encoder(batch).cpu())
    return torch.cat(out_chunks, dim=0)
