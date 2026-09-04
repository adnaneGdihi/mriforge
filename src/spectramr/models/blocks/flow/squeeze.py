"""Squeeze and split operators for multi-scale normalizing flows.

``Squeeze2x`` trades spatial resolution for channel depth: a
``[B, C, H, W]`` tensor becomes ``[B, 4C, H/2, W/2]`` (and the inverse
``unsqueeze`` recovers the original). ``Split`` factors a tensor into a
part that continues through the flow and a part that is emitted to the
Gaussian prior, which is the mechanism by which Glow models a
multi-scale latent.

Reference: D. P. Kingma and P. Dhariwal, "Glow: Generative flow with
invertible 1x1 convolutions," NeurIPS 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Squeeze2x(nn.Module):
    """Reshape spatial dims into channels (factor of 2 per axis)."""

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            return self._unsqueeze(x)
        return self._squeeze(x)

    @staticmethod
    def _squeeze(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Squeeze2x requires even H and W, got ({h}, {w}).")
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unsqueeze(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c % 4 != 0:
            raise ValueError(f"Squeeze2x inverse requires C divisible by 4, got {c}.")
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(b, c // 4, h * 2, w * 2)


class Split(nn.Module):
    """Split channels into (kept, emitted-to-prior) halves.

    Forward returns ``(x_kept, z_split)``; ``inverse`` re-concatenates a
    sampled ``z_split`` back onto the kept tensor.
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = x.shape[1]
        if c % 2 != 0:
            raise ValueError(f"Split requires an even channel count, got {c}.")
        x_kept, z_split = x.chunk(2, dim=1)
        return x_kept, z_split

    def inverse(self, x_kept: torch.Tensor, z_split: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_kept, z_split], dim=1)
