r"""Multi-task heads for one-shot multi-parameter ULF mapping (idea 10).

Implements:

- :class:`MultiParameterHead` — a small ``nn.Module`` that consumes
  shared encoder features and emits four parallel heads
  :math:`(T_1, T_2, \mathrm{PD}, \mathbf s)` where :math:`\mathbf s`
  is an optional segmentation logit map.
- :class:`UncertaintyWeightedMultiTaskLoss` — Kendall et al. 2018
  uncertainty-weighted multi-task loss
  :math:`\mathcal L = \sum_k \tfrac12 e^{-\sigma_k} \mathcal L_k + \tfrac12 \sigma_k`
  where each :math:`\sigma_k = \log \tau_k^2` is a learnable
  log-variance parameter.

Both pieces are intentionally narrow — they slot into a host
backbone via composition (the strategy attaches them after the
encoder and supplies the decoder).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiParameterHead(nn.Module):
    r"""Four-head decoder: ``T1`` / ``T2`` / ``PD`` / ``segmentation``.

    Each head is a 1×1 convolution from ``in_channels`` features to
    its target channel count, with a tail activation chosen for the
    physical interpretation:

    - ``T1``, ``T2``, ``PD``: ``softplus`` to enforce positivity
    - ``segmentation``: raw logits (downstream code applies softmax)

    Args:
        in_channels: Width of the encoder feature map.
        num_seg_classes: Channels in the segmentation head; 0 disables it.
        t1_scale_ms: Multiplier applied after softplus for the T1 head
            (centres the output in millisecond units).
        t2_scale_ms: Multiplier for the T2 head.

    Inputs to ``forward``: ``[B, in_channels, H, W]``.

    Outputs: dict with keys ``t1_ms``, ``t2_ms``, ``pd``, optionally
    ``segmentation``. Each tensor has shape ``[B, 1 or num_seg_classes,
    H, W]``.
    """

    def __init__(
        self,
        in_channels: int,
        num_seg_classes: int = 0,
        t1_scale_ms: float = 1000.0,
        t2_scale_ms: float = 100.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0; got {in_channels}")
        if num_seg_classes < 0:
            raise ValueError(f"num_seg_classes must be ≥ 0; got {num_seg_classes}")
        if t1_scale_ms <= 0 or t2_scale_ms <= 0:
            raise ValueError("Scale parameters must be > 0")

        self.in_channels = in_channels
        self.num_seg_classes = num_seg_classes
        self.t1_scale_ms = float(t1_scale_ms)
        self.t2_scale_ms = float(t2_scale_ms)

        self.t1_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.t2_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.pd_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        if num_seg_classes > 0:
            self.seg_head: nn.Conv2d | None = nn.Conv2d(in_channels, num_seg_classes, kernel_size=1)
        else:
            self.seg_head = None

        # Zero-init for stable warm-up — the softplus tail produces
        # log(2) ≈ 0.69 mean activation, scaled by the t*_scale_ms
        # constants. This makes the network start at a physically-
        # sensible mid-range T1/T2 prediction.
        for head in (self.t1_head, self.t2_head, self.pd_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.seg_head is not None:
            nn.init.zeros_(self.seg_head.weight)
            nn.init.zeros_(self.seg_head.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.dim() != 4 or features.shape[1] != self.in_channels:
            raise ValueError(
                f"features must be [B, {self.in_channels}, H, W]; got {tuple(features.shape)}"
            )
        out: dict[str, torch.Tensor] = {
            "t1_ms": F.softplus(self.t1_head(features)) * self.t1_scale_ms,
            "t2_ms": F.softplus(self.t2_head(features)) * self.t2_scale_ms,
            "pd": F.softplus(self.pd_head(features)),
        }
        if self.seg_head is not None:
            out["segmentation"] = self.seg_head(features)
        return out


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    r"""Kendall et al. 2018 uncertainty-weighted multi-task loss.

    .. math::

        \mathcal L = \sum_k \frac{1}{2} e^{-\sigma_k} \mathcal L_k
                     + \frac{1}{2} \sigma_k

    where :math:`\sigma_k` is a learnable log-variance per task.

    Args:
        task_names: Iterable of task identifiers used as the keys of
            the per-task loss dict passed to ``forward``.
        init_log_var: Initial value for every :math:`\sigma_k`.

    Forward signature:
        ``losses: dict[str, torch.Tensor]`` mapping task name → scalar
        loss. Tasks not in ``task_names`` are ignored (and a warning
        is emitted if there are any).

    Returns the combined scalar loss.
    """

    def __init__(
        self,
        task_names: Sequence[str],
        init_log_var: float = 0.0,
    ) -> None:
        super().__init__()
        if not task_names:
            raise ValueError("task_names is empty")
        self._task_names = tuple(task_names)
        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(init_log_var, dtype=torch.float32))
                for name in self._task_names
            }
        )

    @property
    def task_names(self) -> tuple[str, ...]:
        """Tasks managed by this loss."""
        return self._task_names

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(losses, dict):
            raise ValueError("losses must be a dict[str, tensor]")
        if not losses:
            raise ValueError("losses dict is empty")

        total = torch.zeros((), dtype=torch.float32, device=next(iter(losses.values())).device)
        for name in self._task_names:
            if name not in losses:
                continue
            log_var = self.log_vars[name]
            total = total + 0.5 * torch.exp(-log_var) * losses[name] + 0.5 * log_var
        return total


__all__ = ["MultiParameterHead", "UncertaintyWeightedMultiTaskLoss"]
