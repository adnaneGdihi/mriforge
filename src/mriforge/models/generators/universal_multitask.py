"""Universal Multitask Generator for multi-domain MRI reconstruction.

Shared backbone with task-specific heads for simultaneous reconstruction,
super-resolution, denoising, and segmentation.

Reference:
    Vandenhende et al., "Multi-Task Learning for Dense Prediction Tasks:
    A Survey", IEEE TPAMI, 2021.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _TaskHead(nn.Module):
    """Lightweight task-specific decoder head."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.GroupNorm(min(32, in_ch // 2), in_ch // 2),
            nn.SiLU(),
            nn.Conv2d(in_ch // 2, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _TaskHead.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.head(x)


@register_model(name="universal_multitask", training_mode="reconstruction")
class UniversalMultitaskGenerator(IGenerator, nn.Module):
    """Multi-task reconstruction network with shared backbone and task heads."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        backbone_dim: int = kwargs.get("backbone_dim", 128)
        num_layers: int = kwargs.get("num_layers", 12)
        shared_blocks: int = kwargs.get("shared_blocks", 8)
        num_task_heads: int = kwargs.get("num_task_heads", 8)

        # Shared backbone
        self.intro = nn.Conv2d(in_channels, backbone_dim, 3, padding=1)
        shared: list[nn.Module] = []
        for _ in range(shared_blocks):
            shared.extend(
                [
                    nn.Conv2d(backbone_dim, backbone_dim, 3, padding=1),
                    nn.GroupNorm(min(32, backbone_dim), backbone_dim),
                    nn.SiLU(),
                ]
            )
        self.shared = nn.Sequential(*shared)

        # Task heads (all produce same out_channels for aggregation)
        self.task_heads = nn.ModuleList(
            [_TaskHead(backbone_dim, out_channels) for _ in range(num_task_heads)]
        )

        # Adaptive task weighting
        self.task_weights = nn.Parameter(torch.ones(num_task_heads) / num_task_heads)

    @property
    def name(self) -> str:
        return "universal_multitask"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for UniversalMultitaskGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        shared_feat = self.shared(self.intro(x))
        # Weighted average of all task heads
        weights = torch.softmax(self.task_weights, dim=0)
        outputs = [head(shared_feat) for head in self.task_heads]
        result = sum(w * o for w, o in zip(weights, outputs, strict=False))
        return result

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@register_model(
    name="universal_multitask_dual",
    training_mode="reconstruction",
    requires_paired_data=True,
)
class UniversalMultitaskDualContrastGenerator(UniversalMultitaskGenerator):
    """Pattern B — channel-stacked dual-contrast fusion variant.

    Declares the dual-contrast contract at the registry layer: ``in_channels``
    and ``out_channels`` must both be even (interpreted as ``2 × N_pair``
    real-stacked complex per contrast) and the two contrasts are concatenated
    along the channel axis. The forward pass is unchanged from the base
    multitask class — convolutions are channel-agnostic — but at init time
    we validate the layout and at forward time we shape-check, so the
    audit's ``model_loss_output_domain`` and ``domain_alignment`` checks
    cannot silently see a single-contrast tensor masquerading as a fusion
    input.

    Used by ``experiment_130_universal_multitask`` to mutually fill both
    T1 and T2 / FLAIR k-spaces from a paired accelerated input
    ``[T1_R, T1_I, T2_R, T2_I]`` (4-channel) — see m4raw cross-contrast
    loader output when ``num_virtual_coils: 1`` and ``single_contrast: false``.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        **kwargs: Any,
    ) -> None:
        if in_channels % 2 != 0:
            raise ValueError(
                f"universal_multitask_dual requires an even in_channels "
                f"(2 × per-contrast channel count); got {in_channels}."
            )
        if out_channels % 2 != 0:
            raise ValueError(
                f"universal_multitask_dual requires an even out_channels "
                f"(must mirror in_channels for mutual completion); "
                f"got {out_channels}."
            )
        if in_channels != out_channels:
            raise ValueError(
                f"universal_multitask_dual: in_channels ({in_channels}) and "
                f"out_channels ({out_channels}) must match — both contrasts "
                f"are reconstructed end-to-end."
            )
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self._channels_per_contrast = in_channels // 2

    @property
    def name(self) -> str:
        return "universal_multitask_dual"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"universal_multitask_dual expects input of shape "
                f"[B, {self.in_channels}, H, W] (paired contrasts); "
                f"got {tuple(x.shape)}."
            )
        return super().forward(x, **kwargs)
