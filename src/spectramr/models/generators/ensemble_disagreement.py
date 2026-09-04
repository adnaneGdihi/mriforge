"""Ensemble Disagreement Generator for uncertainty-aware MRI reconstruction.

Multi-headed ensemble where each head produces an independent reconstruction.
Disagreement between heads quantifies epistemic uncertainty, enabling active
learning and out-of-distribution detection.

Reference:
    Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles", NeurIPS 2017.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _EnsembleMember(nn.Module):
    """Single ensemble member (lightweight reconstruction head)."""

    def __init__(self, feature_dim: int, out_ch: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim // 2), feature_dim // 2),
            nn.SiLU(),
            nn.Conv2d(feature_dim // 2, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _EnsembleMember.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.head(x)


@register_model(name="ensemble_disagreement", training_mode="reconstruction")
class EnsembleDisagreementGenerator(IGenerator, nn.Module):
    """Multi-headed ensemble with disagreement-based uncertainty."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        feature_dim: int = kwargs.get("feature_dim", 64)
        num_layers: int = kwargs.get("num_layers", 8)
        num_members: int = kwargs.get("num_ensemble_members", 5)
        self.disagreement_weight: float = kwargs.get("disagreement_weight", 0.5)

        # Shared backbone
        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                    nn.GroupNorm(min(32, feature_dim), feature_dim),
                    nn.SiLU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)

        # Independent ensemble heads
        self.members = nn.ModuleList(
            [_EnsembleMember(feature_dim, out_channels) for _ in range(num_members)]
        )

    @property
    def name(self) -> str:
        return "ensemble_disagreement"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for EnsembleDisagreementGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        shared = self.backbone(self.intro(x))
        predictions = [member(shared) for member in self.members]
        # Mean prediction
        mean_pred = torch.stack(predictions).mean(dim=0)
        return mean_pred

    def forward_with_uncertainty(
        self, x: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean prediction and pixel-wise uncertainty (std)."""
        shared = self.backbone(self.intro(x))
        predictions = torch.stack([m(shared) for m in self.members])  # [M, B, C, H, W]
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        return mean_pred, uncertainty

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
