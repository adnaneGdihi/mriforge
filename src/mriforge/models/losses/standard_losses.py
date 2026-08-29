"""Standard PyTorch losses wrapped for registry.

This module provides registry-compatible wrappers for standard PyTorch losses.
Metrics can be optionally computed during loss calculation.
"""

import torch
import torch.nn as nn

from mriforge.models.losses.metrics_aware_loss import MetricsAwareLossMixin
from mriforge.models.losses.registry import register_loss
from mriforge.models.losses.validation import validate_loss_inputs


@register_loss(name="l1", aliases=["mae", "mean_absolute_error"], domain="agnostic")
class L1Loss(MetricsAwareLossMixin, nn.L1Loss):
    """L1 Loss (Mean Absolute Error) with optional metrics.

    **DOMAIN**: AGNOSTIC (works on any tensor)
    **Input**: [B, C, H, W] or any tensor shape
    **3D Support**: Yes

    Standard MAE. Domain-agnostic - works on k-space, image, latent, etc.
    """

    def __init__(self, reduction: str = "mean", compute_metrics: bool = False):
        """__init__.

        Args:
            reduction (str): Description.
            compute_metrics (bool): Description.
        """
        super().__init__(reduction=reduction)
        self.compute_metrics_flag = compute_metrics

    @validate_loss_inputs
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: Description.
        """
        if pred.is_complex() or target.is_complex():
            diff = pred - target
            diff_mag = torch.sqrt(diff.real.square() + diff.imag.square() + 1e-12)
            if self.reduction == "mean":
                loss_value = diff_mag.mean()
            elif self.reduction == "sum":
                loss_value = diff_mag.sum()
            else:
                loss_value = diff_mag
        else:
            loss_value = nn.functional.l1_loss(pred, target, reduction=self.reduction)

        if self.compute_metrics_flag:
            metrics = self.compute_metrics(pred, target, loss_value)
            return loss_value, metrics
        return loss_value

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        **kwargs,
    ) -> dict[str, float]:
        """Compute L1-specific metrics."""
        diff = pred - target
        errors = (
            torch.sqrt(diff.real.square() + diff.imag.square() + 1e-12)
            if diff.is_complex()
            else diff.abs()
        )
        if pred.is_complex():
            pred = torch.sqrt(pred.real.square() + pred.imag.square() + 1e-12)
        if target.is_complex():
            target = torch.sqrt(target.real.square() + target.imag.square() + 1e-12)

        return {
            "loss": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "error_mean": errors.mean().item(),
            "error_std": errors.std().item(),
            "error_min": errors.min().item(),
            "error_max": errors.max().item(),
            "error_median": errors.median().item(),
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "target_mean": target.mean().item(),
            "target_std": target.std().item(),
            "pred_range": (pred.max().item() - pred.min().item()),
            "target_range": (target.max().item() - target.min().item()),
        }


@register_loss(name="l2", aliases=["mse", "mean_squared_error"], domain="agnostic")
class MSELoss(MetricsAwareLossMixin, nn.MSELoss):
    """L2 Loss (Mean Squared Error) with optional metrics.

    **DOMAIN**: AGNOSTIC (works on any tensor)
    **Input**: [B, C, H, W] or any tensor shape
    **3D Support**: Yes

    Standard MSE. Domain-agnostic - works on k-space, image, latent, etc.
    """

    def __init__(self, reduction: str = "mean", compute_metrics: bool = False):
        """__init__.

        Args:
            reduction (str): Description.
            compute_metrics (bool): Description.
        """
        super().__init__(reduction=reduction)
        self.compute_metrics_flag = compute_metrics

    @validate_loss_inputs
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: Description.
        """
        if pred.is_complex() or target.is_complex():
            diff = pred - target
            squared_err = diff.real.square() + diff.imag.square()
            if self.reduction == "mean":
                loss_value = squared_err.mean()
            elif self.reduction == "sum":
                loss_value = squared_err.sum()
            else:
                loss_value = squared_err
        else:
            loss_value = nn.functional.mse_loss(pred, target, reduction=self.reduction)

        if self.compute_metrics_flag:
            metrics = self.compute_metrics(pred, target, loss_value)
            return loss_value, metrics
        return loss_value

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        **kwargs,
    ) -> dict[str, float]:
        """Compute MSE-specific metrics."""
        diff = pred - target
        errors = diff.real.square() + diff.imag.square() if diff.is_complex() else diff.square()

        if pred.is_complex():
            pred = torch.sqrt(pred.real.square() + pred.imag.square() + 1e-12)
        if target.is_complex():
            target = torch.sqrt(target.real.square() + target.imag.square() + 1e-12)

        return {
            "loss": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "mse": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "rmse": (
                (loss_value.item() ** 0.5)
                if loss_value.numel() == 1
                else (loss_value.mean().item() ** 0.5)
            ),
            "error_squared_mean": errors.mean().item(),
            "error_squared_std": errors.std().item(),
            "error_squared_max": errors.max().item(),
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "target_mean": target.mean().item(),
            "target_std": target.std().item(),
            "pred_range": (pred.max().item() - pred.min().item()),
            "target_range": (target.max().item() - target.min().item()),
        }


# `huber` alias intentionally dropped — `huber` is the canonical name of the
# diffusion-Huber loss in `diffusion_losses.py:78`. The standard `smooth_l1`
# handle remains; if you specifically want PyTorch's `nn.SmoothL1Loss`
# semantics, use `smooth_l1` (or the legacy alias `diffusion_huber`). See
# TODO/audit/12_losses.md F3.
@register_loss(name="smooth_l1", aliases=["diffusion_huber"], domain="agnostic")
class SmoothL1Loss(nn.SmoothL1Loss):
    """Smooth L1 Loss (Huber Loss).
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SmoothL1} = \text{SmoothL1}(|\\hat{y} - y|)"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.is_complex() or target.is_complex():
            diff = pred - target
            diff_mag = torch.sqrt(diff.real.square() + diff.imag.square() + 1e-12)
            beta = getattr(self, "beta", 1.0)
            return nn.functional.smooth_l1_loss(
                diff_mag,
                torch.zeros_like(diff_mag),
                reduction=self.reduction,
                beta=beta,
            )
        return super().forward(pred, target)


@register_loss(
    name="bce", aliases=["binary_crossentropy", "binary_cross_entropy"], domain="agnostic"
)
class BCELoss(nn.BCELoss):
    """Binary Cross Entropy Loss."""

    pass


@register_loss(name="bce_with_logits", aliases=["bce_logits", "bcewithlogits"], domain="agnostic")
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Binary Cross Entropy with Logits Loss."""

    pass


@register_loss(name="cross_entropy", aliases=["crossentropy", "ce"], domain="agnostic")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss."""

    pass


@register_loss(name="tv", aliases=["total_variation"])
class TotalVariationLoss(nn.Module):
    """Total Variation Loss."""

    def __init__(self, reduction: str = "mean"):
        """__init__.

        Args:
            reduction (str): Description.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """forward.

        Args:
            pred (torch.Tensor): Prediction tensor
            target (torch.Tensor, optional): Ignored. Target image.
        Returns:
            torch.Tensor: Total variation loss penalty.

        forward method for TotalVariationLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        x = pred
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        diff_h = x[:, :, 1:, :] - x[:, :, : h_x - 1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, : w_x - 1]

        if x.is_complex():
            h_tv = (diff_h.real.square() + diff_h.imag.square()).sum()
            w_tv = (diff_w.real.square() + diff_w.imag.square()).sum()
        else:
            h_tv = torch.pow(diff_h, 2).sum()
            w_tv = torch.pow(diff_w, 2).sum()

        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        """_tensor_size.

        Args:
            t (Any): Description.
        Returns:
            Any: Description.
        """
        return t.size(1) * t.size(2) * t.size(3)
