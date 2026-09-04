r"""Self-Supervised learning via Data Undersampling (SSDU) and the
multi-mask robust variant.

SSDU (Yaman *et al.* 2020) splits the acquired k-space sample
:math:`\mathbf{y}` into a network input partition
:math:`\mathbf{y}_\Lambda` (mask :math:`\Lambda`) and a held-out target
partition :math:`\mathbf{y}_\Theta` (mask :math:`\Theta`,
:math:`\Theta\cap\Lambda=\emptyset`). The network is trained without
ground truth via

.. math::
    \mathcal{L}_{\text{SSDU}} = \big\|\,\mathbf{M}_\Theta\mathbf{F}\mathbf{S}\,
        f_{\boldsymbol{\theta}}(\mathbf{y}_\Lambda) - \mathbf{y}_\Theta\,\big\|_2^2.

The loss therefore needs the *target* k-space mask :math:`\Theta` to be
supplied via the loss ``context`` dict. ``prediction`` here is taken to
be the network output already in image space (the loss applies the
forward operator internally).

The "multi-mask" variant (Yaman *et al.* 2022) averages the SSDU loss
over many random :math:`(\Lambda,\Theta)` splits per example, which
makes the supervision much closer to fully-sampled at high acceleration
factors.

References:
    [1] B. Yaman *et al.*, "Self-supervised learning of physics-guided
        reconstruction neural networks without fully sampled reference
        data," *Magn. Reson. Med.*, 84(6), 2020.
    [2] B. Yaman *et al.*, "Multi-mask self-supervised learning for
        physics-guided neural networks in highly accelerated MRI,"
        *NMR Biomed.*, 35(12), 2022.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.registry import register_loss


def _coerce_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a sampling mask to the k-space tensor's shape.

    Mask is expected as a real ``(H, W)``, ``(1, 1, H, W)``, or
    ``(B, 1, H, W)`` boolean / float tensor.
    """
    while mask.dim() < like.dim():
        mask = mask.unsqueeze(0)
    return mask.to(like.real.dtype if like.is_complex() else like.dtype)


def _kspace_residual(
    pred_image: torch.Tensor, target_kspace: torch.Tensor, theta_mask: torch.Tensor
) -> torch.Tensor:
    """Compute :math:`\\|\\mathbf{M}_\\Theta(F\\hat{x} - y)\\|_2^2`.

    pred_image: real-valued :math:`(B, C, H, W)` or complex :math:`(B, 1, H, W)`.
    target_kspace: complex :math:`(B, C, H, W)`.
    theta_mask: real, broadcastable to ``target_kspace``.
    """
    pred_k = fft2c(pred_image)
    if pred_k.shape != target_kspace.shape:
        raise ValueError(
            f"pred k-space shape {tuple(pred_k.shape)} != target k-space "
            f"shape {tuple(target_kspace.shape)}"
        )
    diff = pred_k - target_kspace
    masked = diff * _coerce_mask(theta_mask, diff)
    # Treat real & imaginary channels equally
    if masked.is_complex():
        masked = torch.view_as_real(masked)
    return masked.pow(2).mean()


@register_loss(name="ssdu", aliases=["SSDULoss"], domain="kspace")
class SSDULoss(nn.Module):
    r"""Single-split SSDU loss.

    Forward expects:

    * ``prediction``: image-space output of the reconstructor on the
      :math:`\Lambda`-undersampled k-space.
    * ``target``: ignored (kept for registry signature).
    * ``context["target_kspace"]``: the *original* (multi-coil) k-space
      :math:`\mathbf{y}` before the :math:`\Theta`-split was applied.
    * ``context["theta_mask"]``: the held-out mask :math:`\Theta`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError(
                "SSDULoss requires context['target_kspace'] and context['theta_mask']."
            )
        target_kspace = context.get("target_kspace")
        theta_mask = context.get("theta_mask")
        if target_kspace is None or theta_mask is None:
            raise ValueError("context must supply 'target_kspace' and 'theta_mask'.")
        return _kspace_residual(prediction, target_kspace, theta_mask)


@register_loss(
    name="multi_mask_ssdu",
    aliases=["MultiMaskSSDULoss", "robust_ssdu"],
    domain="kspace",
)
class MultiMaskSSDULoss(nn.Module):
    r"""Multi-mask robust SSDU.

    Averages the SSDU residual over a batch of independently-sampled
    :math:`(\Lambda,\Theta)` partitions. Two usage patterns are
    supported:

    1. **Pre-rolled**: the strategy supplies ``context["theta_masks"]``
       as a list of length :math:`N_{\text{masks}}` and the
       reconstructor was already run :math:`N_{\text{masks}}` times,
       producing ``prediction`` of shape
       ``(N_masks, B, C, H, W)``. The loss then averages residuals.

    2. **On-the-fly L1**: a simpler fallback used when only one
       reconstruction is available — degenerates to single-split SSDU
       averaged over multiple :math:`\Theta` masks (information leak,
       provided as a warning for low-budget runs).

    Args:
        norm: ``"l2"`` (default, paper) or ``"l1"``.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError(
                "MultiMaskSSDULoss requires context['target_kspace'] and context['theta_masks']."
            )
        target_kspace = context.get("target_kspace")
        theta_masks = context.get("theta_masks")
        if target_kspace is None or theta_masks is None:
            raise ValueError(
                "context must supply 'target_kspace' and 'theta_masks' (list of held-out masks)."
            )

        if isinstance(theta_masks, torch.Tensor):
            theta_masks = [theta_masks[i] for i in range(theta_masks.shape[0])]

        loss = prediction.new_zeros(())
        # Pre-rolled: prediction has leading mask axis matching theta_masks
        if prediction.dim() == 5 and prediction.shape[0] == len(theta_masks):
            for i, mk in enumerate(theta_masks):
                pred_i = prediction[i]
                pred_k = fft2c(pred_i)
                diff = (pred_k - target_kspace) * _coerce_mask(mk, pred_k)
                if self.norm == "l1":
                    loss = loss + diff.abs().mean()
                else:
                    if diff.is_complex():
                        diff = torch.view_as_real(diff)
                    loss = loss + diff.pow(2).mean()
            return loss / len(theta_masks)

        # Single-prediction fallback: average over multiple holdouts
        for mk in theta_masks:
            pred_k = fft2c(prediction)
            diff = (pred_k - target_kspace) * _coerce_mask(mk, pred_k)
            if self.norm == "l1":
                loss = loss + diff.abs().mean()
            else:
                if diff.is_complex():
                    diff = torch.view_as_real(diff)
                loss = loss + diff.pow(2).mean()
        return loss / max(len(theta_masks), 1)


# Suppress the lint about unused F import — kept for symmetry with siblings.
_ = F
