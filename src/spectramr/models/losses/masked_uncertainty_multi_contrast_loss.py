"""Foreground-masked, per-contrast uncertainty-weighted L1 loss.

Implements the multi-contrast composite from GeoMamba-ULF plan §5.3:

.. math::

    \\mathcal{L} = \\sum_{c} \\Big(
        \\frac{1}{2\\sigma_c^{2}}\\;
        \\frac{\\sum_{F} \\bigl|\\hat I_{HF}^{(c)} - I_{HF}^{(c)}\\bigr|}{|F|}
        + \\log \\sigma_c
    \\Big)

The per-contrast log-sigma parameters are learnable (Kendall & Gal,
*Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
Geometry and Semantics*, CVPR 2018). The :math:`\\log \\sigma` term
keeps :math:`\\sigma` from collapsing to infinity; the
:math:`1/(2\\sigma^{2})` factor automatically downweights noisy /
hard contrasts.

Reductions are computed over the foreground mask only — background
voxels carry no signal at ULF and dilute the gradient.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(
    name="masked_uncertainty_multi_contrast",
    aliases=[
        "MaskedUncertaintyMultiContrast",
        "uncertainty_multi_contrast",
        "kendall_gal_multi_contrast",
    ],
)
class MaskedUncertaintyMultiContrastLoss(nn.Module):
    """Per-contrast uncertainty-weighted, foreground-masked L1.

    Args:
        n_contrasts: vocabulary size for the FiLM contrast embedding
            (T1 / T2 / FLAIR ⇒ 3). One ``log_sigma`` parameter is
            allocated per contrast.
        init_log_sigma: starting value for every ``log_sigma`` slot
            (corresponds to ``sigma = exp(init)``; ``0.0`` ⇒ ``sigma=1``).
        eps: numerical floor on the precision factor
            ``1 / (2 * sigma^2)`` to prevent NaN / Inf at extreme
            ``log_sigma`` values.
        reduction: ``"mean"`` averages across present contrasts in a
            batch; ``"sum"`` returns the raw sum (matches the formula
            literally). Per-step diagnostics are returned regardless.

    The loss has no fixed shape contract beyond ``pred.shape == target.shape``
    and a per-volume ``contrast_id`` of shape ``(B,)`` (or scalar).
    Mask broadcast follows the same convention as
    :class:`spectramr.infrastructure.training.strategies.geomamba_ulf_strategy.GeoMambaULFStrategy`.
    """

    def __init__(
        self,
        n_contrasts: int = 3,
        init_log_sigma: float = 0.0,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if n_contrasts < 1:
            raise ValueError(f"n_contrasts must be >= 1, got {n_contrasts}")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

        self.n_contrasts = n_contrasts
        self.eps = eps
        self.reduction = reduction
        # Learnable per-contrast log-sigma. Optimised jointly with the
        # generator via the standard optimizer parameter list.
        self.log_sigma = nn.Parameter(torch.full((n_contrasts,), float(init_log_sigma)))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        contrast_id: torch.Tensor | int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the uncertainty-weighted, foreground-masked L1 loss.

        Args:
            pred: prediction of shape ``(B, C, ...)``.
            target: ground truth of identical shape.
            mask: optional foreground mask, same shape (or broadcastable
                from ``(B, 1, ...)`` / ``(B, ...)``).
            contrast_id: per-volume contrast index in ``[0, n_contrasts)``.
                When ``None``, defaults to contrast 0 for every sample —
                the loss then collapses to a single learned σ.

        Returns:
            Scalar loss tensor with autograd through both the
            reconstruction residual and the ``log_sigma`` parameters.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must share shape, got {tuple(pred.shape)} "
                f"vs {tuple(target.shape)}"
            )

        cid = self._normalise_contrast_id(contrast_id, batch_size=pred.shape[0], device=pred.device)
        if int(cid.max().item()) >= self.n_contrasts or int(cid.min().item()) < 0:
            raise ValueError(
                f"contrast_id values must be in [0, {self.n_contrasts}), "
                f"got min={int(cid.min().item())} max={int(cid.max().item())}"
            )

        residual = (pred - target).abs()
        if mask is not None:
            mask_b = self._broadcast_mask(mask, pred)
            residual = residual * mask_b
            denom = mask_b.flatten(1).sum(dim=1).clamp_min(1.0)  # (B,)
        else:
            denom = torch.full(
                (pred.shape[0],),
                fill_value=float(residual[0].numel()),
                device=pred.device,
                dtype=pred.dtype,
            )

        per_sample_l1 = residual.flatten(1).sum(dim=1) / denom  # (B,)

        # Aggregate per contrast and weight by 1 / (2 * sigma^2) + log sigma.
        total = pred.new_zeros(())
        present = 0
        for c in range(self.n_contrasts):
            sel = cid == c
            if not bool(sel.any()):
                continue
            present += 1
            l1_c = per_sample_l1[sel].mean()
            log_sigma_c = self.log_sigma[c]
            precision = (-2.0 * log_sigma_c).exp().clamp_min(self.eps)
            total = total + 0.5 * precision * l1_c + log_sigma_c

        if present == 0:
            # Defensive: should not happen if contrast_id was validated.
            return per_sample_l1.mean() * pred.new_zeros(())
        if self.reduction == "mean":
            total = total / present
        return total

    # ------------------------------------------------------------------
    # Diagnostics for the strategy / logger
    # ------------------------------------------------------------------

    def per_contrast_sigma(self) -> torch.Tensor:
        """Returns the current σ per contrast (no autograd graph)."""
        with torch.no_grad():
            return self.log_sigma.detach().exp()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_contrast_id(
        cid: torch.Tensor | int | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if cid is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if isinstance(cid, int):
            return torch.full((batch_size,), cid, dtype=torch.long, device=device)
        cid = cid.to(device=device, dtype=torch.long)
        if cid.ndim == 0:
            return cid.expand(batch_size).clone()
        if cid.shape[0] != batch_size:
            raise ValueError(f"contrast_id has batch dim {cid.shape[0]} but batch is {batch_size}")
        return cid

    @staticmethod
    def _broadcast_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if mask.shape == ref.shape:
            return mask.to(dtype=ref.dtype)
        if mask.ndim == ref.ndim - 1:  # missing channel dim
            mask = mask.unsqueeze(1)
        return mask.to(dtype=ref.dtype).expand_as(ref)


__all__ = ["MaskedUncertaintyMultiContrastLoss"]
