r"""Self-supervised k-space consistency loss for test-time adaptation (plan §D).

Single-subject TTA has no ground-truth target at inference time, so the
adaptation objective must be *self-supervised*: it can only use the
acquired (undersampled) k-space of the subject in front of it. The natural
objective is k-space data consistency — the reconstruction, pushed back to
k-space, must agree with the measured samples at the acquired locations:

.. math::

    \mathcal{L}_{dc} = \big\lVert M \odot \big(\mathcal{F}\,\hat{x} - y\big) \big\rVert_1

where :math:`\hat{x}` is the current reconstruction, :math:`\mathcal{F}` the
(orthonormal) 2-D DFT, :math:`y` the measured k-space, and :math:`M` the
sampling mask (1 at acquired locations, 0 elsewhere). Only the *acquired*
locations enter the loss — the network is free at the un-sampled locations,
which is exactly what makes this self-supervised rather than a supervised
k-space L1.

This is the objective the :class:`SpectraTestTimeAdaptationStrategy` minimises
during per-subject adaptation (plan §D "a self-supervised k-space consistency
loss is the natural objective for single-subject adaptation"). It pairs with
the ``tta_adapter_only`` audit guard, which keeps the backbone frozen so the
adaptation memory footprint stays bounded.

Reference
---------
Plan §D of ``docs/SPECTRA_implementation_plan.md`` (SPECTRA → MRIForge).
Darestani, Liu & Heckel, "Test-Time Training ... Compressed Sensing"
(arXiv:2204.07204) — self-supervised single-subject adaptation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.losses.registry import register_loss


@register_loss(
    name="kspace_consistency_tta",
    aliases=["KspaceConsistencyTTALoss"],
    domain="kspace",
    # Accepts an image-domain reconstruction too: it FFTs internally, so an
    # image-domain TTA arm may declare it under losses.image_losses.
    compatible_with=["image"],
)
class KspaceConsistencyTTALoss(nn.Module):
    """Masked k-space data-consistency loss for single-subject TTA.

    The forward signature mirrors the registry convention ``forward(pred,
    target, **kwargs)`` and returns a scalar tensor. Here ``pred`` is the
    current reconstruction (image domain by default; set ``pred_is_kspace=True``
    if the arm already emits k-space) and ``target`` is the measured k-space
    ``y``. The sampling mask ``M`` is supplied via ``mask=`` (a 0/1 tensor
    broadcastable to k-space); when omitted the loss degrades gracefully to a
    dense k-space L1 (all locations acquired) and emits no error so the probe
    path still exercises the gradient.

    Args:
        weight: Scalar multiplier on the consistency term.
        pred_is_kspace: If True, ``pred`` is already k-space and the internal
            FFT is skipped.
        reduction: ``"mean"`` (default) or ``"sum"`` over the masked residual.
    """

    def __init__(
        self,
        weight: float = 1.0,
        pred_is_kspace: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.weight = float(weight)
        self.pred_is_kspace = bool(pred_is_kspace)
        self.reduction = reduction

    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        """Coerce a real [B, 2, ...] real/imag tensor to complex; pass complex through."""
        if torch.is_complex(x):
            return x
        if x.dim() >= 2 and x.shape[1] == 2:
            return torch.complex(x[:, 0:1], x[:, 1:2])
        # real magnitude image -> zero-phase complex
        return torch.complex(x, torch.zeros_like(x))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        pred_c = self._to_complex(pred)

        # Push the reconstruction to k-space via the centered, orthonormal SSOT DFT
        # (``fft2c``) — never a raw ``torch.fft.fft2`` (CLAUDE.md pitfall #2). The
        # measured k-space ``y`` and the sampling mask ``M`` are in the centered
        # convention (DC in the middle); an uncentered forward would mask the wrong
        # k-space lines whenever an external (centered) ``y``/``M`` is supplied.
        if self.pred_is_kspace:
            pred_k = pred_c
        else:
            from mriforge.infrastructure.physics.fft_ops import fft2c

            pred_k = fft2c(pred_c)

        target_k = self._to_complex(target)

        residual = pred_k - target_k
        if mask is not None:
            residual = residual * mask.to(residual.real.dtype)

        # |·|_1 on a complex residual = sum of complex magnitudes.
        mag = torch.abs(residual)
        loss = mag.mean() if self.reduction == "mean" else mag.sum()
        return self.weight * loss
