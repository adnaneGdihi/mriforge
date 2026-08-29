r"""Cellular-sheaf consistency loss for k-space / multi-contrast reconstruction.

This loss gives the ``sheaf_kspace`` / ``sheaf_multi_contrast`` strategy keys a
*genuine* sheaf-theoretic objective rather than a vanilla reconstruction
surrogate. It consumes the primitives in
:mod:`mriforge.infrastructure.physics.cellular_sheaf` directly.

A reconstructed volume is interpreted as a section of a *cellular sheaf*
:math:`\mathcal{F}` on a cover graph :math:`G=(V,E)`: each channel of the
``[B, C, H, W]`` tensor (a contrast, a coil, or an overlapping k-space patch)
is a **cover cell** / vertex, with stalk
:math:`\mathcal{F}(v)=\mathbb{R}^{HW}` carrying the flattened spatial signal.
Adjacent cells are joined by edges whose restriction maps are the identity
(the simplest "agreement" sheaf): two cells are *consistent* when they carry
the same section on their shared support.

The objective is the **sheaf-Laplacian quadratic form**

.. math::

    \langle x, L_{\mathcal{F}}\,x\rangle
        = \sum_{e=(u,v)} \| \mathcal{F}_{v\trianglelefteq e}x_v
                          - \mathcal{F}_{u\trianglelefteq e}x_u \|^2 ,

which is :math:`0` iff :math:`x` is a **global section**
(:math:`x\in H^0(G;\mathcal{F})`) — i.e. all cover cells agree — and strictly
positive when local sections disagree [Hansen-Ghrist 2019]. We additionally
penalise the **Mayer-Vietoris obstruction** :math:`\|[w]\|_{H^1}` of glueing
the *prediction* and *target* on a two-patch cover, which vanishes iff the two
agree on their overlap.

See :mod:`mriforge.infrastructure.physics.cellular_sheaf` for the algebraic
primitives.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.cellular_sheaf import (
    mayer_vietoris_obstruction_norm,
    sheaf_laplacian_quadratic_form,
)
from mriforge.models.losses.registry import register_loss


@register_loss(
    name="sheaf_consistency",
    aliases=["cellular_sheaf", "sheaf_kspace_consistency"],
    domain="image",
)
class SheafConsistencyLoss(nn.Module):
    r"""Cellular-sheaf consistency penalty over a cover of cover cells.

    The prediction's channels are treated as the cells of a cellular cover
    (contrasts / coils / overlapping k-space patches). The loss penalises the
    sheaf-Laplacian energy of the prediction (disagreement between cells under
    identity restriction maps) plus the Mayer-Vietoris obstruction between the
    prediction and the target.

    The ``prediction`` and ``target`` are ``[B, C, H, W]`` (or
    ``[B, C, ...]``) real tensors. ``C`` is the number of cover cells; the
    remaining axes are flattened into the per-cell stalk.

    Args:
        laplacian_weight: Coefficient on the sheaf-Laplacian quadratic form
            of the prediction. Default 1.0.
        obstruction_weight: Coefficient on the Mayer-Vietoris obstruction
            norm between prediction and target. Default 1.0.
        reduction: ``"mean"`` (average over the batch) or ``"sum"``.
    """

    def __init__(
        self,
        laplacian_weight: float = 1.0,
        obstruction_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if laplacian_weight < 0:
            raise ValueError("laplacian_weight must be >= 0")
        if obstruction_weight < 0:
            raise ValueError("obstruction_weight must be >= 0")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}")
        self.laplacian_weight = laplacian_weight
        self.obstruction_weight = obstruction_weight
        self.reduction = reduction

    @staticmethod
    def _path_edges(num_cells: int, device: torch.device) -> torch.Tensor:
        """Edges of a path graph over the cover cells (each cell agrees with
        its neighbour). For a single cell there are no edges.
        """
        if num_cells < 2:
            return torch.empty(0, 2, dtype=torch.long, device=device)
        u = torch.arange(num_cells - 1, device=device)
        v = u + 1
        return torch.stack([u, v], dim=1)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} and target shape "
                f"{tuple(target.shape)} must match."
            )
        if prediction.ndim < 3:
            raise ValueError(f"expected [B, C, ...] with >=3 dims; got {tuple(prediction.shape)}.")

        batch = prediction.shape[0]
        num_cells = prediction.shape[1]
        # Per sample: (C cells, n=stalk dim) where n = product of spatial dims.
        pred_flat = prediction.reshape(batch, num_cells, -1)
        tgt_flat = target.reshape(batch, num_cells, -1)
        stalk = pred_flat.shape[-1]

        edges = self._path_edges(num_cells, prediction.device)
        num_edges = edges.shape[0]
        # Identity restriction maps F_{u<=e} = F_{v<=e} = I_n: the agreement
        # sheaf. (E, n, n) per edge.
        eye = torch.eye(stalk, dtype=prediction.dtype, device=prediction.device)
        restriction = eye.unsqueeze(0).expand(num_edges, stalk, stalk)

        per_sample = prediction.new_zeros(batch)
        for b in range(batch):
            # --- Sheaf-Laplacian energy of the prediction (= 0 iff global
            #     section, i.e. all cover cells agree). ---
            if self.laplacian_weight > 0 and num_edges > 0:
                lap = sheaf_laplacian_quadratic_form(pred_flat[b], edges, restriction, restriction)
                # Normalise by edges*stalk so the energy is a mean disagreement.
                per_sample[b] = per_sample[b] + self.laplacian_weight * (lap / (num_edges * stalk))

            # --- Mayer-Vietoris obstruction between prediction and target on a
            #     two-patch cover (whole-section overlap). Zero iff they agree. ---
            if self.obstruction_weight > 0:
                pred_section = pred_flat[b].reshape(-1)
                tgt_section = tgt_flat[b].reshape(-1)
                obstruction = mayer_vietoris_obstruction_norm(
                    pred_section, tgt_section, pred_section, tgt_section
                )
                per_sample[b] = per_sample[b] + self.obstruction_weight * (
                    obstruction / pred_section.shape[0]
                )

        if self.reduction == "mean":
            return per_sample.mean()
        return per_sample.sum()
