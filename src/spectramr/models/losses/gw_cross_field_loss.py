r"""Gromov-Wasserstein correspondence-free cross-field alignment (GW-CFA).

Aligns ultra-low-field and high-field representations with *no* correspondence by
matching their intra-domain distance structures. The physical premise: for two
contrasts/fields of the same anatomy the intra-image distance structures are
related by an approximate isometry, because contrast remaps intensities
monotonically (preserving rank-based relational distances) while leaving spatial
adjacency unchanged. Hence the GW optimum recovers the anatomical correspondence
from unpaired multi-field data.

The loss is the entropic Gromov-Wasserstein value (Peyre-Cuturi 2016), reusing
:func:`spectramr.models.ot.optimal_transport.entropic_gromov_wasserstein` as the
solver. Differentiable in the descriptors via the envelope theorem: the coupling
``T`` is solved on detached distance matrices, then the GW value is evaluated at
fixed ``T``. Optional anchor landmarks bias the cost to break the up-to-isometry
indeterminacy on symmetric anatomy.

Pilot status: the GW objective is non-convex; per the validation routing this arm
is empirical-pilot-gated. The loss and its diagnostics ship; experiment metadata
marks the arm experimental until a multi-field pilot stabilises it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss
from spectramr.models.ot.optimal_transport import entropic_gromov_wasserstein


def _intra_distance(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalised intra-domain squared-euclidean distance matrix ``[n, n]``."""
    d = torch.cdist(features, features).pow(2)
    return d / (d.max() + eps)


@register_loss(
    name="gw_cross_field",
    domain="latent",
    compatible_with=["cross_field_translation", "reconstruction", "self_supervised"],
)
class GromovWassersteinCrossFieldLoss(nn.Module):
    """Entropic Gromov-Wasserstein cross-field alignment loss.

    Args:
        reg: Sinkhorn entropic regularisation.
        max_iter: Inner Sinkhorn iterations.
        gw_iter: Outer GW linearisation iterations.
        anchor_bias: Magnitude of the cost reduction applied at anchor pairs
            (``context["anchor_pairs"]``) to resolve isometric symmetries.
    """

    def __init__(
        self,
        reg: float = 0.05,
        max_iter: int = 50,
        gw_iter: int = 10,
        anchor_bias: float = 5.0,
    ) -> None:
        super().__init__()
        self.reg = float(reg)
        self.max_iter = int(max_iter)
        self.gw_iter = int(gw_iter)
        self.anchor_bias = float(anchor_bias)

    def gw_coupling(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        anchor_pairs: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """Solve the entropic GW coupling ``T [n, m]`` on detached distances."""
        c1 = _intra_distance(source).detach()
        c2 = _intra_distance(target).detach()
        if anchor_pairs:
            # Reduce the relational cost at anchored vertices by shrinking their
            # row/col distances toward the matched landmark, breaking symmetry.
            for i, j in anchor_pairs:
                c1[i] = c1[i] / (1.0 + self.anchor_bias)
                c2[j] = c2[j] / (1.0 + self.anchor_bias)
        coupling, _ = entropic_gromov_wasserstein(
            c1, c2, reg=self.reg, max_iter=self.max_iter, gw_iter=self.gw_iter
        )
        return coupling

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """GW value between ``prediction`` (source field) and ``target`` embeddings.

        Args:
            prediction: Source-field embeddings ``[n, d]``.
            target: Target-field embeddings ``[m, d]``.
            context: Optional ``{"anchor_pairs": [(i, j), ...]}``.

        Returns:
            Scalar GW value (differentiable in both embedding sets).
        """
        if prediction.ndim != 2 or target.ndim != 2:
            raise ValueError(
                "GW-CFA expects 2-D embedding batches [n, d]; got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        anchor_pairs = (context or {}).get("anchor_pairs")
        with torch.no_grad():
            coupling = self.gw_coupling(prediction, target, anchor_pairs)

        c1 = _intra_distance(prediction)
        c2 = _intra_distance(target)
        n, m = c1.shape[0], c2.shape[0]
        p = torch.full((n,), 1.0 / n, device=c1.device, dtype=c1.dtype)
        q = torch.full((m,), 1.0 / m, device=c2.device, dtype=c2.dtype)
        const_c = (c1.pow(2) @ p)[:, None] + (c2.pow(2) @ q)[None, :]
        value = ((const_c - 2.0 * (c1 @ coupling @ c2)) * coupling).sum()
        return value


__all__ = ["GromovWassersteinCrossFieldLoss"]
