r"""Extra registration losses: mutual information, bending energy, inverse
consistency.

These three are the canonical *missing trio* for unsupervised
deformable registration on top of NCC/LNCC and the simple gradient
smoothness regulariser already in ``registration.py``.

* **Mutual Information** (Pluim, Maintz, Viergever 2003) for multi-modal
  alignment via differentiable Parzen-window joint histograms.
* **Bending Energy** (Rueckert *et al.* 1999) for a smoother
  deformation field than first-order smoothness.
* **Inverse Consistency** (Christensen, Johnson 2001) for symmetric
  registration: requires that the forward and inverse transforms
  compose to identity.

References:
    [46/48] J. P. W. Pluim *et al.*, *IEEE TMI*, 22(8), 2003.
    [50] D. Rueckert *et al.*, "Nonrigid registration using free-form
        deformations," *IEEE TMI*, 18(8), 1999.
    [49/52] G. E. Christensen, H. J. Johnson, *IEEE TMI*, 20(7), 2001.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.losses.registry import register_loss


def _parzen_joint_histogram(
    x: torch.Tensor, y: torch.Tensor, bins: int, sigma: float
) -> torch.Tensor:
    """Soft joint histogram via Gaussian Parzen windows.

    x, y: tensors of identical shape, real-valued, normalised into [0, 1].
    Returns a (bins, bins) joint histogram, normalised to sum to 1.
    """
    centres = torch.linspace(0.0, 1.0, bins, device=x.device, dtype=x.dtype)
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    # Soft-binning via (squared) Gaussian
    wx = torch.exp(-0.5 * ((x_flat - centres) / sigma).pow(2))  # (N, bins)
    wy = torch.exp(-0.5 * ((y_flat - centres) / sigma).pow(2))  # (N, bins)
    wx = wx / wx.sum(dim=1, keepdim=True).clamp_min(1e-8)
    wy = wy / wy.sum(dim=1, keepdim=True).clamp_min(1e-8)
    joint = wx.t() @ wy  # (bins, bins)
    joint = joint / joint.sum().clamp_min(1e-8)
    return joint


@register_loss(
    name="mutual_information",
    aliases=["MILoss", "mi", "mutual_info"],
    domain="image",
)
class MutualInformationLoss(nn.Module):
    r"""Negative Parzen-window mutual information.

    .. math::
        I(X;Y) = \sum_{i,j} p(i,j) \log\frac{p(i,j)}{p(i)p(j)},
        \quad \mathcal{L} = -I(X;Y).

    Args:
        bins: Number of histogram bins (default 32).
        sigma: Parzen kernel bandwidth on the [0, 1] histogram axis
            (default 1/(bins-1), one bin width).
        normalise: If True (default), inputs are min-max normalised
            into [0, 1] per-batch before histogramming. Disable when
            inputs are already calibrated (e.g. 0–1 magnitude images).
    """

    def __init__(
        self,
        bins: int = 32,
        sigma: float | None = None,
        normalise: bool = True,
    ) -> None:
        super().__init__()
        if bins < 4:
            raise ValueError(f"bins must be >= 4, got {bins}")
        self.bins = bins
        self.sigma = sigma if sigma is not None else 1.0 / (bins - 1)
        self.normalise = normalise

    def _norm01(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalise:
            return x.clamp(0.0, 1.0)
        flat = x.reshape(x.shape[0], -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        return ((flat - lo) / (hi - lo + 1e-8)).reshape_as(x).clamp(0.0, 1.0)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}"
            )
        x = self._norm01(prediction)
        y = self._norm01(target)
        joint = _parzen_joint_histogram(x, y, self.bins, self.sigma)
        p_x = joint.sum(dim=1, keepdim=True)
        p_y = joint.sum(dim=0, keepdim=True)
        # Mutual information = sum p(x,y) * log(p(x,y) / (p(x)p(y)))
        denom = (p_x * p_y).clamp_min(1e-12)
        mi = (joint * (joint.clamp_min(1e-12) / denom).log()).sum()
        return -mi


@register_loss(
    name="bending_energy",
    aliases=["BendingEnergyLoss"],
    domain="image",
)
class BendingEnergyLoss(nn.Module):
    r"""Second-order bending-energy regulariser on a displacement field.

    .. math::
        \mathcal{L}_{\text{be}} = \sum_{i,j}\big\|\partial^2_{ij}\boldsymbol{\phi}\big\|^2.

    Forward signature: ``prediction`` is the displacement field of
    shape ``(B, ndim, H, W)`` (2D) or ``(B, ndim, D, H, W)`` (3D).
    ``target`` is ignored. Returns the average squared second
    derivative across all axis pairs and field components.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _second_diff(field: torch.Tensor, axis: int) -> torch.Tensor:
        """:math:`\\partial^2 f/\\partial \\text{axis}^2` via central differences.

        No padding: returns the interior of the field along ``axis`` so
        the output is two voxels shorter on that axis. Boundary effects
        on the other regulariser terms wash out under spatial averaging.
        """
        slicer_l = [slice(None)] * field.dim()
        slicer_c = [slice(None)] * field.dim()
        slicer_r = [slice(None)] * field.dim()
        slicer_l[axis] = slice(0, -2)
        slicer_c[axis] = slice(1, -1)
        slicer_r[axis] = slice(2, None)
        return field[tuple(slicer_r)] - 2 * field[tuple(slicer_c)] + field[tuple(slicer_l)]

    @staticmethod
    def _first_diff(field: torch.Tensor, axis: int) -> torch.Tensor:
        """Central first difference :math:`(f[i+1]-f[i-1])/2` along ``axis``.

        No padding (two voxels shorter on ``axis``). Composing this along two
        different axes yields the mixed second partial
        :math:`\\partial^2 f/\\partial a\\,\\partial b` — composing two
        ``_second_diff`` operators instead would give a 4th-order term.
        """
        slicer_l = [slice(None)] * field.dim()
        slicer_r = [slice(None)] * field.dim()
        slicer_l[axis] = slice(0, -2)
        slicer_r[axis] = slice(2, None)
        return (field[tuple(slicer_r)] - field[tuple(slicer_l)]) / 2.0

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.dim() < 4:
            raise ValueError(
                f"BendingEnergyLoss expects (B, ndim, *spatial); got shape "
                f"{tuple(prediction.shape)}"
            )
        spatial_axes = list(range(2, prediction.dim()))
        loss = prediction.new_zeros(())
        for ax in spatial_axes:
            d2 = self._second_diff(prediction, ax)
            loss = loss + d2.pow(2).mean()
            for ax2 in spatial_axes:
                if ax2 <= ax:
                    continue
                # Mixed second partial d^2 f / (d ax)(d ax2): a central first
                # difference in each axis. Composing two SECOND differences (the
                # previous code) approximates d^4 f / (d ax^2)(d ax2^2), not the
                # mixed term the bending energy (Rueckert 1999) calls for.
                d_mixed = self._first_diff(self._first_diff(prediction, ax), ax2)
                loss = loss + 2.0 * d_mixed.pow(2).mean()
        return loss


@register_loss(
    name="inverse_consistency",
    aliases=["InverseConsistencyLoss"],
    domain="image",
)
class InverseConsistencyLoss(nn.Module):
    r"""Forward∘backward = identity penalty.

    .. math::
        \mathcal{L}_{\text{ic}} = \|\boldsymbol{\phi}_{AB}\circ\boldsymbol{\phi}_{BA}-\mathrm{id}\|^2.

    Forward expects:

    * ``prediction``: forward-then-backward composed displacement field
      of shape matching the identity grid (the strategy is responsible
      for composing :math:`\phi_{AB}\circ\phi_{BA}`).
    * ``target``: ignored (the identity field is constructed
      internally with the same shape).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.dim() < 4:
            raise ValueError(
                f"InverseConsistencyLoss expects (B, ndim, *spatial); got {tuple(prediction.shape)}"
            )
        # Build identity displacement field (zero displacement everywhere).
        identity = torch.zeros_like(prediction)
        return (prediction - identity).pow(2).mean()
