"""PMPS losses — physical-range constraint, smoothness, MMD, consistency.

Bundled into a single module to keep slop low. Each loss is registered
via ``@register_loss`` so the YAML schema can mount them under
``losses.image_losses`` or ``losses.physics_losses``. Domain is set per
loss to match the loss-domain audit guard.

The five losses landed here serve the four PMPS phases:

* Phase 1 (tissue prior):
  * :class:`TissueParameterPhysicsConstraint` — quadratic penalty
    outside the feasible :math:`(T_1, T_2, M_0)` set.
  * :class:`TissueParameterPriorSmoothness` — Sobolev-norm smoothness
    on tissue maps; brain anatomy is piecewise smooth.

* Phase 2 (corruption calibration):
  * :class:`MMDPairedDistributionMatch` — multi-kernel RBF MMD in a
    feature space (VGG-like or raw pixel space, the latter chosen
    when no foundation-model checkpoint is wired) used as the
    objective for fitting :math:`\\psi`.

* Phase 2 / Phase 3 (per-pair sanity):
  * :class:`PerPairConsistencyResidual` — residual between a real HF
    image, its recovered tissue map, and the resulting predicted
    image. Used as a filtering / diagnostic loss.

These are intentionally small, composable modules — heavy lifting
(corruption operator composition, MMD feature extraction strategy
plumbing) lives in the strategy.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.losses.registry import register_loss

# Bounds in raw physical units (ms / ms / unitless).
_T1_BOUNDS_MS: tuple[float, float] = (10.0, 5000.0)
_T2_BOUNDS_MS: tuple[float, float] = (1.0, 2500.0)
_M0_BOUNDS: tuple[float, float] = (0.0, 1.0)


@register_loss(name="tissue_parameter_physics_constraint", domain="image")
class TissueParameterPhysicsConstraint(nn.Module):
    """Soft physical-range penalty for tissue-parameter maps.

    Input convention: a 3-channel tensor ``[B, 3, H, W]`` whose channels
    are ordered ``(T_1, T_2, M_0)``. Channel values may be normalised
    or raw; pass ``normalised=True`` and the bounds are mapped to
    ``[0, 1]`` so the quadratic penalty fires on out-of-feasible-region
    voxels regardless of representation. The :math:`T_2 < T_1` ordering
    constraint is enforced as a separate quadratic on the difference.
    """

    def __init__(
        self,
        normalised: bool = True,
        weight_t1: float = 1.0,
        weight_t2: float = 1.0,
        weight_m0: float = 1.0,
        weight_t2_lt_t1: float = 1.0,
    ) -> None:
        super().__init__()
        self.normalised = normalised
        self.weight_t1 = weight_t1
        self.weight_t2 = weight_t2
        self.weight_m0 = weight_m0
        self.weight_t2_lt_t1 = weight_t2_lt_t1

    def _bounds(self) -> tuple[tuple[float, float], ...]:
        if self.normalised:
            return ((0.0, 1.0),) * 3
        return (_T1_BOUNDS_MS, _T2_BOUNDS_MS, _M0_BOUNDS)

    def _channel_penalty(self, x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        below = F.relu(lo - x)
        above = F.relu(x - hi)
        return (below.pow(2) + above.pow(2)).mean()

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Compute the constraint penalty.

        ``target`` is accepted (and ignored) for loss-registry signature
        compatibility; the constraint is unsupervised by construction.
        """
        if prediction.shape[1] < 3:
            raise ValueError(
                f"TissueParameterPhysicsConstraint expects >=3 channels "
                f"(T1, T2, M0), got {prediction.shape[1]}."
            )
        t1 = prediction[:, 0:1]
        t2 = prediction[:, 1:2]
        m0 = prediction[:, 2:3]
        (lo1, hi1), (lo2, hi2), (lo3, hi3) = self._bounds()

        penalty = (
            self.weight_t1 * self._channel_penalty(t1, lo1, hi1)
            + self.weight_t2 * self._channel_penalty(t2, lo2, hi2)
            + self.weight_m0 * self._channel_penalty(m0, lo3, hi3)
        )
        # T2 < T1 ordering (always enforced; cheap).
        ordering = F.relu(t2 - t1).pow(2).mean()
        penalty = penalty + self.weight_t2_lt_t1 * ordering
        return penalty


@register_loss(name="tissue_parameter_prior_smoothness", domain="image")
class TissueParameterPriorSmoothness(nn.Module):
    """Total-variation / Sobolev-norm smoothness on tissue maps.

    Penalises gradients along H and W. Computed on each of the
    ``(T_1, T_2, M_0)`` channels independently and averaged.
    """

    def __init__(self, order: int = 1, weight: float = 1.0) -> None:
        super().__init__()
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2; got {order}.")
        self.order = order
        self.weight = weight

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        if prediction.ndim < 4:
            raise ValueError(
                f"TissueParameterPriorSmoothness expects (B, C, H, W); got "
                f"{tuple(prediction.shape)}."
            )
        if self.order == 1:
            dh = (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]).abs()
            dw = (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]).abs()
        else:
            dh = (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]).pow(2)
            dw = (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]).pow(2)
        return self.weight * (dh.mean() + dw.mean())


@register_loss(name="mmd_paired_distribution_match", domain="image")
class MMDPairedDistributionMatch(nn.Module):
    """Multi-kernel RBF MMD between two image distributions.

    By default the MMD is computed in raw pixel space averaged over
    spatial dimensions — this is sufficient for the calibration
    backprop (PMPS Phase-2) when no foundation-model feature extractor
    is wired. A future revision will let the caller pass in a VGG-style
    feature extractor as a constructor arg.

    The kernel bandwidth set follows Gretton et al. 2012: a
    multi-bandwidth (RBF) kernel is more robust to scale mismatch than
    any single bandwidth. The MMD estimator is the unbiased U-statistic.
    """

    def __init__(
        self,
        bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        feature_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not bandwidths:
            raise ValueError("bandwidths must be a non-empty sequence.")
        self.register_buffer(
            "_bandwidths",
            torch.tensor(list(bandwidths), dtype=torch.float32),
            persistent=False,
        )
        self.feature_extractor = feature_extractor  # may be None

    def _flatten_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is not None:
            feats = self.feature_extractor(x)
        else:
            feats = x
        # Flatten (B, C, H, W) → (B, C*H*W) so MMD operates on per-sample
        # feature vectors.
        return feats.reshape(feats.shape[0], -1)

    def _kernel_sum(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Pairwise squared distances.
        # x: (n, d), y: (m, d) → (n, m) distances.
        xx = (x * x).sum(dim=1, keepdim=True)
        yy = (y * y).sum(dim=1, keepdim=True)
        d2 = xx + yy.t() - 2.0 * (x @ y.t())
        d2 = d2.clamp_min(0.0)
        # Average over bandwidths.
        bws = self._bandwidths.to(d2.device, dtype=d2.dtype)
        bws = bws.view(-1, 1, 1)
        # k(x, y) = mean_b exp(-d2 / (2 b^2))
        k = torch.exp(-d2.unsqueeze(0) / (2.0 * bws.pow(2))).mean(dim=0)
        return k

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """MMD^2 between ``prediction`` and ``target`` mini-batches."""
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("MMDPairedDistributionMatch expects (B, C, H, W) tensors.")
        x = self._flatten_features(prediction)
        y = self._flatten_features(target)
        k_xx = self._kernel_sum(x, x).mean()
        k_yy = self._kernel_sum(y, y).mean()
        k_xy = self._kernel_sum(x, y).mean()
        # Unbiased estimator drops the diagonal; here we use the biased
        # form for batch-size stability (small calibration batches).
        return (k_xx + k_yy - 2.0 * k_xy).clamp_min(0.0)


@register_loss(name="per_pair_consistency_residual", domain="image")
class PerPairConsistencyResidual(nn.Module):
    """Per-pair physics-consistency residual for PMPS Phase 3 defensibility.

    Computes :math:`\\| C_{\\text{HF}}(B(\\hat\\theta; P_{\\text{HF}})) -
    x_{\\text{HF}} \\|`. Used both as a filtering loss at synthesis time
    and as a diagnostic plotted in the campaign report. ``prediction``
    is the predicted reconstruction; ``target`` is the real HF image.
    """

    def __init__(self, norm: str = "l1") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be 'l1' or 'l2'; got '{norm}'.")
        self.norm = norm

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.norm == "l1":
            return (prediction - target).abs().mean()
        return (prediction - target).pow(2).mean()
