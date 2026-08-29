r"""Cramér-Rao-optimal multi-sensor pose fusion (idea 9).

Combines per-modality SE(3)-pose estimates with their reported
covariances using inverse-variance weighting. For Gaussian
observation noise this is the maximum-likelihood estimator and
attains the Cramér-Rao lower bound on the fused pose variance.

Mathematics
-----------
Given :math:`K` per-modality estimates :math:`\hat\zeta_k` with
diagonal covariances :math:`\Sigma_k`, the optimal fused pose is

.. math::

    \hat\zeta_{\mathrm{fused}} = \Sigma_{\mathrm{fused}}
        \sum_{k=1}^{K} \Sigma_k^{-1} \hat\zeta_k,
    \qquad
    \Sigma_{\mathrm{fused}} = \Bigl(\sum_k \Sigma_k^{-1}\Bigr)^{-1}.

This is exactly equivalent to averaging the per-sensor estimates
weighted by the inverse of their variance — the standard form used
in robotics multi-sensor SLAM (Durrant-Whyte & Henderson 2016).

Pose representation
-------------------
We use a 6-vector ``(t_x, t_y, t_z, r_x, r_y, r_z)`` with
:math:`r` an axis-angle (Lie-algebra) rotation. The fusion is
correct when *all* sensors are pre-rotated to the MR scanner frame.

Reference
---------
Durrant-Whyte H., Henderson T. C. *Multisensor Data Fusion*. In:
Siciliano B., Khatib O. (eds), Springer Handbook of Robotics
(2nd ed., 2016).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PoseEstimate:
    """A single per-modality SE(3) pose estimate with a noise covariance.

    Attributes:
        pose: 1-D ``[6]`` tensor: ``(t_x, t_y, t_z, r_x, r_y, r_z)``.
        covariance: ``[6, 6]`` positive-semi-definite covariance.
        modality: short identifier (``"rgbd"`` / ``"imu"`` / ``"ecg"``
            / ``"respiratory"``) — used by the audit hook to flag
            unsupported names.
    """

    pose: torch.Tensor
    covariance: torch.Tensor
    modality: str

    def __post_init__(self) -> None:
        if self.pose.shape != (6,):
            raise ValueError(f"pose must be a 6-vector; got {tuple(self.pose.shape)}")
        if self.covariance.shape != (6, 6):
            raise ValueError(f"covariance must be 6×6; got {tuple(self.covariance.shape)}")


@dataclass(frozen=True)
class FusedPose:
    """The fused output: pose + covariance + per-modality weights.

    Attributes:
        pose: 1-D ``[6]`` MLE pose.
        covariance: ``[6, 6]`` fused covariance.
        weights: per-modality scalar weights summing to 1
            (interpretation: ``modality_k`` contributed ``weights[k]``
            of the trace of ``Σ⁻¹_fused``).
    """

    pose: torch.Tensor
    covariance: torch.Tensor
    weights: torch.Tensor


def _normalise_information_traces(infos: Sequence[torch.Tensor]) -> torch.Tensor:
    """Normalise ``tr(info_k)`` over precomputed information matrices.

    Shared by :func:`inverse_variance_weights` and
    :func:`cramer_rao_optimal_fusion` so the two never disagree on how a
    weight is computed (and so the fusion path does not re-invert
    covariances it has already inverted).
    """
    traces = torch.stack([info.diagonal().sum() for info in infos])
    total = traces.sum()
    if total.item() <= 0:
        raise ValueError("All covariances are degenerate (zero information).")
    return (traces / total).to(torch.float32)


def inverse_variance_weights(
    estimates: Sequence[PoseEstimate],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-modality scalar weights, normalised to sum to 1.

    The weight for modality :math:`k` is

    .. math:: w_k = \\frac{\\mathrm{tr}(\\Sigma_k^{-1})}{\\sum_j \\mathrm{tr}(\\Sigma_j^{-1})}.

    Useful for diagnostics — *not* what the fusion math actually
    uses (see :func:`cramer_rao_optimal_fusion`).

    Args:
        estimates: Per-modality :class:`PoseEstimate` list (length ≥ 1).
        eps: Tikhonov regulariser added to each covariance before
            inversion. Matches :func:`cramer_rao_optimal_fusion` so a
            singular (e.g. zero-variance) sensor does not crash the
            diagnostic weights while the fusion itself succeeds.
    """
    if len(estimates) == 0:
        raise ValueError("estimates is empty")
    eye = torch.eye(6, dtype=torch.float64)
    infos = [torch.linalg.inv(e.covariance.to(torch.float64) + eps * eye) for e in estimates]
    return _normalise_information_traces(infos)


def cramer_rao_optimal_fusion(
    estimates: Sequence[PoseEstimate],
    eps: float = 1e-12,
) -> FusedPose:
    r"""Fuse SE(3) estimates with inverse-variance (Cramér-Rao) weighting.

    Args:
        estimates: Per-modality :class:`PoseEstimate` list (length ≥ 1).
        eps: Tikhonov regulariser added to each covariance before
            inversion to keep degenerate inputs from blowing up.

    Returns:
        :class:`FusedPose`. ``covariance`` is guaranteed to have
        diagonal entries no larger than the minimum per-modality
        diagonal entry (Cramér-Rao optimality, see test).

    Raises:
        ValueError: empty list, or covariance shape / pose shape mismatch.
    """
    if len(estimates) == 0:
        raise ValueError("estimates is empty")

    # Reference dtype and device are taken from the first estimate.
    dtype = estimates[0].pose.dtype
    device = estimates[0].pose.device
    eye = torch.eye(6, dtype=dtype, device=device)

    info_sum = torch.zeros(6, 6, dtype=dtype, device=device)
    weighted_sum = torch.zeros(6, dtype=dtype, device=device)
    infos: list[torch.Tensor] = []
    for est in estimates:
        cov = est.covariance.to(dtype).to(device) + eps * eye
        info = torch.linalg.inv(cov)
        infos.append(info)
        info_sum = info_sum + info
        weighted_sum = weighted_sum + info @ est.pose.to(dtype).to(device)

    fused_cov = torch.linalg.inv(info_sum)
    fused_pose = fused_cov @ weighted_sum
    # Reuse the regularised per-modality information matrices already built
    # above for the diagnostic weights — re-calling ``inverse_variance_weights``
    # here would invert every covariance a second time (and, before the eps was
    # threaded through, without regularisation, so a singular sensor crashed the
    # weights path while the fusion itself succeeded).
    return FusedPose(
        pose=fused_pose,
        covariance=fused_cov,
        weights=_normalise_information_traces(infos).to(device),
    )
