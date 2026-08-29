"""Conformal risk control on quantitative parameter maps (qCRC).

Lifts the Conformal Risk Control (CRC) procedure of Angelopoulos et al.
[1] onto the *geodesic* error functional of quantitative MRI parameter
maps (M0, T1, T2) on the Bloch relaxation manifold. Rather than
certifying coverage of pixel intensities, qCRC certifies coverage of the
parameter map itself: it returns a geodesic tolerance ``lambda_hat`` such
that the nested geodesic prediction set

    C_lambda(y) = { theta' : d_g(theta', theta_hat(y)) <= lambda }

has expected miscoverage at most ``alpha`` on a future scan, with the
distribution-free finite-sample CRC guarantee.

Mathematics
-----------
For a calibration cohort of ``n`` scans, let ``r_{i,v}`` be the geodesic
residual ``d_g(theta_i, theta_hat_i)`` at voxel ``v`` of scan ``i``. The
miscoverage loss of scan ``i`` at radius ``lambda`` is the exceedance
fraction ``ell_i(lambda) = mean_v 1[r_{i,v} > lambda]`` (bounded by
``B = 1``), which is monotone non-increasing in ``lambda`` because the
geodesic balls are nested (``lambda_1 <= lambda_2 => C_{lambda_1} subset
C_{lambda_2}`` under the affine-invariant metric). CRC then selects

    lambda_hat = inf { lambda : R_hat_n(lambda) <= alpha - (B - alpha) / n }

with ``R_hat_n(lambda) = mean_i ell_i(lambda)``, which guarantees
``E[ell(C_{lambda_hat}, theta_{n+1})] <= alpha`` distribution-free [1].
The trivial limit ``alpha = 0`` forces full coverage (``lambda_hat`` is
the largest residual); a perfect reconstruction certifies a zero radius.

Calibration size: the conservative slack ``(B - alpha) / n`` falls below
a tolerance ``eps`` once ``n >= (B - alpha) / eps`` (e.g. ``alpha = 0.1``,
``eps = 0.01`` needs ``n >= 90`` calibration scans).

References
----------
[1] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster,
    "Conformal Risk Control," *ICLR*, 2024, arXiv:2208.02814.
"""

from __future__ import annotations

import math

import torch

from mriforge.core.metrics.registry import register_metric
from mriforge.infrastructure.physics.manifolds import BlochRelaxationManifold


def conformal_risk_lambda(
    residuals: torch.Tensor,
    alpha: float,
    loss_bound: float = 1.0,
) -> float:
    """Calibrate the CRC geodesic radius ``lambda_hat``.

    Args:
        residuals: Non-negative nonconformity scores ``[n, V]`` — ``n``
            calibration scans, ``V`` voxels each (geodesic residuals).
        alpha: Target expected miscoverage in ``(0, 1]`` (``0`` ⇒ full
            coverage).
        loss_bound: Upper bound ``B`` on the per-scan miscoverage loss
            (``1.0`` for the exceedance-fraction loss).

    Returns:
        The smallest geodesic radius whose empirical risk satisfies the
        finite-sample CRC threshold ``alpha - (B - alpha) / n``.
    """
    if residuals.ndim != 2:
        raise ValueError(f"residuals must be [n, V], got shape {tuple(residuals.shape)}")
    n = residuals.shape[0]
    flat = residuals.reshape(-1)
    total = flat.numel()
    if total == 0:
        raise ValueError("residuals tensor is empty")

    alpha_prime = alpha - (loss_bound - alpha) / n
    if alpha_prime <= 0.0:
        # Cannot certify any miscoverage budget; the only safe radius
        # covers every calibration residual.
        return float(flat.max().item())

    # Allow at most ``k`` of the ``total`` residuals to exceed the radius.
    k = math.floor(alpha_prime * total)
    k = max(0, min(k, total - 1))
    sorted_desc = flat.sort(descending=True).values
    # The (k+1)-th largest value: at most ``k`` residuals are strictly
    # greater than it, so R_hat(lambda_hat) <= k / total <= alpha_prime.
    return float(sorted_desc[k].item())


def conformal_coverage(
    residuals: torch.Tensor,
    alpha: float,
    loss_bound: float = 1.0,
) -> float:
    """Realized coverage ``1 - R_hat(lambda_hat)`` at the calibrated radius.

    Guaranteed ``>= 1 - alpha`` on the calibration cohort by construction.
    """
    lam = conformal_risk_lambda(residuals, alpha, loss_bound)
    flat = residuals.reshape(-1)
    return float((flat <= lam).float().mean().item())


class _QCRCBase:
    """Shared geodesic-residual computation for the qCRC metrics."""

    def __init__(self, alpha: float = 0.1, loss_bound: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.loss_bound = float(loss_bound)
        self.manifold = BlochRelaxationManifold()

    def _residuals(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(prediction.shape)} vs target {tuple(target.shape)}"
            )
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError(
                f"expected (M0, T1, T2) maps [B, 3, H, W], got {tuple(prediction.shape)}"
            )
        b = prediction.shape[0]
        p = prediction.permute(0, 2, 3, 1).reshape(-1, 3)
        q = target.permute(0, 2, 3, 1).reshape(-1, 3)
        d = self.manifold.geodesic_distance(p, q)  # [B * H * W]
        return d.reshape(b, -1)


@register_metric("conformal_risk_control", aliases=["qCRC", "crc_radius"])
class ConformalRiskControlMetric(_QCRCBase):
    """Certified geodesic radius (lower is better) over a calibration cohort.

    The batch dimension is treated as the calibration cohort of ``n``
    scans; see module docstring for the calibration-size rule.
    """

    @property
    def name(self) -> str:
        return "conformal_risk_control"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> float:
        residuals = self._residuals(prediction, target)
        return conformal_risk_lambda(residuals, self.alpha, self.loss_bound)


@register_metric("qmap_conformal_coverage", aliases=["qCRC_coverage"])
class QMapConformalCoverageMetric(_QCRCBase):
    """Realized coverage at the CRC-calibrated radius (higher is better)."""

    @property
    def name(self) -> str:
        return "qmap_conformal_coverage"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> float:
        residuals = self._residuals(prediction, target)
        return conformal_coverage(residuals, self.alpha, self.loss_bound)


__all__ = [
    "ConformalRiskControlMetric",
    "QMapConformalCoverageMetric",
    "conformal_coverage",
    "conformal_risk_lambda",
]
