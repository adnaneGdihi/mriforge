"""k-space trajectory RMSE (cycles/FOV) — the headline metric for exp_vf_35.

Grades a model's estimated trajectory deviation ``Δk̂`` against the
*independently measured* deviation ``Δk_true = k_measured - k_nominal`` (e.g.
kasper's field-camera measurement). The claim is k-space-domain accuracy, so the
metric must refuse to grade anything that is not the *same trajectory
parametrization*: a Cartesian per-readout-line ``Δk [B, 2, H]`` graded against a
spiral ``Δk(t) [n_samp, 2]`` is the diff_trajectory_opt facade (pitfall #16). The
shared guard's shape check is what blocks it; a mismatch (or a missing tensor)
returns ``NaN`` (skip), never a silent grade.

The static twin of this runtime guard is the model's declared
``trajectory_parametrization`` capability (e.g. ``"spiral"``), which a Tier-1
check pairs with this metric once the spiral estimator model lands.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.core.metrics._guards import field_comparability
from spectramr.core.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("k_space_trajectory_rmse", aliases=["traj_rmse"], direction="lower")
class KSpaceTrajectoryRMSE:
    """RMS error (cycles/FOV) between an estimated and measured ``Δk``.

    Lower is better. Element-wise RMS over the readout coordinates; returns
    ``NaN`` (skip) when the estimate and reference are not the same trajectory
    parametrization (shape mismatch) or a tensor is missing.
    """

    @property
    def name(self) -> str:
        return "k_space_trajectory_rmse"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        verdict = field_comparability(
            prediction,
            target,
            estimate_kind="trajectory",
            reference_kind="trajectory",
            estimate_units="cycles_per_fov",
            reference_units="cycles_per_fov",
        )
        if not verdict.ok:
            logger.debug("k_space_trajectory_rmse skipped: %s", verdict.reason)
            return float("nan")
        assert prediction is not None and target is not None  # guaranteed by verdict.ok
        diff = prediction.float() - target.float()
        return float(torch.sqrt(torch.mean(diff * diff)))
