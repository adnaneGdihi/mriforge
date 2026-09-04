"""Spiral trajectory-recovery training strategy (exp_vf_35 G4).

Supervises the model's estimated trajectory deviation ``Δk̂`` against the
*measured* deviation ``Δk_true = trajectory_measured − trajectory_nominal`` (the
field-camera measurement kasper uniquely provides). This is the only
differentiable training path: ``Δk̂ = G_θ(k_nominal)`` is pure-torch (conv1d +
cumsum) so ``‖Δk̂ − Δk_true‖`` trains θ, whereas a through-NUFFT image objective
cannot reach θ (torchkbnufft does not pass gradients to the trajectory — wiring
a sharpness objective to train θ would be an inert mechanism, pitfall #16).

Validation grades ``Δk̂`` against ``Δk_true`` with the guarded
``k_space_trajectory_rmse`` metric (``_score_trajectory``): the guard's shape
check returns NaN for a Cartesian-vs-spiral mismatch, so a wrong-parametrisation
estimate is never silently graded.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.nn.functional import mse_loss

from spectramr.core.metrics.k_space_trajectory_rmse import KSpaceTrajectoryRMSE
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)

__all__ = ["TrajectoryReconstructionStrategy"]


class TrajectoryReconstructionStrategy(BaseTrainingStrategy):
    """Supervise a spiral ``Δk̂`` against the measured trajectory deviation."""

    def _setup_strategy_specific_components(self) -> None:
        cfg = getattr(self.config.training, "trajectory_recon", None)
        # Read + validate + stamp the knobs (pitfall #15).
        self._read_measured = bool(getattr(cfg, "read_measured_trajectory", True))
        self._trajectory_units = str(getattr(cfg, "trajectory_units", "cycles_per_fov"))
        self._self_supervised_sharpness = bool(getattr(cfg, "self_supervised_sharpness", False))
        self._sharpness_weight = float(getattr(cfg, "sharpness_weight", 0.0))
        if self._self_supervised_sharpness:
            # The self-supervised sharpness objective needs a differentiable
            # through-NUFFT recon, which torchkbnufft does not provide (no
            # trajectory autograd). Raise rather than silently no-op (pitfall #15);
            # the supervised Δk loss is the working mechanism.
            raise ValueError(
                "trajectory_recon.self_supervised_sharpness=true is not wired: the "
                "NUFFT backend does not pass gradients to the trajectory, so a "
                "through-NUFFT sharpness objective cannot train θ. Use the "
                "supervised Δk loss (self_supervised_sharpness=false)."
            )
        self._traj_metric = KSpaceTrajectoryRMSE()
        logger.info(
            "[TrajectoryRecon] read_measured=%s units=%s sharpness=%s",
            self._read_measured,
            self._trajectory_units,
            self._self_supervised_sharpness,
        )

    def _deviation(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(Δk̂, Δk_true) from the batch's nominal + measured trajectories, or None."""
        if not isinstance(batch, dict):
            return None
        k_nom = batch.get("trajectory_nominal")
        k_meas = batch.get("trajectory_measured")
        if k_nom is None or k_meas is None:
            return None
        k_nom = self._norm_traj(k_nom)
        k_meas = self._norm_traj(k_meas)
        dk_true = k_meas - k_nom
        dk_hat = self.generator_model(k_nom)
        return dk_hat, dk_true

    def _norm_traj(self, t: torch.Tensor) -> torch.Tensor:
        """Coerce a collated trajectory to ``[B, N, 2]`` (drop leading singleton dims)."""
        t = self._to_device(t.float())
        while t.dim() > 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() > 3:  # [B, n_acq, N, 2] → fold arms into batch
            t = t.flatten(0, t.dim() - 3)
        return t

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(kwargs.get("batch"), kwargs)
        dev = self._deviation(batch)
        if dev is None:
            raise ValueError(
                "trajectory_recon requires trajectory_nominal + trajectory_measured "
                "in the batch (ismrmrd_kspace.emit_paired_trajectory=true)."
            )
        dk_hat, dk_true = dev
        loss = mse_loss(dk_hat, dk_true)
        self._last_visual_pred = dk_hat.detach()
        self._last_visual_target = dk_true.detach()
        return {"g_total_loss": loss, "traj_mse": loss.detach()}

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        trajectory_measured: torch.Tensor | None = None,
        trajectory_nominal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        if trajectory_measured is None or trajectory_nominal is None:
            return {"val_traj_reference_real": 0.0}
        with torch.no_grad():
            k_nom = self._norm_traj(trajectory_nominal)
            k_meas = self._norm_traj(trajectory_measured)
            dk_true = k_meas - k_nom
            dk_hat = self.generator_model(k_nom)
        result: dict[str, float] = {"val_traj_reference_real": 1.0}
        result.update(self._score_trajectory(dk_hat, dk_true))
        return result

    def _score_trajectory(
        self, estimate: torch.Tensor | None, reference: torch.Tensor | None
    ) -> dict[str, float]:
        """Grade Δk̂ against Δk_true with the guarded trajectory metric.

        Self-gates on both tensors present; the metric returns NaN (and we omit
        the key) on a parametrisation mismatch — never a silent grade (pitfall #16).
        """
        if estimate is None or reference is None:
            return {}
        try:
            rmse = self._traj_metric(estimate, reference)
            if rmse != rmse:  # NaN → guard refused (shape/param mismatch)
                logger.debug("[TrajectoryRecon] traj metric skipped (not comparable)")
                return {}
            endpoint = float((estimate[..., -1, :] - reference[..., -1, :]).norm(dim=-1).mean())
            return {
                "val_traj_rmse": float(rmse),
                "val_traj_endpoint_err": endpoint,
                "val_traj_reference_real": 1.0,
            }
        except Exception as exc:  # diagnostics must never break validation
            logger.debug("[TrajectoryRecon] trajectory scoring skipped: %s", exc)
            return {}
