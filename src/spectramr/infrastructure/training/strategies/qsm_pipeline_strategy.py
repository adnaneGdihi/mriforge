"""Full QSM pipeline (PR-13 / M5).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-13.

End-to-end Quantitative-Susceptibility-Mapping pipeline:

    raw multi-echo phase
      → :func:`compute_field_map`         (echo combination → field B0 map)
      → :func:`laplacian_phase_unwrap`    (Schofield-Zhu unwrapping)
      → :func:`vsharp_background_removal` (V-SHARP harmonic removal)
      → :func:`tkd_dipole_inversion`      (TKD dipole inversion → χ)

All four primitives live in :mod:`spectramr.infrastructure.physics.qsm` and
are differentiable w.r.t. their inputs (FFT-based).  This strategy
chains them and computes per-stage losses against any references
provided in the batch.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.physics.qsm import (
    compute_field_map,
    laplacian_phase_unwrap,
    tkd_dipole_inversion,
    vsharp_background_removal,
)

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class QSMPipelineStrategy(ReconstructionTrainingStrategy):
    """Differentiable end-to-end QSM pipeline."""

    lambda_field: float = 0.1
    lambda_unwrap: float = 0.1
    lambda_background: float = 0.1
    lambda_chi: float = 1.0
    lambda_tv: float = 1e-3

    def _setup_strategy_specific_components(self) -> None:
        """Resolve QSM loss weights from the typed ``training.qsm_pipeline``
        block (pitfall #15: read + validate + stamp).

        When the block is present its Pydantic-validated fields override the
        class-attribute defaults; when absent the defaults stand. Either way the
        resolved weights are stamped into the run log so a config edit is
        observable (previously a YAML ``training.qsm_pipeline.lambda_chi`` had no
        effect — the weights were hardcoded).
        """
        super()._setup_strategy_specific_components()
        cfg = getattr(self.config.training, "qsm_pipeline", None)
        if cfg is not None:
            self.lambda_field = float(cfg.lambda_field)
            self.lambda_unwrap = float(cfg.lambda_unwrap)
            self.lambda_background = float(cfg.lambda_background)
            self.lambda_chi = float(cfg.lambda_chi)
            self.lambda_tv = float(cfg.lambda_tv)
        logger.info(
            "QSMPipelineStrategy loss weights (source=%s): lambda_field=%.4g, "
            "lambda_unwrap=%.4g, lambda_background=%.4g, lambda_chi=%.4g, "
            "lambda_tv=%.4g",
            "config" if cfg is not None else "defaults",
            self.lambda_field,
            self.lambda_unwrap,
            self.lambda_background,
            self.lambda_chi,
            self.lambda_tv,
        )

    @staticmethod
    def _isotropic_tv(x: torch.Tensor) -> torch.Tensor:
        gx = x[..., 1:, :] - x[..., :-1, :]
        gy = x[..., :, 1:] - x[..., :, :-1]
        return gx.pow(2).mean() + gy.pow(2).mean()

    def run_pipeline(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Execute the four-stage pipeline; returns intermediate tensors."""
        out: dict[str, torch.Tensor] = {}

        if "phase_echoes" in batch and "TE" in batch:
            # compute_field_map needs two single-echo phase maps + two scalar
            # echo times; phase_echoes stacks echoes along dim=1 and TE holds
            # the echo times. The old 2-arg call passed the whole stack as
            # phase_te1 and the TE tensor as phase_te2 -> TypeError.
            phase = batch["phase_echoes"]
            te = batch["TE"]
            te_flat = te.reshape(-1) if torch.is_tensor(te) else torch.as_tensor(te).reshape(-1)
            if phase.shape[1] < 2 or te_flat.numel() < 2:
                return out
            field = compute_field_map(
                phase[:, 0:1], phase[:, 1:2], float(te_flat[0]), float(te_flat[1])
            )
            out["field_map"] = field
        elif "field_map" in batch:
            field = batch["field_map"]
            out["field_map"] = field
        elif "wrapped_phase" in batch:
            field = batch["wrapped_phase"]
            out["field_map"] = field
        else:
            return out

        mask = batch.get("brain_mask")
        if mask is None:
            mask = torch.ones_like(field)
        unwrapped = laplacian_phase_unwrap(field, mask=mask)
        out["unwrapped"] = unwrapped

        bg_removed = vsharp_background_removal(unwrapped, mask=mask)
        out["background_removed"] = bg_removed

        chi = tkd_dipole_inversion(bg_removed, mask)
        out["chi"] = chi
        return out

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict):
            return base
        # No silent fallback (pitfall #10): a failure inside the differentiable
        # QSM pipeline must propagate so audit/smoke fail loudly, instead of
        # masking the error behind a constant, grad-free
        # ``loss_qsm_pipeline_error`` that silently degrades QSM to a no-op recon.
        stages = self.run_pipeline(batch)
        if not stages:
            return base

        chi_target = batch.get("chi_target")
        chi_loss = stages["chi"].new_zeros(())
        if chi_target is not None:
            chi_loss = self.lambda_chi * (stages["chi"] - chi_target).abs().mean()
            base["loss_chi"] = chi_loss

        tv = self.lambda_tv * self._isotropic_tv(stages["chi"])
        base["loss_chi_tv"] = tv

        # Per-stage references (optional)
        for stage, weight, key in [
            ("field_map", self.lambda_field, "field_target"),
            ("unwrapped", self.lambda_unwrap, "unwrapped_target"),
            ("background_removed", self.lambda_background, "background_target"),
        ]:
            ref = batch.get(key)
            if ref is not None and stage in stages:
                base[f"loss_{stage}"] = weight * (stages[stage] - ref).abs().mean()

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                add = chi_loss + tv
                for stage in ("field_map", "unwrapped", "background_removed"):
                    if f"loss_{stage}" in base:
                        add = add + base[f"loss_{stage}"]
                base[total_key] = base[total_key] + add
                break
        return base
