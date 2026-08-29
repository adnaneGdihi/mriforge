"""Inverse-Bloch Phase Extraction (ULF-PR-13 / II.10).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-13.

ULF phase carries B0 inhomogeneity, motion, and concomitant-field
contributions all mixed together. This strategy predicts (T1, T2, PD) from
the multi-contrast magnitude (via the parent multi-parameter heads) and adds a
Tikhonov smoothness prior on a ``phase_residual`` map: once the slowly-varying
background phase is removed, the residual should be spatially smooth.

The ``phase_residual`` key is produced by
:class:`~mriforge.data.transforms.phase_residual.PhaseResidualTransform`
(background-removed measured phase) — add it to the arm's transform pipeline.
When the key is absent the smoothness term is simply not added.

NOTE (audit 2026-06): the full *Bloch-SPGR forward-model* residual (forward
-simulate the SPGR phase from the predicted tissue + sequence parameters and
subtract it from the measured phase) is not implemented — it needs sequence
metadata and a differentiable SPGR phase model, and remains roadmap work (see
``docs/strategy_audit_design_triage_2026_06.rst``). The strategy currently
regularises the background-removed phase residual described above, NOT a
Bloch-model residual.

Inherits from :class:`OneShotMultiParameterStrategy` so we get the
multi-parameter heads + Kendall weighting "for free".
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .multi_param_mapping_strategy import OneShotMultiParameterStrategy

logger = logging.getLogger(__name__)


class InverseBlochPhaseStrategy(OneShotMultiParameterStrategy):
    """Multi-parameter mapping + phase-residual smoothness regulariser.

    Naming honesty: "inverse-Bloch" is a legacy label. This strategy does NOT
    invert a Bloch/SPGR phase forward model; it adds a Tikhonov (first-order)
    smoothness prior on a background-removed ``phase_residual`` map on top of the
    inherited multi-parameter heads. See the module docstring for why the true
    inverse-Bloch residual is unimplemented (roadmap).
    """

    # Phase-residual smoothness weight; overridable via the typed
    # training.inverse_bloch_phase block (previously a hardcoded 0.01).
    lambda_phase_smooth: float = 0.01

    def _setup_strategy_specific_components(self) -> None:
        super()._setup_strategy_specific_components()
        cfg = getattr(self.config.training, "inverse_bloch_phase", None)
        if cfg is not None:
            self.lambda_phase_smooth = float(cfg.lambda_phase_smooth)
        logger.info(
            "InverseBlochPhaseStrategy (source=%s): lambda_phase_smooth=%.4g",
            "config" if cfg is not None else "defaults",
            self.lambda_phase_smooth,
        )

    def _phase_residual_smoothness(self, phase: torch.Tensor) -> torch.Tensor:
        """L2 on first-order spatial differences of the residual phase."""
        if phase.dim() < 3:
            return torch.tensor(0.0, device=self.device)
        dy = phase[..., 1:, :] - phase[..., :-1, :]
        dx = phase[..., :, 1:] - phase[..., :, :-1]
        return dy.pow(2).mean() + dx.pow(2).mean()

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )

        has_residual = isinstance(batch, dict) and "phase_residual" in batch
        if not has_residual and self.lambda_phase_smooth > 0.0:
            # The Tikhonov phase-smoothness prior is this strategy's defining
            # term, and it was silently skipped whenever the key was absent --
            # which was ALWAYS, because PhaseResidualTransform had no way to be
            # constructed (it is not built by the transform builder and there
            # was no registry to declare it through). The weight knob was
            # validated, read and INFO-logged over a structurally unreachable
            # term: pitfall #15 wrapped around pitfall #16.
            raise ValueError(
                "inverse_bloch_phase declares lambda_phase_smooth="
                f"{self.lambda_phase_smooth:g} but the batch carries no "
                "'phase_residual' key, so the phase-smoothness prior cannot be "
                "computed and would be silently dropped. Declare the producer:\n"
                "  data:\n"
                "    processing:\n"
                "      transforms:\n"
                "        - name: phase_residual\n"
                "Or set lambda_phase_smooth: 0.0 to run without the prior."
            )
        if has_residual:
            smooth = self._phase_residual_smoothness(batch["phase_residual"])
            base["loss_phase_smooth"] = self.lambda_phase_smooth * smooth
            for total_key in ("loss_total",):
                if total_key in base:
                    base[total_key] = base[total_key] + base["loss_phase_smooth"]
                    break
        return base
