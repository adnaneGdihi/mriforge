"""PILOT — Physics-Informed Learned Optimal Trajectories (PR-11 / M3).

Continuous-trajectory variant of LOUPE: we learn (k_x, k_y) coordinates
under hardware-feasibility constraints (slew rate, max gradient).

Reference: Weiss et al. "PILOT: Physics-Informed Learned Optimal
Trajectories for Accelerated MRI" (MRM 2021).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from torch import nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.models.capabilities import StrategyCapabilities

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class PILOTStrategy(ReconstructionTrainingStrategy):
    """Joint trajectory + reconstruction strategy with hardware constraints."""

    #: Declared rather than inherited. Inheriting ``ReconstructionTrainingStrategy``
    #: also inherited its ``{RECONSTRUCTION}`` tag, so the one strategy in the
    #: repo that genuinely *designs an acquisition* never said so — which is why
    #: ``ACQUISITION_DESIGN`` read as a ``Task`` no profile supported. It is
    #: BOTH: PILOT learns the (k_x, k_y) trajectory *and* reconstructs from it,
    #: and dropping RECONSTRUCTION here would un-tag the half that already works.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.STRUCTURAL}),
        tasks=frozenset({Task.RECONSTRUCTION, Task.ACQUISITION_DESIGN}),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._codesign = self.config.acquisition.codesign
        if not self._codesign.enabled:
            logger.warning(
                "PILOTStrategy instantiated but acquisition.codesign.enabled is False — "
                "trajectory will stay frozen at the YAML-declared init."
            )

        self._trajectory: nn.Parameter | None = None
        # ``model.model_kwargs`` is a ``dict[str, Any]`` — getattr() never finds
        # these as attributes, so the historical getattr() form silently ignored
        # any YAML override and always used the defaults (pitfall #15: an
        # exposed knob that is never read). Use dict access, validate, and stamp.
        model_kwargs = self.config.model.model_kwargs or {}
        n_shots = int(model_kwargs.get("n_shots", 16))
        samples_per_shot = int(model_kwargs.get("samples_per_shot", 256))
        if n_shots <= 0:
            raise ValueError(
                f"PILOTStrategy: model.model_kwargs.n_shots must be > 0, got {n_shots}."
            )
        if samples_per_shot <= 0:
            raise ValueError(
                "PILOTStrategy: model.model_kwargs.samples_per_shot must be > 0, "
                f"got {samples_per_shot}."
            )
        # Stamp the resolved knob values into run provenance (pitfall #15).
        self._n_shots = n_shots
        self._samples_per_shot = samples_per_shot
        logger.info(
            "PILOTStrategy trajectory init: n_shots=%d, samples_per_shot=%d (total points=%d)",
            n_shots,
            samples_per_shot,
            n_shots * samples_per_shot,
        )
        # Initial spiral trajectory in (k_x, k_y) space, normalised to [-0.5, 0.5].
        t = torch.linspace(0, 1, samples_per_shot)
        init_traj = []
        for i in range(n_shots):
            phi = 2 * torch.pi * i / n_shots
            r = 0.5 * t
            kx = r * torch.cos(2 * torch.pi * t * 8 + phi)
            ky = r * torch.sin(2 * torch.pi * t * 8 + phi)
            init_traj.append(torch.stack([kx, ky], dim=-1))
        self._trajectory = nn.Parameter(torch.cat(init_traj, dim=0).to(self.device))

        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            opt.add_param_group({"params": [self._trajectory], "lr": 1e-4})

    def _feasibility_penalty(self) -> torch.Tensor:
        """Slew-rate + gradient-amplitude + acoustic-comfort penalty.

        Weiss 2021 Eq. 8 + acoustic-comfort term (Eq. 12) that
        penalises spectral content of the slew waveform near the
        scanner's resonant frequency band (~1 kHz).
        """
        traj = self._trajectory
        if traj is None or self._samples_per_shot < 3:
            return torch.tensor(0.0, device=self.device)

        # The flat ``[n_shots * samples_per_shot, 2]`` trajectory must be
        # reshaped to ``[n_shots, samples_per_shot, 2]`` before differencing:
        # otherwise ``torch.diff`` over the flat axis differences shot i's last
        # sample against shot i+1's first sample as if they were physically
        # adjacent, injecting a spurious slew/gradient spike at every shot
        # boundary. Reshape and difference along the per-shot (samples) axis so
        # penalties are computed strictly within each shot.
        traj = traj.reshape(self._n_shots, self._samples_per_shot, 2)

        dt = self._codesign.constraints.gradient_dwell_us * 1e-6
        grad = torch.diff(traj, dim=1) / dt
        slew = torch.diff(grad, dim=1) / dt

        gmax = self._codesign.constraints.gradient_max_mT_per_m / 1000.0
        smax = self._codesign.constraints.slew_rate_max_T_per_m_per_s

        g_overshoot = torch.relu(grad.abs() - gmax)
        s_overshoot = torch.relu(slew.abs() - smax)
        feas = g_overshoot.pow(2).sum() + s_overshoot.pow(2).sum()

        # Acoustic-comfort term: high-frequency slew energy. Compute the
        # spectrum along the per-shot (samples) axis (dim=1) so the rfft never
        # straddles a shot boundary. This is a spectral op on a real-valued
        # waveform, not a complex MRI k-space round-trip — the fft_ops rule does
        # not apply (see physics.md scope clarification).
        if slew.shape[1] >= 8:
            slew_fft = torch.fft.rfft(slew, dim=1)
            # Penalise the upper third of the slew spectrum.
            cutoff = slew_fft.shape[1] // 3
            acoustic = slew_fft[:, cutoff:].abs().pow(2).mean()
            feas = feas + 0.1 * acoustic
        return feas * self._codesign.feasibility_lambda

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        #
        # The learned trajectory must be CONSUMED by the recon forward, not just
        # penalised — otherwise it receives no data gradient and never influences
        # the output. The parent (ReconstructionMixin) reads
        # ``batch_context["trajectory"]`` (sourced from the batch dict under
        # ``kwargs["batch"]``), so we inject ``self._trajectory`` there. This
        # routes the data/recon loss gradient back into the trajectory params,
        # mirroring GIRFAwareStrategy's waveform→trajectory injection.
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        forward_kwargs = kwargs
        if self._trajectory is not None and isinstance(batch, dict):
            patched = dict(batch)
            patched["trajectory"] = self._trajectory
            forward_kwargs = {**kwargs, "batch": patched}

        base_losses = super()._compute_losses_impl(
            input_batch=input_batch,
            target_batch=target_batch,
            epoch=epoch,
            **forward_kwargs,
        )
        if not self._codesign.enabled:
            return base_losses

        feas = self._feasibility_penalty()
        base_losses["loss_gradient_feasibility"] = feas
        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base_losses:
                base_losses[total_key] = base_losses[total_key] + feas
                break
        return base_losses
