"""MRS quantification — the trainable backing for ``mri_spectroscopy``.

The model maps a measured free induction decay to per-voxel Lorentzian resonance
parameters ``(amplitude, frequency, linewidth, phase)``, and the amplitudes ARE
the metabolite concentrations. The objective is **self-supervised**:
``MRSFIDResidualLoss`` pushes the parameters back through the forward signal model
and matches the measured FID, so no ground-truth concentrations are needed — which
is what makes spectroscopy trainable on real MRS data, since real acquisitions
have no concentration labels. This is the AMARES/QUEST time-domain fit
(Vanhamme et al. 1997) with a network in place of the optimiser.

This is also a ``SignalModelRegistry`` production reader: it resolves
``data.spectroscopy.signal_model`` through the registry and checks the resolved
model's parameter contract before calling it, so the key is a live dispatch knob
rather than documentation (CLAUDE.md #15) and the ledger's LIVE claim and the
runtime dispatch resolve the same name from the same registry.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.models.capabilities import StrategyCapabilities

from .reconstruction import ReconstructionTrainingStrategy


class MRSQuantificationStrategy(ReconstructionTrainingStrategy):
    """Fit Lorentzian resonance parameters to a measured FID.

    Batch contract (emitted by ``data.spectroscopy``):
        - ``input``: ``[B, 2*T, H, W]`` measured FID, real and imag as channels
        - ``t_s``: ``[T]`` acquisition time axis, seconds, starting at 0
        - ``resonance_params``: ``[B, 4*M, H, W]`` optional supervised target
    """

    #: The trainable backing for mri_spectroscopy. PARAMETER_MAPPING only — this
    #: strategy reconstructs nothing, and the profile supports nothing else.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.SPECTROSCOPIC}),
        tasks=frozenset({Task.PARAMETER_MAPPING}),
    )

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("mrs_quantification", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)

        spectroscopy = getattr(self.config.data, "spectroscopy", None)
        if spectroscopy is None or not spectroscopy.enabled:
            raise ValueError(
                "MRSQuantificationStrategy requires `data.spectroscopy.enabled: "
                "true` — it supplies the time axis and the resonance count."
            )
        self._num_points = int(spectroscopy.num_points)
        self._dwell_time_s = float(spectroscopy.dwell_time_s)
        self._num_resonances = int(spectroscopy.num_resonances)
        self._time_axis_checked = False

        from spectramr.infrastructure.physics.signal_models.registry import (
            get_signal_model,
        )

        self._signal_model = get_signal_model(spectroscopy.signal_model)
        if self._signal_model.regime is not Regime.SPECTROSCOPIC:
            raise ValueError(
                f"data.spectroscopy.signal_model={spectroscopy.signal_model!r} "
                f"resolves to a {self._signal_model.regime.value} signal model, "
                "not a spectroscopic one."
            )
        expected_params = (
            "amplitudes",
            "frequencies_hz",
            "linewidths_hz",
            "phases_rad",
        )
        if self._signal_model.parameters != expected_params:
            raise ValueError(
                f"data.spectroscopy.signal_model={spectroscopy.signal_model!r} "
                f"maps from {self._signal_model.parameters}, but this strategy "
                f"predicts {expected_params}. The generator's output channels "
                "would bind to the wrong physical parameters."
            )

        in_channels = int(getattr(self.config.model, "in_channels", 0))
        if in_channels != 2 * self._num_points:
            raise ValueError(
                f"data.spectroscopy.num_points={self._num_points} means a complex "
                f"FID of 2*{self._num_points}={2 * self._num_points} real "
                f"channels, but model.in_channels={in_channels}. A mismatch "
                "silently truncates the FID, which shortens the effective "
                "acquisition and biases every linewidth."
            )
        out_channels = int(getattr(self.config.model, "out_channels", 0))
        if out_channels != 4 * self._num_resonances:
            raise ValueError(
                f"MRSQuantificationStrategy predicts 4 parameters per resonance "
                f"(amplitude, frequency, linewidth, phase), so "
                f"data.spectroscopy.num_resonances={self._num_resonances} needs "
                f"model.out_channels={4 * self._num_resonances}; got {out_channels}."
            )

        cfg = getattr(self.config.training, "mrs_quantification", None)
        self._parameter_activation = (
            str(getattr(cfg, "parameter_activation", "softplus")) if cfg is not None else "softplus"
        )

        self._lambda_fid = self._get_loss_weight("mrs_fid_residual")
        self._lambda_prior = self._get_loss_weight("mrs_prior_knowledge")
        if max(self._lambda_fid, self._lambda_prior) <= 0.0:
            raise ValueError(
                "No SPECTROSCOPIC loss carries a non-zero weight, so this arm "
                "would train as a plain regressor while advertising MRS "
                "quantification (pitfall #16). Declare at least "
                "losses.physics.lambda_mrs_fid_residual > 0."
            )

    def predict_resonance_parameters(
        self, fid_channels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, 2T, H, W]`` FID -> ACTIVATED ``(a, f, lw, phi)``, each ``[B, M, H, W]``.

        The single seam where the generator's raw output becomes physical
        resonance parameters. Everything downstream — the FID refit, the prior,
        ``_last_prediction`` and validation — must consume THESE tensors, never
        ``generator(...)`` directly. Activating in only some places is how the
        parameters that get GRADED stop being the ones that got FITTED.

        The activation is **per-parameter**, unlike perfusion's global softplus,
        because only two of the four are sign-constrained:

        - ``amplitudes`` and ``linewidths_hz`` -> softplus. A concentration cannot
          be negative, and a negative linewidth flips the FID's ``exp(-pi*lw*t)``
          into a growing exponential that overflows to ``inf`` — an untrained
          generator emits negatives from step 0, so this is load-bearing, not
          cosmetic.
        - ``frequencies_hz`` and ``phases_rad`` -> raw. Both are legitimately
          signed (a resonance below the carrier has a negative offset), and
          forcing them positive would make half the spectrum unreachable.
        """
        raw = self.env.generator(fid_channels)
        if isinstance(raw, tuple):
            raw = raw[0]
        m = self._num_resonances
        amplitudes = raw[:, 0:m]
        frequencies = raw[:, m : 2 * m]
        linewidths = raw[:, 2 * m : 3 * m]
        phases = raw[:, 3 * m : 4 * m]
        if self._parameter_activation == "softplus":
            amplitudes = torch.nn.functional.softplus(amplitudes)
            linewidths = torch.nn.functional.softplus(linewidths)
        return amplitudes, frequencies, linewidths, phases

    @staticmethod
    def _fid_channels_to_complex(fid_channels: torch.Tensor) -> torch.Tensor:
        """``[B, 2T, H, W]`` real/imag channels -> ``[B, H, W, T]`` complex.

        The trailing time axis matches what ``lorentzian_fid`` produces: it
        broadcasts the resonance parameters over the leading dims and puts time
        LAST.
        """
        if fid_channels.shape[1] % 2 != 0:
            raise ValueError(
                f"a complex FID needs an even channel count (real then imag); got "
                f"{fid_channels.shape[1]}."
            )
        n_t = fid_channels.shape[1] // 2
        real = fid_channels[:, :n_t]  # [B, T, H, W]
        imag = fid_channels[:, n_t:]
        return torch.complex(real, imag).permute(0, 2, 3, 1)  # [B, H, W, T]

    def _check_time_axis(self, time_axis: torch.Tensor) -> None:
        """Assert the batch's time axis matches the declared acquisition.

        This is what makes ``data.spectroscopy.dwell_time_s`` and ``num_points``
        real knobs rather than documentation (#15). The axis arrives on the batch,
        so without this the config values are inert and a config/data
        disagreement passes silently — and silently is the only way it can pass:
        ``lorentzian_fid`` integrates any uniform axis happily, so a dwell time in
        the wrong units simply refits to different, wrong frequencies and
        linewidths with a perfectly healthy residual.

        Checked ONCE, on the first batch. ``torch.allclose`` returns a Python
        bool, which is a device->host sync; running it every iteration would put
        a sync in the training loop (CLAUDE.md #9). The dwell time is a property
        of the acquisition, so one check is all it can ever need.
        """
        if self._time_axis_checked:
            return
        self._time_axis_checked = True
        if time_axis.shape[0] != self._num_points:
            raise ValueError(
                f"data.spectroscopy.num_points={self._num_points} but the batch's "
                f"'t_s' has {time_axis.shape[0]} samples."
            )
        if self._num_points < 2:
            return
        expected = torch.full_like(time_axis[1:], self._dwell_time_s)
        if not torch.allclose(time_axis.diff(), expected, rtol=1e-3, atol=1e-9):
            observed = float(time_axis.diff().mean())
            raise ValueError(
                f"data.spectroscopy.dwell_time_s={self._dwell_time_s} s does not "
                f"match the batch's 't_s' spacing (mean {observed:.4g} s, and it "
                "must be uniform). The spectral bandwidth would be wrong, so every "
                "fitted frequency and linewidth would be scaled — with a healthy "
                "looking residual. t_s is in SECONDS."
            )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del target_batch, epoch
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):
            raise ValueError(
                "MRSQuantificationStrategy requires a mapping batch "
                f"(dict/TrainingBatch) carrying the FID; got {type(batch)!r}."
            )

        fid_channels = batch["input"]  # [B, 2T, H, W]
        time_axis = batch.get("t_s")
        if time_axis is None:
            raise ValueError(
                "MRSQuantificationStrategy requires a 't_s' batch key (the [T] "
                "time axis in seconds; data.spectroscopy emits it). Assuming a "
                "default would silently change every fitted frequency."
            )
        if time_axis.ndim != 1:
            raise ValueError(f"'t_s' must be a 1-D [T] time axis; got {tuple(time_axis.shape)}.")
        self._check_time_axis(time_axis)

        from spectramr.models.losses.spectroscopy_losses import (
            MRSFIDResidualLoss,
            MRSPriorKnowledgeLoss,
        )

        amplitudes, frequencies, linewidths, phases = self.predict_resonance_parameters(
            fid_channels
        )

        components: dict[str, torch.Tensor] = {}
        total = amplitudes.new_zeros(())

        # EVERY call below is BY KEYWORD. MRSFIDResidualLoss.forward is
        # (t_s, amplitudes, frequencies_hz, linewidths_hz, phases_rad,
        # measured_fid) — six positional tensors, NOT (pred, target). Through
        # ComposedLoss it would bind pred -> t_s and target -> amplitudes and
        # raise a shape error that reads like a data bug.
        if self._lambda_fid > 0.0:
            measured = self._fid_channels_to_complex(fid_channels)  # [B, H, W, T]
            # [B, M, H, W] -> [B, H, W, M]: lorentzian_fid broadcasts the
            # resonance params over the leading dims and sums over the LAST one.
            loss_fid = MRSFIDResidualLoss(forward_model=self._signal_model.fn)(
                t_s=time_axis,
                amplitudes=amplitudes.permute(0, 2, 3, 1),
                frequencies_hz=frequencies.permute(0, 2, 3, 1),
                linewidths_hz=linewidths.permute(0, 2, 3, 1),
                phases_rad=phases.permute(0, 2, 3, 1),
                measured_fid=measured,
            )
            components["loss_mrs_fid_residual"] = loss_fid.detach()
            total = total + self._lambda_fid * loss_fid

        if self._lambda_prior > 0.0:
            loss_prior = MRSPriorKnowledgeLoss()(amplitudes=amplitudes, linewidths_hz=linewidths)
            components["loss_mrs_prior_knowledge"] = loss_prior.detach()
            total = total + self._lambda_prior * loss_prior

        prediction = torch.cat([amplitudes, frequencies, linewidths, phases], dim=1)

        # Fold the declarative losses.image_losses block ONLY when a supervised
        # parameter target exists. Image losses grade a (pred, target) pair, and
        # the self-supervised MRS arm has no target — folding against the FID
        # would grade resonance parameters against a time signal (pitfall #18).
        params_target = batch.get("resonance_params")
        if params_target is not None:
            folded = self._apply_builder_image_losses(prediction, params_target, components)
            if folded is not None:
                total = total + folded

        self._last_prediction = prediction
        self._last_target = params_target
        return {"g_total_loss": total, **components}

    def _validation_forward(
        self,
        input_batch: Any,
        batch_context: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the ACTIVATED resonance parameters — what the metrics grade.

        Returns the parameters rather than the refit FID: the spectroscopy
        metrics grade the acquisition's quality and the fit, not the time signal.
        Returning ``generator(...)`` raw would be pitfall #18 twice over — under
        the default softplus those are pre-activation logits, so roughly half the
        amplitudes and linewidths would be negative and every metric would grade a
        tensor the fit never optimised.
        """
        del kwargs
        fid_channels = (
            batch_context.get("input", input_batch)
            if hasattr(batch_context, "get")
            else input_batch
        )
        amplitudes, frequencies, linewidths, phases = self.predict_resonance_parameters(
            fid_channels
        )
        return torch.cat([amplitudes, frequencies, linewidths, phases], dim=1)


__all__ = ["MRSQuantificationStrategy"]
