"""Tracer-kinetic parameter mapping — the trainable backing for ``mri_perfusion``.

The model maps a measured DCE concentration-time curve to the extended-Tofts
parameter maps ``(Ktrans, ve, vp)``. The primary objective is **self-supervised**:
``ToftsResidualLoss`` pushes the maps back through the forward kinetic model and
matches the measured curve, so no ground-truth parameter maps are needed. That is
what makes perfusion trainable on real DCE data without labels — and it is why
``extended_tofts_forward`` had to be vectorised first: it runs every step.

This strategy is also the ``SignalModelRegistry``'s production reader: it
resolves ``data.perfusion.kinetic_model`` through the registry and checks the
resolved model's parameter contract before calling it. That is what makes the
key a live dispatch knob rather than documentation (CLAUDE.md #15) — and it is
why ``mri_perfusion`` is LIVE without a ``forward_operator``. The regime's
physics is a nonlinear ``parameters -> curve`` map with no adjoint; it is a
signal model, not a linear operator, and the ledger's LIVE rule now asks for
"a registered forward model" rather than specifically a linear one. Registering
a fake operator to satisfy the old rule would have been the facade the ledger
exists to catch (pitfall #16).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from mriforge.config.schemas.enums import Regime, Task
from mriforge.models.capabilities import StrategyCapabilities

from .reconstruction import ReconstructionTrainingStrategy


class PerfusionKineticMappingStrategy(ReconstructionTrainingStrategy):
    """Fit extended-Tofts ``(Ktrans, ve, vp)`` maps to a measured DCE curve.

    Batch contract (emitted by ``data.perfusion``):
        - ``input``: ``[B, T, H, W]`` measured concentration-time curve, mM
        - ``t_s``: ``[T]`` time axis, seconds, uniformly sampled
        - ``aif``: ``[T]`` measured AIF, only if ``aif_source == "measured"``
        - ``kinetic_maps``: ``[B, 3, H, W]`` optional supervised map target
    """

    #: The trainable backing for mri_perfusion. PARAMETER_MAPPING only — this
    #: strategy does no reconstruction, and the profile supports nothing else.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.PERFUSION}),
        tasks=frozenset({Task.PARAMETER_MAPPING}),
    )

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("perfusion_kinetic", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)

        perfusion = getattr(self.config.data, "perfusion", None)
        if perfusion is None or not perfusion.enabled:
            raise ValueError(
                "PerfusionKineticMappingStrategy requires `data.perfusion.enabled: "
                "true` — it supplies the time axis and the AIF source."
            )
        self._aif_source = perfusion.aif_source
        self._num_frames = int(perfusion.num_frames)
        self._temporal_resolution_s = float(perfusion.temporal_resolution_s)
        self._time_axis_checked = False

        # Resolve data.perfusion.kinetic_model through the SignalModelRegistry.
        # This is the knob's only reader: without it the key is documentation
        # (#15), and it is also the clause that makes mri_perfusion LIVE, so a
        # ledger claim and a runtime dispatch now resolve the SAME name from the
        # SAME registry rather than agreeing by coincidence.
        from mriforge.infrastructure.physics.signal_models.registry import (
            get_signal_model,
        )

        self._kinetic_model = get_signal_model(perfusion.kinetic_model)
        if self._kinetic_model.regime is not Regime.PERFUSION:
            raise ValueError(
                f"data.perfusion.kinetic_model={perfusion.kinetic_model!r} "
                f"resolves to a {self._kinetic_model.regime.value} signal model, "
                "not a perfusion one."
            )
        # Sharing a regime does NOT make two signal models interchangeable:
        # gamma_variate is PERFUSION physics too, but its signature is
        # (t_s, amplitude, t0_s, alpha, beta_s) — calling it with Tofts kinetics
        # would bind ktrans->amplitude and ve->t0_s and return a plausible curve.
        expected = ("ktrans", "ve", "vp")
        if self._kinetic_model.parameters != expected:
            raise ValueError(
                f"data.perfusion.kinetic_model={perfusion.kinetic_model!r} maps "
                f"from {self._kinetic_model.parameters}, but this strategy "
                f"predicts {expected}. The generator's 3 output maps would bind "
                "to the wrong physical parameters."
            )

        out_channels = int(getattr(self.config.model, "out_channels", 0))
        if out_channels != 3:
            raise ValueError(
                "PerfusionKineticMappingStrategy predicts 3 parameter maps "
                f"(Ktrans, ve, vp); got model.out_channels={out_channels}."
            )
        in_channels = int(getattr(self.config.model, "in_channels", 0))
        if in_channels != self._num_frames:
            raise ValueError(
                f"data.perfusion.num_frames={self._num_frames} but "
                f"model.in_channels={in_channels}. The generator consumes the T "
                "dynamic frames as channels, so a mismatch silently truncates or "
                "pads the curve."
            )

        cfg = getattr(self.config.training, "perfusion_kinetic", None)
        self._parameter_activation = (
            str(getattr(cfg, "parameter_activation", "softplus")) if cfg is not None else "softplus"
        )

        self._lambda_tofts = self._get_loss_weight("tofts_residual")
        self._lambda_aif = self._get_loss_weight("aif_consistency")
        self._lambda_box = self._get_loss_weight("perfusion_physiological_box")
        self._lambda_smooth = self._get_loss_weight("perfusion_map_smoothness")
        weights = (
            self._lambda_tofts,
            self._lambda_aif,
            self._lambda_box,
            self._lambda_smooth,
        )
        if max(weights) <= 0.0:
            raise ValueError(
                "No PERFUSION loss carries a non-zero weight, so this arm would "
                "train as a plain regressor while advertising tracer-kinetic "
                "physics (pitfall #16). Declare at least "
                "losses.physics.lambda_tofts_residual > 0."
            )
        if self._lambda_aif > 0.0:
            raise ValueError(
                "lambda_aif_consistency > 0 grades a LEARNED AIF against the "
                "population reference, but no registered model exposes an AIF "
                "head, so there is no aif_pred to grade (data.perfusion has no "
                "aif_source='learned' for the same reason). Set it to 0 until a "
                "model with an AIF head lands — see workflow_backlog.md."
            )

    def predict_parameter_maps(self, curve: torch.Tensor) -> torch.Tensor:
        """``[B, T, H, W]`` curve -> ACTIVATED ``[B, 3, H, W]`` (Ktrans, ve, vp).

        The single seam where the generator's raw output becomes physical
        parameter maps. Everything downstream — the kinetic refit, the smoothness
        prior, the supervised fold, ``_last_prediction`` and validation — must
        consume THIS tensor, never ``generator(curve)`` directly.

        Activating in only some of those places is how the maps that get GRADED
        stop being the maps that get FITTED. That is exactly what happened here:
        the softplus lived inside ``_split_parameter_maps``, whose local rebinding
        never escaped, so the arm trained on ``softplus(out)`` and was graded on
        ``out`` — with the default activation that is pre-activation logits,
        roughly half of them negative (pitfall #18).

        The softplus itself is load-bearing, not cosmetic. The extended-Tofts
        kernel is ``exp(-(Ktrans/ve) * t)``; ``ve`` is clamped positive inside the
        kinetics, so the sign of ``kep`` follows ``Ktrans``. A negative Ktrans
        flips the exponent positive and the curve overflows to ``inf`` — and an
        untrained generator emits negatives from step 0. Forcing ``Ktrans >= 0``
        keeps ``-kep*t <= 0`` and the kernel in ``(0, 1]``.
        """
        maps = self.env.generator(curve)
        if isinstance(maps, tuple):
            maps = maps[0]
        if self._parameter_activation == "softplus":
            maps = torch.nn.functional.softplus(maps)
        return maps

    @staticmethod
    def _split_parameter_maps(
        maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Activated ``[B, 3, H, W]`` -> ``(Ktrans, ve, vp)``, each ``[B, H, W]``.

        A pure split. The activation lives in :meth:`predict_parameter_maps` so it
        happens exactly once, on the tensor every consumer shares.
        """
        return maps[:, 0], maps[:, 1], maps[:, 2]

    def _check_time_axis(self, time_axis: torch.Tensor) -> None:
        """Assert the batch's time axis matches the declared acquisition.

        This is what makes ``data.perfusion.temporal_resolution_s`` and
        ``num_frames`` real knobs rather than documentation (CLAUDE.md #15). The
        time axis arrives on the batch, so without this check the config values
        would be inert and a config/data disagreement would pass silently — and
        silently is the only way it can pass: ``extended_tofts_forward`` happily
        integrates any uniform axis, so a t_s in the wrong units simply refits to
        different, wrong kinetics with a perfectly healthy-looking residual. The
        AIF makes it worse, since Parker's constants are in minutes.

        Checked ONCE, on the first batch — not per step. ``torch.allclose``
        returns a Python bool, which is a device->host transfer: running it every
        iteration would put a sync back in the training loop (CLAUDE.md #9),
        which is the very thing the vectorised ``extended_tofts_forward`` exists
        to remove. The time axis is a property of the acquisition, so one check
        is all it can ever need.
        """
        if self._time_axis_checked:
            return
        self._time_axis_checked = True
        if time_axis.shape[0] != self._num_frames:
            raise ValueError(
                f"data.perfusion.num_frames={self._num_frames} but the batch's "
                f"'t_s' has {time_axis.shape[0]} frames."
            )
        if self._num_frames < 2:
            return
        expected = torch.full_like(time_axis[1:], self._temporal_resolution_s)
        if not torch.allclose(time_axis.diff(), expected, rtol=1e-3, atol=1e-4):
            observed = float(time_axis.diff().mean())
            raise ValueError(
                "data.perfusion.temporal_resolution_s="
                f"{self._temporal_resolution_s} s does not match the batch's 't_s' "
                f"spacing (mean {observed:.4g} s, and it must be uniform). The "
                "kinetics would refit to a different, wrong answer with a "
                "healthy-looking residual — t_s is in SECONDS."
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
            # dict OR TrainingBatch — the canonical pipeline passes the latter.
            raise ValueError(
                "PerfusionKineticMappingStrategy requires a mapping batch "
                f"(dict/TrainingBatch) carrying the DCE series; got {type(batch)!r}."
            )

        curve = batch["input"]  # [B, T, H, W]
        time_axis = batch.get("t_s")
        if time_axis is None:
            raise ValueError(
                "PerfusionKineticMappingStrategy requires a 't_s' batch key (the "
                "[T] time axis in seconds; data.perfusion emits it). Assuming a "
                "default would silently change the fitted kinetics."
            )
        if time_axis.ndim != 1:
            raise ValueError(f"'t_s' must be a 1-D [T] time axis; got {tuple(time_axis.shape)}.")
        self._check_time_axis(time_axis)

        from mriforge.infrastructure.physics.signal_models.perfusion_kinetics import (
            parker_population_aif,
        )
        from mriforge.models.losses.perfusion_losses import (
            PerfusionMapSmoothnessLoss,
            PerfusionPhysiologicalBoxLoss,
            ToftsResidualLoss,
        )

        if self._aif_source == "measured":
            aif = batch.get("aif")
            if aif is None:
                raise ValueError(
                    "data.perfusion.aif_source='measured' but the batch carries "
                    "no 'aif' key. Substituting the population AIF would change "
                    "the physics without saying so (pitfall #9)."
                )
        else:
            aif = parker_population_aif(time_axis)

        maps = self.predict_parameter_maps(curve)
        ktrans, ve, vp = self._split_parameter_maps(maps)

        components: dict[str, torch.Tensor] = {}
        total = maps.new_zeros(())

        # EVERY call below is BY KEYWORD. ToftsResidualLoss.forward is
        # (t_s, aif, ktrans, ve, vp, measured_curve) — six positional tensors,
        # NOT (pred, target). Through ComposedLoss it would bind pred -> t_s and
        # target -> aif and raise a shape error that reads like a data bug.
        if self._lambda_tofts > 0.0:
            # [B, T, H, W] -> [B, H, W, T]: extended_tofts_forward broadcasts the
            # kinetic params over the leading dims and puts time LAST.
            measured = curve.permute(0, 2, 3, 1)
            loss_tofts = ToftsResidualLoss(forward_model=self._kinetic_model.fn)(
                t_s=time_axis,
                aif=aif,
                ktrans=ktrans,
                ve=ve,
                vp=vp,
                measured_curve=measured,
            )
            components["loss_tofts_residual"] = loss_tofts.detach()
            total = total + self._lambda_tofts * loss_tofts

        if self._lambda_box > 0.0:
            loss_box = PerfusionPhysiologicalBoxLoss()(ktrans=ktrans, ve=ve, vp=vp)
            components["loss_perfusion_physiological_box"] = loss_box.detach()
            total = total + self._lambda_box * loss_box

        if self._lambda_smooth > 0.0:
            loss_smooth = PerfusionMapSmoothnessLoss()(param_map=maps)
            components["loss_perfusion_map_smoothness"] = loss_smooth.detach()
            total = total + self._lambda_smooth * loss_smooth

        # Fold the declarative losses.image_losses block ONLY when a supervised
        # map target exists. Image losses grade a (pred, target) pair, and the
        # self-supervised DCE arm has no target — folding against the measured
        # CURVE would grade parameter maps against a concentration series
        # (pitfall #18).
        maps_target = batch.get("kinetic_maps")
        if maps_target is not None:
            folded = self._apply_builder_image_losses(maps, maps_target, components)
            if folded is not None:
                total = total + folded

        self._last_prediction = maps
        self._last_target = maps_target
        return {"g_total_loss": total, **components}

    def _validation_forward(
        self,
        input_batch: Any,
        batch_context: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the ACTIVATED kinetic maps — what the PERFUSION metrics grade.

        Two ways to get this wrong, both pitfall #18: returning the refit curve
        instead of the maps (cbf_rmse / att_mae / negative_voxels all score
        parameter maps), or returning ``generator(curve)`` RAW — under the default
        softplus that is pre-activation logits, so ``negative_voxels`` would report
        ~50% and every map metric would grade a tensor the fit never optimised.
        Hence :meth:`predict_parameter_maps`, never the generator directly.
        """
        del kwargs
        curve = (
            batch_context.get("input", input_batch)
            if hasattr(batch_context, "get")
            else input_batch
        )
        return self.predict_parameter_maps(curve)


__all__ = ["PerfusionKineticMappingStrategy"]
