"""Cycle-Bloch Digital-Twin Strategy (Integration plan idea 4).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §4.

Closes the loop  ``x → encode → params → simulate → x' → encode → params'``
so the network is forced to produce *parameter maps consistent under the
Bloch simulator*.  Equivalent to a CycleGAN pattern but with the
generator replaced by a fixed (analytic) digital twin.

Wired components:

- Forward Bloch decoder via :mod:`spectramr.models.blocks.spgr_signal`.
- Two cycle-consistency terms:
    1. Image-level: ``‖x − decode(encode(simulate(encode(x))))‖₁``.
    2. Parameter-level: ``‖params − encode_again(decode(params))‖₁``.

The encoder/decoder networks themselves come from the parent
:class:`CycleBlochStrategy`.  This subclass adds the twin-loop
pipeline and the two extra loss terms.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.models.blocks.spgr_signal import spgr_signal, spin_echo_signal

from .cycle_bloch_strategy import CycleBlochStrategy
from .mixins.utils import pick_present


class CycleBlochDigitalTwinStrategy(CycleBlochStrategy):
    """Cycle-Bloch with explicit closed-form digital twin."""

    lambda_image_cycle: float = 1.0
    lambda_param_cycle: float = 0.5
    lambda_observed: float = 1.0

    @staticmethod
    def _simulate(params: dict[str, torch.Tensor], seq: dict[str, Any]) -> torch.Tensor:
        family = (seq.get("family") or "spin_echo").lower()
        TR = seq["TR"]
        TE = seq["TE"]
        M0 = params["M0"]
        T1 = params["T1"]
        T2 = params["T2"]
        if family == "spgr":
            alpha = seq.get("alpha_rad", 0.3)
            T2_star = params.get("T2_star", T2)
            return spgr_signal(M0, T1, T2_star, TR, TE, alpha)
        return spin_echo_signal(M0, T1, T2, TR, TE)

    def _re_encode(self, image: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Optional re-encoder hook.  Subclasses with an explicit
        encoder module can override this to close the parameter cycle.
        """
        encoder = getattr(self.env, "encoder", None) or getattr(self.env, "model", None)
        if encoder is None:
            return None
        try:
            params = encoder(image)
        except Exception:
            return None
        if isinstance(params, dict):
            return params
        return None

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Canonical signature (audit F1); the full batch dict comes from kwargs."""
        base = super()._compute_losses_impl(
            input_batch=input_batch,
            target_batch=target_batch,
            epoch=epoch,
            **kwargs,
        )
        batch = kwargs.get("batch")
        if not isinstance(batch, dict):
            return base

        params = batch.get("predicted_params")
        seq = batch.get("target_sequence")
        observed = pick_present(batch.get("observed_signal"), batch.get("digital_twin_signal"))
        x = pick_present(batch.get("source_contrast"), batch.get("input"))

        if params is None or seq is None:
            return base

        synth = self._simulate(params, seq)
        # Single zero-tensor allocation (the previous version assigned
        # twice — the audit flagged the first as dead code).
        loss_observed = torch.zeros((), device=synth.device)
        if observed is not None:
            loss_observed = self.lambda_observed * (synth - observed).abs().mean()
            base["loss_digital_twin"] = loss_observed

        image_cycle = torch.zeros((), device=synth.device)
        param_cycle = torch.zeros((), device=synth.device)
        if x is not None:
            params_again = self._re_encode(synth)
            if params_again is not None:
                src_seq = batch.get("source_sequence", seq)
                back = self._simulate(params_again, src_seq)
                image_cycle = self.lambda_image_cycle * (back - x).abs().mean()
                base["loss_image_cycle"] = image_cycle
                # parameter-level cycle
                if all(k in params for k in ("M0", "T1", "T2")) and all(
                    k in params_again for k in ("M0", "T1", "T2")
                ):
                    param_cycle = (
                        self.lambda_param_cycle
                        * sum(
                            (params[k] - params_again[k]).abs().mean() for k in ("M0", "T1", "T2")
                        )
                        / 3.0
                    )
                    base["loss_param_cycle"] = param_cycle

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + loss_observed + image_cycle + param_cycle
                break
        return base
