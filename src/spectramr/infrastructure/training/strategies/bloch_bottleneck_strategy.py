"""Bloch-Bottleneck Multi-Contrast Strategy (Integration plan idea 3).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §3.

Encoder → quantitative parameter triple ``(T1, T2, M0)`` → Bloch
SPGR/spin-echo simulator → arbitrary-contrast image.  The bottleneck
*is* the physical parameter space, forcing the encoder to extract a
genuinely quantitative-MRI representation rather than a free latent.

Wired components:

- Quantitative parameter maps come from the parent
  :class:`OneShotMultiParameterStrategy` (already produces T1, T2, PD).
- The forward Bloch decoder is :func:`spgr_signal` /
  :func:`spin_echo_signal` from :mod:`spectramr.models.blocks.spgr_signal`,
  selectable per-batch via ``batch["pulse_family"]``.
- Loss = parent multi-task + L1(decoded synthetic, target_contrast)
  + identity ridge ``‖decode(encode(x), source_TR_TE_FA) − x‖₁``.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.models.blocks.spgr_signal import spgr_signal, spin_echo_signal

from .multi_param_mapping_strategy import OneShotMultiParameterStrategy


class BlochBottleneckStrategy(OneShotMultiParameterStrategy):
    """Forced-Bloch bottleneck with arbitrary-contrast decoding."""

    lambda_decode: float = 0.5
    lambda_identity: float = 0.1

    def _decode(
        self,
        params: dict[str, torch.Tensor],
        seq: dict[str, Any],
    ) -> torch.Tensor:
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

        params = batch.get("predicted_params")
        target = batch.get("target_contrast")
        target_seq = batch.get("target_sequence")
        if params is None or target is None or target_seq is None:
            return base

        decoded = self._decode(params, target_seq)
        decode_l = self.lambda_decode * (decoded - target).abs().mean()
        base["loss_bloch_decode"] = decode_l

        identity_l = (
            base.new_zeros(())
            if hasattr(base, "new_zeros")
            else torch.zeros((), device=decoded.device)
        )
        identity_l = torch.zeros((), device=decoded.device)
        src_seq = batch.get("source_sequence")
        src = batch.get("source_contrast")
        if src is not None and src_seq is not None:
            decoded_src = self._decode(params, src_seq)
            identity_l = self.lambda_identity * (decoded_src - src).abs().mean()
            base["loss_bloch_identity"] = identity_l

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + decode_l + identity_l
                break
        return base
