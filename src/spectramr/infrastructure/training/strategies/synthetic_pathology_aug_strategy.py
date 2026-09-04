"""Synthetic Pathology Augmentation Strategy (Integration plan idea 7).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §7.

On-the-fly synthetic lesion injection plus a region-weighted loss.  We
no longer rely on an external transform — the lesion sampler is in
this strategy:

- :class:`LesionSampler` produces (B, 1, H, W) Gaussian-blob masks
  with random center / radius / intensity per sample.
- The strategy composites the lesion onto the input *and* the target,
  with a per-batch random toggle so a fraction of samples remain clean.
- Loss = parent recon fidelity + lesion-region-weighted L1 + an
  auxiliary lesion-classification head loss when ``batch["lesion_logits"]``
  is supplied.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixins.utils import pick_present
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class LesionSampler(nn.Module):
    """Random Gaussian-blob lesion sampler — differentiable."""

    def __init__(
        self, max_radius: float = 0.15, intensity_range: tuple[float, float] = (0.3, 1.2)
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.intensity_range = intensity_range

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = image.shape
        device = image.device
        cx = torch.rand(B, device=device) * 0.6 - 0.3
        cy = torch.rand(B, device=device) * 0.6 - 0.3
        radius = torch.rand(B, device=device) * (self.max_radius - 0.03) + 0.03
        intensity = (
            torch.rand(B, device=device) * (self.intensity_range[1] - self.intensity_range[0])
            + self.intensity_range[0]
        )

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        cx_b = cx.view(-1, 1, 1, 1)
        cy_b = cy.view(-1, 1, 1, 1)
        r_b = radius.view(-1, 1, 1, 1)
        d2 = (xx - cx_b) ** 2 + (yy - cy_b) ** 2
        mask = torch.exp(-d2 / r_b.pow(2).clamp_min(1e-4))
        # Composite onto image
        composite = image + intensity.view(-1, 1, 1, 1) * mask
        return composite, (mask > 0.4).float()


class SyntheticPathologyAugStrategy(ReconstructionTrainingStrategy):
    """Recon with on-the-fly synthetic lesion augmentation."""

    p_augment: float = 0.5
    lambda_lesion_l1: float = 0.5
    lambda_lesion_cls: float = 0.1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sampler: LesionSampler | None = None
        # Typed sub-block (pitfall #15: read + validate + stamp). Previously the
        # augmentation probability + lesion-loss weights were hardcoded class
        # attributes that no YAML could override.
        cfg = getattr(self.config.training, "synthetic_pathology_aug", None)
        if cfg is not None:
            self.p_augment = float(cfg.p_augment)
            self.lambda_lesion_l1 = float(cfg.lambda_lesion_l1)
            self.lambda_lesion_cls = float(cfg.lambda_lesion_cls)
        logger.info(
            "SyntheticPathologyAugStrategy (source=%s): p_augment=%.3g, "
            "lambda_lesion_l1=%.4g, lambda_lesion_cls=%.4g",
            "config" if cfg is not None else "defaults",
            self.p_augment,
            self.lambda_lesion_l1,
            self.lambda_lesion_cls,
        )

    def _ensure_sampler(self, device: torch.device) -> LesionSampler:
        if self._sampler is None:
            self._sampler = LesionSampler().to(device)
        return self._sampler

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if isinstance(batch, dict) and "input" in batch and "target" in batch:
            sampler = self._ensure_sampler(batch["input"].device)
            B = batch["input"].shape[0]
            apply = torch.rand(B, device=batch["input"].device) < self.p_augment

            # Sample lesions for each item; gate via apply mask.
            lesion_input, lesion_mask_pred = sampler(batch["input"])
            lesion_target, lesion_mask_target = sampler(batch["target"])
            gate = apply.view(-1, 1, 1, 1).float()
            patched = dict(batch)
            patched["input"] = batch["input"] * (1 - gate) + lesion_input * gate
            patched["target"] = batch["target"] * (1 - gate) + lesion_target * gate
            patched["lesion_mask"] = lesion_mask_target * gate
            # Forward the COMPOSITED lesion tensors as the actual model
            # input/target (the parent builds lr/hr from these positionals).
            # The old code forwarded the original input_batch/target_batch, so
            # the sampled lesions never reached the model or the recon loss —
            # the augmentation was a no-op.
            base = super()._compute_losses_impl(
                input_batch=patched["input"],
                target_batch=patched["target"],
                epoch=epoch,
                **{**kwargs, "batch": patched},
            )
        else:
            base = super()._compute_losses_impl(
                input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
            )
            if not isinstance(batch, dict):
                return base
            patched = batch

        lesion_mask = patched.get("lesion_mask")
        # The model prediction is never written into the batch dict — read it from
        # the parent's prediction seam (set in
        # ReconstructionTrainingStrategy._compute_losses_impl). pred carries grad to
        # the generator (NOT detached), so the region-weighted term backprops.
        pred = getattr(self, "_last_prediction", None)
        # Prefer the parent's loss target (matches what pred was scored against);
        # fall back to the composited batch target. pick_present is the SSOT for
        # first-non-None when operands may be tensors.
        target = pick_present(getattr(self, "_last_target", None), patched.get("target"))
        if lesion_mask is not None and pred is not None and target is not None:
            weighted = ((pred - target).abs() * (1.0 + 4.0 * lesion_mask)).mean()
            base["loss_lesion_weighted"] = self.lambda_lesion_l1 * weighted

            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + base["loss_lesion_weighted"]
                    break
            else:
                raise KeyError(
                    "synthetic_pathology: no total-loss key (loss_total / "
                    "g_total_loss) present to fold in loss_lesion_weighted — the "
                    "lesion term would be silently dropped (audit 2026-06)."
                )

        # Optional auxiliary lesion-presence classifier.
        logits = patched.get("lesion_logits")
        if logits is not None and lesion_mask is not None:
            label = (lesion_mask.amax(dim=(-1, -2, -3)) > 0.5).float()
            cls = F.binary_cross_entropy_with_logits(logits.flatten(), label.flatten())
            base["loss_lesion_cls"] = self.lambda_lesion_cls * cls
            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + base["loss_lesion_cls"]
                    break
            else:
                raise KeyError(
                    "synthetic_pathology: no total-loss key (loss_total / "
                    "g_total_loss) present to fold in loss_lesion_cls — the "
                    "lesion-classifier term would be silently dropped."
                )
        return base
