"""Adversarial Robustness Suite (PR-18 / L5).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-18.

Wired components:

- :meth:`pgd_attack` — projected-gradient-descent attack with
  configurable L∞/L2 norm constraint and step count.
- :meth:`fgsm_attack` — single-step FGSM.
- :meth:`certified_radius` — randomized-smoothing certificate
  (Cohen-Rosenfeld-Kolter 2019 style) returning a lower bound on the
  L2 radius for which the smoothed predictor's *answer* is robust.
- :meth:`_compute_losses_impl` — adversarial training: clean (small
  weight) + adversarial (large weight), with optional certified-
  radius reporting.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from .mixins.utils import pick_present
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class AdversarialRobustnessStrategy(ReconstructionTrainingStrategy):
    """PGD/FGSM adversarial training + randomized-smoothing certificate."""

    epsilon: float = 0.01
    attack_steps: int = 5
    attack_norm: str = "linf"  # "linf" | "l2"
    sigma_smooth: float = 0.25
    n_smooth_samples: int = 32
    clean_weight: float = 0.1
    adv_weight: float = 0.9

    def fgsm_attack(
        self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        x_adv = x.detach().clone().requires_grad_(True)
        loss = loss_fn(x_adv)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        if self.attack_norm == "linf":
            return (x + self.epsilon * grad.sign()).detach()
        # L2
        flat = grad.flatten(1)
        norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
        step = (grad / norm.view(-1, *([1] * (grad.dim() - 1)))) * self.epsilon
        return (x + step).detach()

    def pgd_attack(
        self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        delta = torch.zeros_like(x, requires_grad=True)
        step_size = self.epsilon / max(1, self.attack_steps - 1)
        for _ in range(self.attack_steps):
            loss = loss_fn(x + delta)
            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
            if self.attack_norm == "linf":
                delta = (delta + step_size * grad.sign()).clamp(-self.epsilon, self.epsilon)
            else:
                step = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-9).view(
                    -1, *([1] * (grad.dim() - 1))
                )
                delta = delta + step_size * step
                # Project to L2 ball
                d_norm = delta.flatten(1).norm(dim=1).clamp_min(1e-9)
                proj = (self.epsilon / d_norm).clamp_max(1.0).view(-1, *([1] * (delta.dim() - 1)))
                delta = delta * proj
            delta = delta.detach().requires_grad_(True)
        return (x + delta).detach()

    @torch.no_grad()
    def certified_radius(
        self,
        predict: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        confidence: float = 0.95,
    ) -> torch.Tensor:
        """Randomized-smoothing certified-radius lower bound.

        Returns ``r`` per-sample such that ``arg max smoothed_predict``
        is guaranteed stable for ``‖δ‖₂ ≤ r``.
        """
        B = x.shape[0]
        votes = torch.zeros(B, device=x.device)
        for _ in range(self.n_smooth_samples):
            noisy = x + self.sigma_smooth * torch.randn_like(x)
            pred = predict(noisy)
            # Use a mean response as a soft vote (works for regressors).
            votes = votes + pred.flatten(1).mean(dim=1)
        p = (votes / self.n_smooth_samples).sigmoid().clamp(1e-3, 1 - 1e-3)
        # Use Phi^{-1} approximation via erfinv.
        return self.sigma_smooth * math.sqrt(2.0) * torch.erfinv(2 * p - 1)

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
        if not isinstance(batch, dict):
            return super()._compute_losses_impl(
                input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
            )
        gen = getattr(self.env, "generator", None)
        x = pick_present(batch.get("kspace"), batch.get("input"))
        target = batch.get("target")
        if gen is None or x is None or target is None:
            return super()._compute_losses_impl(
                input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
            )

        def loss_fn(z: torch.Tensor) -> torch.Tensor:
            return F.l1_loss(gen(z), target)

        adv_x = self.pgd_attack(x, loss_fn)
        clean = super()._compute_losses_impl(
            input_batch=x, target_batch=target, epoch=epoch, **kwargs
        )
        adv_batch = dict(batch)
        adv_batch["kspace"] = adv_x
        if "input" in batch:
            adv_batch["input"] = adv_x
        adv = super()._compute_losses_impl(
            input_batch=adv_x,
            target_batch=target,
            epoch=epoch,
            **{**kwargs, "batch": adv_batch},
        )

        merged: dict[str, torch.Tensor] = {}
        # The canonical grad-carrying total key in this repo is ``g_total_loss``
        # (base.py:811 reads it for backward); ReconstructionTrainingStrategy emits
        # ``g_total_loss`` + a ``loss`` alias, NOT ``loss_total``. Reading the wrong
        # key here left the merged dict with no grad-carrying total at all (the run
        # would raise "No g_total_loss" downstream), which also made the A-8.3
        # CertRob penalty inert (#16). Read the canonical key, write the canonical key.
        _total_keys = ("g_total_loss", "loss_total", "loss")
        clean_total = pick_present(*(clean.get(k) for k in _total_keys))
        adv_total = pick_present(*(adv.get(k) for k in _total_keys))
        if clean_total is not None and adv_total is not None:
            merged["g_total_loss"] = self.clean_weight * clean_total + self.adv_weight * adv_total
        _skip = {"g_total_loss", "loss_total", "loss"}
        for k, v in clean.items():
            if k not in _skip and isinstance(v, torch.Tensor):
                merged[f"clean_{k}"] = v.detach()
        for k, v in adv.items():
            if k not in _skip and isinstance(v, torch.Tensor):
                merged[f"adv_{k}"] = v.detach()
        # Diagnostic only — no gradient through certificate.
        try:
            with torch.no_grad():
                merged["certified_radius"] = self.certified_radius(gen, x).mean().detach()
        except Exception:
            pass
        return merged
