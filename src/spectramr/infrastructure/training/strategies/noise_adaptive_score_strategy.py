"""Noise-Adaptive Score-Based Reconstruction (ULF-PR-6 / II.3).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-6.

A σ-conditioned score SDE for low-SNR ULF: the network learns
``s_θ(x_t, σ_t)`` where ``σ_t`` is per-pixel from an SNR map.  Two
public entry points:

- :meth:`_compute_losses_impl` — denoising-score-matching loss with
  Karras-style ``λ(σ) = (σ² + σ_data²) / (σ · σ_data)²`` weighting,
  optionally re-weighted by an external SNR map ``batch["snr_map"]``.
- :meth:`reverse_sde_sample` — Euler–Maruyama VE-SDE solver for
  inference.  Not invoked by the training loop; provided so callers
  can run conditional sampling outside the trainer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .reconstruction import ReconstructionTrainingStrategy


class NoiseAdaptiveScoreStrategy(ReconstructionTrainingStrategy):
    """σ-conditioned VE-SDE score-matching strategy."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5

    @staticmethod
    def _karras_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
        return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2

    def _sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        log_sig = torch.empty(batch_size, device=device).uniform_(
            torch.log(torch.tensor(self.sigma_min)).item(),
            torch.log(torch.tensor(self.sigma_max)).item(),
        )
        return log_sig.exp()

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

        x0 = batch.get("target")
        score_pred = batch.get("score_prediction")
        x_noisy = batch.get("noisy_input")
        sigma = batch.get("sigma")
        snr_map = batch.get("snr_map")

        if x0 is None or score_pred is None or x_noisy is None or sigma is None:
            return base

        eps = x_noisy - x0
        target = -eps / sigma.view(-1, 1, 1, 1).clamp_min(1e-6)
        sq = (score_pred - target).pow(2)

        weight = self._karras_weight(sigma.view(-1, 1, 1, 1), self.sigma_data)
        per_pixel = weight * sq
        if snr_map is not None:
            per_pixel = per_pixel * (snr_map**2).clamp_min(1e-6)
        dsm = per_pixel.mean()

        base["loss_score_dsm"] = dsm
        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + dsm
                break
        return base

    @torch.no_grad()
    def reverse_sde_sample(
        self,
        score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        n_steps: int = 50,
        device: torch.device | str = "cpu",
        snr_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler–Maruyama reverse-time VE-SDE sampling."""
        device = torch.device(device)
        sigmas = torch.exp(
            torch.linspace(
                float(torch.log(torch.tensor(self.sigma_max))),
                float(torch.log(torch.tensor(self.sigma_min))),
                n_steps + 1,
                device=device,
            )
        )
        x = torch.randn(*shape, device=device) * sigmas[0]
        for k in range(n_steps):
            sigma_now, sigma_next = sigmas[k], sigmas[k + 1]
            score = score_fn(x, sigma_now.expand(shape[0]))
            d_sigma2 = sigma_now**2 - sigma_next**2
            drift = -d_sigma2 * score
            noise = torch.randn_like(x) * torch.sqrt(d_sigma2.clamp_min(0.0))
            if snr_map is not None:
                noise = noise * snr_map
            x = x + drift + noise
        return x
