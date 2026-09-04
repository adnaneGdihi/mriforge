"""Posterior Langevin Ensemble Sampler for Synthetic NEX Averaging.

Generates K independent posterior samples from a conditional score-based
diffusion model, then computes MMSE (mean) and epistemic uncertainty (var)
across the ensemble. This emulates hardware multi-NEX averaging at inference
time from a single noisy ULF acquisition.

Mathematical Foundation:
    Each sample k follows the reverse SDE:

        x^{(k)}_{t-dt} = x^{(k)}_t + dt·[f(x^{(k)}_t, t)
                          - g(t)²·∇_x log p_t(x_t | y)] + g(t)√dt·ω^{(k)}

    The MMSE estimate and epistemic variance are:

        x̂_NEX = (1/K) Σ_k x^{(k)}_0
        σ²_epistemic = Var(x^{(1..K)}_0)

    SNR improvement: √K (same as hardware averaging).

Supports:
    - Variance-Preserving (VP) and Variance-Exploding (VE) noise schedules
    - Optional data consistency projection per step (via PhysicsGuidedReverseSampler)
    - Configurable number of ensemble members and diffusion steps
    - Batched ensemble generation for GPU efficiency

References:
    - Kawar, B. et al. (2022). "Denoising Diffusion Restoration Models." ICLR.
    - Song, Y. et al. (2021). "Score-Based Generative Modeling through SDEs." ICLR.
    - Jalal, A. et al. (2021). "Robust Compressed Sensing MRI with Deep
      Generative Priors." NeurIPS.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import NamedTuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SDEType(str, Enum):
    """Noise schedule type for reverse SDE."""

    VP = "vp"  # Variance Preserving
    VE = "ve"  # Variance Exploding


class EnsembleResult(NamedTuple):
    """Output of posterior ensemble sampling."""

    mean: torch.Tensor
    """MMSE estimate (averaged over K samples) [B, C, H, W]."""
    variance: torch.Tensor
    """Epistemic uncertainty (per-pixel variance) [B, C, H, W]."""
    samples: torch.Tensor
    """Individual posterior samples [K, B, C, H, W]."""


class PosteriorLangevinEnsemble(nn.Module):
    """Posterior Langevin Ensemble for synthetic NEX averaging.

    Runs K independent stochastic reverse SDE chains conditioned on a
    noisy measurement y, then returns the MMSE (mean) and epistemic
    uncertainty (variance) across the ensemble.

    Args:
        score_model: Conditional score network s_θ(x_t, y, t).
            Expected signature: ``forward(x_t, y, t) -> score``.
        num_samples: Number of ensemble members K. SNR gain ≈ √K.
        num_steps: Number of reverse diffusion steps T.
        sde_type: Noise schedule type ('vp' or 've').
        beta_min: Minimum noise rate (VP schedule).
        beta_max: Maximum noise rate (VP schedule).
        sigma_min: Minimum noise level (VE schedule).
        sigma_max: Maximum noise level (VE schedule).
        dc_projector: Optional data consistency projector applied per step.
            Should accept (x_pred, measured_kspace, mask) → x_corrected.
        snr_clamp: Clamp signal-to-noise ratio for numerical stability.
        batched: If True, process all K samples in a single batch
            (faster but uses K× GPU memory).
    """

    def __init__(
        self,
        score_model: nn.Module,
        num_samples: int = 8,
        num_steps: int = 200,
        sde_type: str = "vp",
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        dc_projector: nn.Module | None = None,
        snr_clamp: float = 1e-5,
        batched: bool = False,
    ) -> None:
        super().__init__()
        self.score_model = score_model
        self.K = num_samples
        self.T = num_steps
        self.sde_type = SDEType(sde_type.lower())
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.dc_projector = dc_projector
        self.snr_clamp = snr_clamp
        self.batched = batched

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        """VP schedule: β(t) = β_min + t·(β_max - β_min)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        """VE schedule: σ(t) = σ_min·(σ_max/σ_min)^t."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def _drift_vp(
        self,
        x: torch.Tensor,
        score: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """VP reverse drift: f(x,t) - g²·score."""
        beta_t = self._beta(t)
        # Reshape for broadcasting
        while beta_t.ndim < x.ndim:
            beta_t = beta_t.unsqueeze(-1)
        drift = -0.5 * beta_t * x
        diffusion_sq = beta_t
        return drift - diffusion_sq * score

    def _diffusion_vp(self, t: torch.Tensor) -> torch.Tensor:
        """VP diffusion coefficient: g(t) = √β(t)."""
        return torch.sqrt(self._beta(t))

    def _drift_ve(
        self,
        x: torch.Tensor,
        score: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """VE reverse drift: -g²·score (no mean reversion)."""
        sigma_t = self._sigma(t)
        while sigma_t.ndim < x.ndim:
            sigma_t = sigma_t.unsqueeze(-1)
        log_ratio = math.log(self.sigma_max / self.sigma_min)
        g_sq = 2 * sigma_t**2 * log_ratio
        return -g_sq * score

    def _diffusion_ve(self, t: torch.Tensor) -> torch.Tensor:
        """VE diffusion coefficient."""
        sigma_t = self._sigma(t)
        log_ratio = math.log(self.sigma_max / self.sigma_min)
        return sigma_t * math.sqrt(2 * log_ratio)

    @torch.no_grad()
    def _sample_single(
        self,
        y: torch.Tensor,
        measured_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one reverse SDE chain from T=1 to T=0.

        Args:
            y: Conditioned measurement [B, C, H, W].
            measured_kspace: For DC projection [B, C, H, W].
            mask: Undersampling mask [B, 1, H, W].

        Returns:
            Single posterior sample [B, C, H, W].
        """
        device = y.device
        B = y.shape[0]

        # Initialize from prior
        x_t = torch.randn_like(y)

        time_steps = torch.linspace(1.0, 0.0, self.T + 1, device=device)
        dt = 1.0 / self.T

        for i in range(self.T):
            t = time_steps[i]
            t_batch = t.expand(B)

            # Score prediction
            score = self.score_model(x_t, y, t_batch)

            # Reverse SDE step
            if self.sde_type == SDEType.VP:
                drift = self._drift_vp(x_t, score, t_batch)
                g = self._diffusion_vp(t_batch)
            else:
                drift = self._drift_ve(x_t, score, t_batch)
                g = self._diffusion_ve(t_batch)

            # Reshape g for broadcasting
            while g.ndim < x_t.ndim:
                g = g.unsqueeze(-1)

            # Euler-Maruyama step (no noise at final step)
            noise = torch.randn_like(x_t) if i < self.T - 1 else 0.0
            x_t = x_t + drift * dt + g * math.sqrt(dt) * noise

            # Optional DC projection
            if self.dc_projector is not None and measured_kspace is not None and mask is not None:
                x_t = self.dc_projector(x_t, measured_kspace, mask)

        return x_t

    @torch.no_grad()
    def forward(
        self,
        y: torch.Tensor,
        measured_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        num_samples: int | None = None,
    ) -> EnsembleResult:
        """Generate K posterior samples and compute MMSE + uncertainty.

        Args:
            y: Conditioned measurement (noisy ULF image) [B, C, H, W].
            measured_kspace: Raw k-space for DC projection [B, C, H, W].
            mask: Undersampling mask [B, 1, H, W].
            num_samples: Override default K for this call.

        Returns:
            EnsembleResult with mean, variance, and individual samples.

        forward method for PosteriorLangevinEnsemble.

        Executes PyTorch tensor operations.

        Args:
            y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            num_samples (int | None): Expected input tensor.

        Returns:
            EnsembleResult: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        K = num_samples or self.K

        if self.batched:
            samples = self._sample_batched(y, measured_kspace, mask, K)
        else:
            samples_list = []
            for k in range(K):
                sample = self._sample_single(y, measured_kspace, mask)
                samples_list.append(sample)
            samples = torch.stack(samples_list, dim=0)  # [K, B, C, H, W]

        # MMSE estimator
        mean = samples.mean(dim=0)
        # Epistemic uncertainty
        variance = samples.var(dim=0)

        logger.info(
            "Posterior ensemble: K=%d, mean_snr_gain=%.2f×",
            K,
            math.sqrt(K),
        )

        return EnsembleResult(mean=mean, variance=variance, samples=samples)

    @torch.no_grad()
    def _sample_batched(
        self,
        y: torch.Tensor,
        measured_kspace: torch.Tensor | None,
        mask: torch.Tensor | None,
        K: int,
    ) -> torch.Tensor:
        """Generate K samples in a single batched forward pass.

        Repeats y along the batch dimension K times for GPU efficiency.
        """
        B = y.shape[0]
        # Expand y to [K*B, C, H, W]
        y_expanded = y.repeat(K, *([1] * (y.ndim - 1)))

        mk_expanded = None
        mask_expanded = None
        if measured_kspace is not None:
            mk_expanded = measured_kspace.repeat(K, *([1] * (measured_kspace.ndim - 1)))
        if mask is not None:
            mask_expanded = mask.repeat(K, *([1] * (mask.ndim - 1)))

        # Run single chain on expanded batch
        result = self._sample_single(y_expanded, mk_expanded, mask_expanded)

        # Reshape to [K, B, C, H, W]
        return result.reshape(K, B, *result.shape[1:])

    def extra_repr(self) -> str:
        return f"K={self.K}, T={self.T}, sde={self.sde_type.value}, batched={self.batched}"


__all__ = ["EnsembleResult", "PosteriorLangevinEnsemble", "SDEType"]
