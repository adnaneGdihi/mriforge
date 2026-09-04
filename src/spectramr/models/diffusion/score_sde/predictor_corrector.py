"""Predictor-Corrector Sampler for Score-Based Models.

Implements PC sampling with optional physics guidance.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from .physics_guidance import BlochGuidance
from .score_network import ScoreNetwork
from .sde import SDE, VESDE, VPSDE


class PredictorCorrector:
    """Predictor-Corrector sampler for SDEs."""

    def __init__(
        self,
        sde: SDE,
        score_net: ScoreNetwork,
        corrector_steps: int = 1,
        snr: float = 0.16,
        guidance: BlochGuidance | None = None,
    ):
        """__init__.

        Args:
            sde (SDE): Description.
            score_net (ScoreNetwork): Description.
            corrector_steps (int): Description.
            snr (float): Description.
            guidance (BlochGuidance | None): Description.
        """
        self.sde = sde
        self.score_net = score_net
        self.corrector_steps = corrector_steps
        self.snr = snr
        self.guidance = guidance

    def get_score(
        self,
        x: Float[Tensor, "B C D H W"],
        t: Float[Tensor, B],
        y_obs: Float[Tensor, "B C D H W"] | None = None,
        physics_params: dict | None = None,
    ) -> Float[Tensor, "B C D H W"]:
        """Compute score with optional guidance."""

        # 1. Model score
        score = self.score_net(x, t)

        # 2. Add physics guidance if available
        if self.guidance is not None and y_obs is not None and physics_params is not None:
            # guidance needs x_0 estimate to map to tissue props
            # x_t -> x_0? For VE-SDE, x_0 \approx x_t / sigma?
            # Or we pass x_t directly if guidance handles it.
            # Our BlochGuidance assumes x is HF image estimate.
            # For accurate guidance, we should estimate x_0 via Tweedie's formula:
            # x_0 = x_t + sigma(t)^2 * score(x_t) (for VE-SDE approx)

            # For now, pass x_t directly as approximation or rely on guidance scaling
            # Correct approach: Tweedie denoising
            if isinstance(self.sde, VESDE):
                _, std = self.sde.marginal_prob(x, t)  # std is sigma(t)
                # score \approx -(x_t - x_0)/sigma^2  => x_0 \approx x_t + sigma^2 * score
                x_0_hat = x + (std**2) * score
            elif isinstance(self.sde, VPSDE):
                # x_0_hat = (x_t + (1-alpha) * score) / sqrt(alpha) ??
                # Standard DDPM parameterization is usually epsilon, here score.
                # Leave as x for now to avoid complexity in this step
                x_0_hat = x
            else:
                x_0_hat = x

            # Check SNR or signal strength to avoid applying gradient on pure noise
            # Typically guidance starts being effective when signal emerges

            grad = self.guidance(
                x_0_hat,
                y_obs,
                tr_vlf=physics_params["tr"],
                te_vlf=physics_params["te"],
                scale=physics_params.get("scale", 1.0),
            )
            score = score + grad

        return score

    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        y_obs: Float[Tensor, "B C D H W"] | None = None,
        physics_params: dict | None = None,
    ) -> Float[Tensor, "B C D H W"]:
        """Generate samples using PC sampler."""

        # 1. Initialize x_T from prior
        x = self.sde.prior_sampling(shape, device)
        timesteps = torch.linspace(self.sde.T, 1e-5, self.sde.N, device=device)
        dt = timesteps[0] - timesteps[1]  # step size (assuming uniform)

        # 2. Reverse diffusion
        for i, t_scalar in enumerate(timesteps):
            t_vec = torch.ones(shape[0], device=device) * t_scalar

            # Corrector (Langevin)
            for _ in range(self.corrector_steps):
                score = self.get_score(x, t_vec, y_obs, physics_params)

                # Step size for Langevin
                # alpha_i = \epsilon * sigma_i^2 / sigma_L^2 (general form)
                # NCSN uses fixed SNR

                # noise norm / score norm
                noise = torch.randn_like(x)
                grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

                step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2

                # x_{i+1} = x_i + step_size * score + sqrt(2 * step_size) * z
                x = x + step_size * score + torch.sqrt(2 * step_size) * noise

            # Predictor (Euler-Maruyama for Reverse SDE)
            # dx = [f(x,t) - g(t)^2 score] dt + g(t) dw
            # Here we solve reverse SDE:
            # dx = [ f(x,t) - g(t)^2 score ] dt_neg + g(t) dw_reverse (if doing SDE solve)
            # OR simple Euler discretisation of ODE probability flow?
            # Standard: Reverse SDE

            score = self.get_score(x, t_vec, y_obs, physics_params)
            drift, diffusion = self.sde.sde(x, t_vec)

            # Reverse drift: f - g^2 * score
            reverse_drift = drift - (diffusion**2) * score

            # x_{t-1} = x_t - reverse_drift * dt + diffusion * sqrt(dt) * z
            # Note: dt is positive magnitude here, so we subtract drift corresponding to positive time?
            # reverse time differential dt_bar < 0.
            # SDE: dx = f dt + g dw
            # Rev: dx = [f - g^2 score] (-dt) + g d_w_bar
            # x_{t-dt} = x_t - [f - g^2 score] * dt + g * sqrt(dt) * z

            x_mean = x - reverse_drift * dt

            # Usually we don't add noise in predictor if using PC sampler
            # because Corrector adds noise. But formally Reverse SDE has noise.
            # Strategy: Deterministic step (Probability Flow ODE) for predictor is common.
            # x = x_mean

            # Let's use Reverse SDE with noise
            z = torch.randn_like(x)
            x = x_mean + diffusion * torch.sqrt(dt) * z

        return x
