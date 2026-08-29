"""MRF-DiPh Sampler
==================

Physics-Constrained Diffusion Sampler that modifies the DDIM reverse
process to project samples onto the Bloch manifold at each step.

Key Modification:
Standard DDIM: x̂₀ = denoise(xₜ)
MRF-DiPh:      x̂₀ = λ·Bloch(match(x̂₀)) + (1-λ)·x̂₀

This ensures that hallucinated textures are "corrected" to be
physically realizable MRI signals.
"""

from collections.abc import Callable

import torch

from mriforge.models.diffusion.base_diffusion import Diffusion
from mriforge.models.diffusion.ddim_sampler import DDIMSampler

from .bloch_projection import BlendedProjector, BlochProjector, create_bloch_projector


class MRFDiPhSampler(DDIMSampler):
    """MRF-DiPh: Physics-Constrained DDIM Sampler.

    Extends DDIM with Bloch manifold projection to prevent hallucinations.
    At each reverse diffusion step:
    1. Standard DDIM denoising predicts x̂₀
    2. Dictionary matching estimates tissue parameters (T1, T2, PD)
    3. Bloch simulation generates physically valid signal
    4. Blend physics projection with diffusion prediction
    5. Continue reverse sampling

    Args:
        diffusion: Base diffusion process
        projector: Bloch projector for manifold projection
        lambda_physics: Blend weight (0 = pure diffusion, 1 = pure physics)
        TR: Repetition time for Bloch simulation (ms)
        TE: Echo time for Bloch simulation (ms)
        alpha: Flip angle for GRE sequences (degrees)
        eta: DDIM stochasticity (0 = deterministic)
        projection_interval: Apply projection every N steps (1 = every step)
    """

    def __init__(
        self,
        diffusion: Diffusion,
        projector: BlochProjector,
        lambda_physics: float = 0.5,
        TR: float = 500.0,
        TE: float = 15.0,
        alpha: float = 90.0,
        eta: float = 0.0,
        use_clipped_model_output: bool = True,
        projection_interval: int = 1,
        dc_weight: float = 0.0,
    ):
        # Initialize parent DDIM sampler
        """__init__.

        Args:
            diffusion (Diffusion): Description.
            projector (BlochProjector): Description.
            lambda_physics (float): Description.
            TR (float): Description.
            TE (float): Description.
            alpha (float): Description.
            eta (float): Description.
            use_clipped_model_output (bool): Description.
            projection_interval (int): Description.
            dc_weight (float): Description.
        """
        super().__init__(
            diffusion=diffusion,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            dc_weight=dc_weight,
        )

        self.projector = projector
        self.lambda_physics = lambda_physics
        self.TR = TR
        self.TE = TE
        self.alpha = alpha
        self.projection_interval = projection_interval

        # Create blended projector for convenient blending
        self.blended_projector = BlendedProjector(
            projector=projector,
            lambda_physics=lambda_physics,
            learnable=False,
        )

    def _physics_correct(self, x: torch.Tensor) -> torch.Tensor:
        """Apply physics correction via Bloch projection.

        Args:
            x: Denoised estimate [B, C, H, W]

        Returns:
            Physics-corrected estimate [B, C, H, W]
        """
        return self.blended_projector(x, self.TR, self.TE, self.alpha)

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        shape: torch.Size,
        num_inference_steps: int = 50,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        return_intermediates: bool = False,
        return_tissue_maps: bool = False,
        measured_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple:
        """Generate samples using MRF-DiPh sampling.

        Args:
            model: Denoising model
            shape: Output shape (batch_size, channels, height, width)
            num_inference_steps: Number of sampling steps
            cond: Conditional input
            guidance_scale: Classifier-free guidance scale
            generator: Random generator
            return_intermediates: Return intermediate steps
            return_tissue_maps: Return final tissue parameter maps
            measured_kspace: k-space data for data consistency
            mask: Sampling mask for data consistency

        Returns:
            Generated samples, optionally with intermediates and tissue maps
        """
        device = self.diffusion.device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device, generator=generator)

        intermediates = [x.clone()] if return_intermediates else None

        # Get DDIM timesteps
        timesteps = self._get_ddim_timesteps(num_inference_steps)

        for i, t in enumerate(timesteps):
            t_index = t.item()
            t_tensor = torch.full(
                (batch_size,),
                t_index,
                device=device,
                dtype=torch.long,
            )

            # Get noise prediction
            if guidance_scale != 1.0 and cond is not None:
                try:
                    noise_uncond = model(x, t_tensor, cond=None)
                except TypeError:
                    noise_uncond = model(x, t_tensor)
                try:
                    noise_cond = model(x, t_tensor, cond=cond)
                except TypeError:
                    noise_cond = model(x, t_tensor)
                predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                try:
                    predicted_noise = model(x, t_tensor, cond=cond)
                except TypeError:
                    predicted_noise = model(x, t_tensor)

            predicted_noise = predicted_noise.to(device)

            if self.use_clipped_model_output:
                predicted_noise = torch.clamp(predicted_noise, -1.0, 1.0)

            # === KEY MODIFICATION: Physics Projection ===
            # Predict x_0 from noise prediction
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t_tensor, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape
            )

            pred_x0 = (
                x - sqrt_one_minus_alphas_cumprod_t * predicted_noise
            ) / sqrt_alphas_cumprod_t
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Apply physics correction at specified intervals
            if i % self.projection_interval == 0:
                pred_x0 = self._physics_correct(pred_x0)

            # Continue with DDIM step using corrected x0
            alphas_cumprod_t = self._extract(self.alphas_cumprod, t_tensor, x.shape)
            alphas_cumprod_prev_t = self._extract(self.alphas_cumprod_prev, t_tensor, x.shape)
            ddim_sigma_t = self._extract(self.ddim_sigmas, t_tensor, x.shape)

            # Direction pointing to x_t
            # Recompute noise estimate from corrected x0
            corrected_noise = (
                x - sqrt_alphas_cumprod_t * pred_x0
            ) / sqrt_one_minus_alphas_cumprod_t

            dir_xt = torch.sqrt(1.0 - alphas_cumprod_prev_t - ddim_sigma_t**2) * corrected_noise

            # Compute x_{t-1}
            x = torch.sqrt(alphas_cumprod_prev_t) * pred_x0 + dir_xt

            # Add noise if eta > 0
            if self.eta > 0:
                noise = torch.randn_like(x)
                x = x + ddim_sigma_t * noise

            # Apply data consistency if available
            if self.data_consistency is not None and measured_kspace is not None:
                x = self.data_consistency(x, measured_kspace, mask)

            if return_intermediates:
                intermediates.append(x.clone())

        # Build output
        result = x

        if return_tissue_maps:
            tissue_maps = self.projector.get_tissue_maps(x, self.TR, self.TE, self.alpha)
            if return_intermediates:
                return result, intermediates, tissue_maps
            return result, tissue_maps

        if return_intermediates:
            return result, intermediates

        return result


def create_mrf_diph_sampler(
    diffusion: Diffusion,
    sequence_type: str = "SE",
    lambda_physics: float = 0.5,
    TR: float = 500.0,
    TE: float = 15.0,
    alpha: float = 90.0,
    dictionary_resolution: str = "medium",
    eta: float = 0.0,
    projection_interval: int = 1,
) -> MRFDiPhSampler:
    """Factory function to create MRF-DiPh sampler.

    Args:
        diffusion: Base diffusion process
        sequence_type: "SE" (Spin Echo) or "GRE" (Gradient Recalled Echo)
        lambda_physics: Blend weight for physics projection
        TR: Repetition time (ms)
        TE: Echo time (ms)
        alpha: Flip angle (degrees)
        dictionary_resolution: "low", "medium", or "high"
        eta: DDIM stochasticity
        projection_interval: Apply projection every N steps

    Returns:
        Configured MRF-DiPh sampler
    """
    projector = create_bloch_projector(
        sequence_type=sequence_type,
        dictionary_resolution=dictionary_resolution,
        smooth_matching=True,
        device=diffusion.device,
    )

    return MRFDiPhSampler(
        diffusion=diffusion,
        projector=projector,
        lambda_physics=lambda_physics,
        TR=TR,
        TE=TE,
        alpha=alpha,
        eta=eta,
        projection_interval=projection_interval,
    )
