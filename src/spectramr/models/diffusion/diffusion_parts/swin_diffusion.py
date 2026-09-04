import torch
import torch.nn.functional as F
from torch import nn

# --- Refactored Imports ---
# Use centralized, robust components from diffusion_parts and unet_parts
try:
    from .score_based_diffusion import ScoreBasedDiffusion
    from .unet_architectures import SwinHybridUNet
except ImportError:
    # Fallback for standalone execution or different project structure
    from .score_based_diffusion import ScoreBasedDiffusion
    from .unet_architectures import SwinHybridUNet


class SwinScoreDiffusion(ScoreBasedDiffusion):
    """A score-based diffusion process using Gaussian noise and Langevin dynamics,
    specifically for the SwinDiffusionUNet.
    """

    def __init__(self, *args, **kwargs):
        """__init__."""
        super().__init__(*args, **kwargs)
        # Remove degrees_of_freedom parameter as we're using Gaussian noise

    def sample_noise(self, shape, device=None):
        """Generate Gaussian noise for the diffusion process."""
        if device is None:
            device = self.device
        return torch.randn(shape, device=device)

    def q_sample(self, x_start, t, noise=None):
        """Add Gaussian noise to clean images at time step t."""
        if noise is None:
            noise = self.sample_noise(x_start.shape, device=x_start.device)

        # Standard Gaussian noise scaling
        noise_scale = torch.sqrt(t.float().view(-1, 1, 1, 1) / self.timesteps).to(
            self.device,
        )
        noisy_image = x_start + noise * noise_scale
        return noisy_image, noise

    def compute_loss(self, model, x_start, t, cond=None, **kwargs):
        """Compute score matching loss for training."""
        noisy_image, noise = self.q_sample(x_start, t)

        predicted_score = model(noisy_image, t)

        true_score = -noise

        loss = F.mse_loss(predicted_score, true_score)
        return loss

    def p_sample_loop(
        self,
        model,
        shape,
        cond=None,
        guidance_scale=7.5,
        steps=None,
        eta=1.0,
    ):
        """Generate samples using Langevin dynamics with Gaussian noise.
        Supports Classifier-Free Guidance (CFG) when cond is provided.
        """
        if steps is None:
            steps = self.timesteps

        device = self.device

        x = self.sample_noise(shape, device=device)

        step_size = 1.0 / steps
        step_size_tensor = torch.tensor(step_size, device=device)

        # Check if model supports conditioning for CFG
        model_supports_cond = False
        try:
            import inspect

            sig = inspect.signature(model.forward)
            model_supports_cond = "cond" in sig.parameters
        except (AttributeError, TypeError):
            model_supports_cond = False

        for i in reversed(range(steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)

            with torch.no_grad():
                # Apply Classifier-Free Guidance if supported
                if cond is not None and guidance_scale > 1.0 and model_supports_cond:
                    # Get unconditional score
                    uncond_score = model(x, t, cond=None)
                    # Get conditional score
                    cond_score = model(x, t, cond=cond)
                    # Apply CFG: uncond + guidance_scale * (cond - uncond)
                    score = uncond_score + guidance_scale * (cond_score - uncond_score)
                # Standard prediction (unconditional or model doesn't
                # support cond)
                elif model_supports_cond and cond is not None:
                    score = model(x, t, cond=cond)
                else:
                    score = model(x, t)

            if i > 0:
                noise = self.sample_noise(x.shape, device=device)
                x = x + step_size * score + eta * torch.sqrt(step_size_tensor) * noise
            else:
                x = x + step_size * score

        return x


def create_swin_diffusion_model(
    in_channels: int = 1,
    out_channels: int = 1,
    time_dim: int = 256,
    **kwargs,
) -> tuple[ScoreBasedDiffusion, nn.Module]:
    """Factory function to create a Swin-based diffusion model and process
    with Gaussian noise and Langevin dynamics sampling.

    Returns:
        Tuple of (generator, discriminator) for compatibility with GAN framework
        The discriminator is None for diffusion models

    """
    # Create the Swin diffusion model (remove unsupported layer norm
    # parameters)
    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["use_layer_norm_unet"]
    }  # Keep layer norm params

    device = filtered_kwargs.pop(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    # Filter out model-specific parameters that the diffusion process doesn't
    # need
    diffusion_kwargs = {
        k: v
        for k, v in filtered_kwargs.items()
        if k
        not in [
            "image_size",
            "patch_size",
            "window_size",
            "embed_dim",
            "depths",
            "num_heads",
        ]
    }

    # Create the diffusion process
    diffusion_process = SwinScoreDiffusion(device=device, **diffusion_kwargs)

    # Create the Swin U-Net model
    model = SwinHybridUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        time_dim=time_dim,
        **filtered_kwargs,
    ).to(device)

    return diffusion_process, model


# Export for use in model factory
__all__ = ["SwinScoreDiffusion", "create_swin_diffusion_model"]
