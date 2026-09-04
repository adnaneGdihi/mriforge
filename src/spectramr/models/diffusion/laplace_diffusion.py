import torch
from torch.distributions.laplace import Laplace

from spectramr.models.registry import register_model

from .base_diffusion import Diffusion


@register_model(name="laplace_diffusion", training_mode="diffusion")
class LaplaceDiffusion(Diffusion):
    """Implements a diffusion process using Laplace noise instead of Gaussian noise.
    The Laplace distribution has heavier tails than the Gaussian distribution,
    which can sometimes be beneficial for modeling sharp features in images.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str = None,
    ):
        # Set device intelligently - use CUDA if available, otherwise CPU
        """__init__.

        Args:
            timesteps (int): Description.
            beta_schedule (str): Description.
            device (str): Description.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(timesteps, beta_schedule, device)
        # For a standard Laplace distribution with mean 0 and variance 1,
        # the scale parameter 'b' must be 1/sqrt(2).
        # Variance = 2 * scale^2. So, 1 = 2 * scale^2 => scale = 1/sqrt(2).
        scale_tensor = torch.tensor(2.0, device=device)
        self.laplace_scale = 1.0 / torch.sqrt(scale_tensor)

    def q_sample(self, x_start, t, noise=None):
        """Forward process: add standardized Laplace noise to the image."""
        if noise is None:
            # Create a Laplace distribution with loc=0 and scale=1/sqrt(2)
            # for unit variance
            laplace_dist = Laplace(
                torch.tensor(0.0, device=x_start.device),
                self.laplace_scale.to(x_start.device),
            )
            noise = laplace_dist.sample(x_start.shape)

        # Use the standard DDPM noising formula, but with our Laplace noise
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod,
            t,
            x_start.shape,
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type: str = "l1"):
        # For Laplace noise, L1 loss is a more natural fit than L2.
        # The base class already supports this, so we can just call it.
        """p_losses.

        Args:
            denoise_model (Any): Description.
            x_start (Any): Description.
            t (Any): Description.
            noise (Any): Description.
            loss_type (str): Description.
        Returns:
            Any: Description.
        """
        return super().p_losses(denoise_model, x_start, t, noise, loss_type)
