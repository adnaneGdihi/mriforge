import numpy as np
import torch

from spectramr.config.schemas.enums import SchedulerTypes

from .base_diffusion import Diffusion

try:
    from scipy.stats import rice

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class RicianDiffusion(Diffusion):
    """Implements a diffusion process using Rician noise.

    The Rician distribution is controlled by a non-centrality parameter 'v'.
    When 'v' is large, the Rician distribution approximates a Gaussian
    distribution. This implementation uses a schedule for 'v' to make the
    noise more Gaussian at higher timesteps, which is crucial for the DDPM
    framework.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str = None,
        v_min: float = 0.1,
        v_max: float = 10.0,
        v_schedule: str = "linear",
    ):
        # Set device to CPU if CUDA is not available or None
        """__init__.

        Args:
            timesteps (int): Description.
            beta_schedule (str): Description.
            device (str): Description.
            v_min (float): Description.
            v_max (float): Description.
            v_schedule (str): Description.
        """
        if device is None or (device == "cuda" and not torch.cuda.is_available()):
            device = "cpu"
        self.device = device

        super().__init__(timesteps, beta_schedule, device)
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "SciPy is required for RicianDiffusion. Please install it: pip install scipy",
            )

        self.v_min = v_min
        self.v_max = v_max

        # Create a schedule for the non-centrality parameter 'v'.
        # We want v to be high for high t (more Gaussian) and low for low t
        # (more Rician).
        if v_schedule == SchedulerTypes.LINEAR.value:
            self.v_schedule = torch.linspace(v_min, v_max, timesteps, device=device)
        else:  # Cosine schedule
            t_steps = torch.linspace(0, 1, timesteps, device=device)
            self.v_schedule = v_min + (v_max - v_min) * (1 - torch.cos(t_steps * torch.pi * 0.5))

        # Precompute mean and std for each 'v' in the schedule for
        # standardization.
        # This is done once at initialization for efficiency.
        v_schedule_np = self.v_schedule.cpu().numpy()
        # Assuming scale (sigma) = 1 for the base Gaussians used to generate
        # Rician noise.
        means_np = np.array([rice.mean(v, scale=1) for v in v_schedule_np])
        stds_np = np.array([rice.std(v, scale=1) for v in v_schedule_np])

        self.means = torch.from_numpy(means_np).float().to(device)
        self.stds = torch.from_numpy(stds_np).float().to(device)

    def q_sample(self, x_start, t, noise=None):
        """q_sample.

        Args:
            x_start (Any): Description.
            t (Any): Description.
            noise (Any): Description.
        Returns:
            Any: Description.
        """
        if noise is None:
            # Extract scheduled parameters for the given timesteps t
            v_t = self._extract(self.v_schedule, t, x_start.shape)
            mean_t = self._extract(self.means, t, x_start.shape)
            std_t = self._extract(self.stds, t, x_start.shape)

            # Generate Rician noise: R = sqrt((g1+v)^2 + g2^2) where
            # g1, g2 ~ N(0,1)
            g1 = torch.randn_like(x_start)
            g2 = torch.randn_like(x_start)
            rician_unstd = torch.sqrt((g1 + v_t) ** 2 + g2**2)

            # Standardize the noise to have mean=0, variance=1
            noise = (rician_unstd - mean_t) / std_t

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
