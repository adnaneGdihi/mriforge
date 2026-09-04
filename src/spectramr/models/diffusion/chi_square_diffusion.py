import torch
from torch.distributions.chi2 import Chi2

from spectramr.config.schemas.enums import SchedulerTypes

from .base_diffusion import Diffusion
from .cold_diffusion import ColdDiffusion


class ChiSquareDiffusion(Diffusion):
    """Implements a diffusion process using Chi-Square noise instead of Gaussian
    noise. This is based on the idea of non-Gaussian diffusion processes.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str = None,
        min_dof: int = 1,
        max_dof: int = 50,
        dof_schedule: str = "cosine",
        gaussian_approximation_threshold: int = 30,
    ):
        # Set device intelligently - use CPU for compatibility
        """__init__.

        Args:
            timesteps (int): Description.
            beta_schedule (str): Description.
            device (str): Description.
            min_dof (int): Description.
            max_dof (int): Description.
            dof_schedule (str): Description.
            gaussian_approximation_threshold (int): Description.
        """
        if device is None:
            device = "cpu"
        # We call super().__init__ to get the beta schedule, but we won't
        # use all of it.
        super().__init__(timesteps, beta_schedule, device)
        self.min_dof = min_dof
        self.max_dof = max_dof
        self.gaussian_approximation_threshold = gaussian_approximation_threshold

        # Create a schedule for the degrees of freedom (DoF)
        if dof_schedule == SchedulerTypes.LINEAR.value:
            self.dof = torch.linspace(min_dof, max_dof, timesteps, device=device)
        elif dof_schedule == SchedulerTypes.COSINE.value:
            t = torch.linspace(0, 1, timesteps, device=device)
            self.dof = min_dof + (max_dof - min_dof) * (torch.cos(t * torch.pi * 0.5))
        elif dof_schedule == "exponential":
            min_dof_tensor = torch.tensor(float(min_dof), device=device)
            max_dof_tensor = torch.tensor(float(max_dof), device=device)
            self.dof = torch.logspace(
                torch.log10(min_dof_tensor),
                torch.log10(max_dof_tensor),
                timesteps,
                device=device,
            )
        else:
            raise ValueError(f"Unknown DoF schedule: {dof_schedule}")
        self.dof = self.dof.to(device)

    def q_sample(self, x_start, t, noise=None):
        """Forward process: add Chi-Square noise to the image.
        For a Chi-Square distribution with k DoF, mean is k and variance is 2k.
        We need to standardize it to have zero mean and unit variance.
        """
        if noise is None:
            dof_t = self._extract(self.dof, t, x_start.shape)
            # For large DoF, Chi-Square approximates Normal(k, 2k).
            # We can use torch.randn for efficiency.
            is_large_dof = (dof_t > self.gaussian_approximation_threshold).squeeze()

            noise = torch.zeros_like(x_start)

            # Handle large DoF cases with Gaussian approximation
            if torch.any(is_large_dof):
                large_indices = torch.where(is_large_dof)[0]
                n_large = len(large_indices)
                mean = dof_t[large_indices].view(n_large, 1, 1, 1)
                std = torch.sqrt(2 * mean)
                noise_large = torch.randn(
                    (n_large, *x_start.shape[1:]),
                    device=x_start.device,
                ).view(n_large, *x_start.shape[1:])
                for i, idx in enumerate(large_indices):
                    mean_val = mean[i, 0, 0, 0].item()
                    std_val = std[i, 0, 0, 0].item()
                    noise[idx] = (noise_large[i] - mean_val) / std_val

            # Handle small DoF cases with actual Chi-Square sampling
            if torch.any(~is_large_dof):
                small_indices = torch.where(~is_large_dof)[0]
                n_small = len(small_indices)
                dof_small = dof_t[small_indices].view(n_small, 1, 1, 1)
                chi2_dist = Chi2(dof_small)
                chi2_noise = chi2_dist.sample().to(x_start.device)
                # Standardize each sample individually
                for i in range(n_small):
                    dof_val = dof_small[i, 0, 0, 0].item()
                    standardized_noise_i = (chi2_noise[i] - dof_val) / torch.sqrt(
                        torch.tensor(2 * dof_val, device=x_start.device),
                    )
                    noise[small_indices[i]] = standardized_noise_i

        # Use the standard DDPM noising formula, but with Chi-Square noise
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


class ChiSquareColdDiffusion(ColdDiffusion):
    """A cold diffusion variant that uses Chi-Square noise for its
    degradation process.
    """

    """
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str = "cuda",
        min_dof: int = 1,
        max_dof: int = 50,
        dof_schedule: str = "cosine",
        gaussian_approximation_threshold: int = 30,
        **kwargs,
    ):
        """__init__.

        Args:
            timesteps (int): Description.
            beta_schedule (str): Description.
            device (str): Description.
            min_dof (int): Description.
            max_dof (int): Description.
            dof_schedule (str): Description.
            gaussian_approximation_threshold (int): Description.
        """
        super().__init__(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            device=device,
            **kwargs,
        )

        self.chi_square_sampler = ChiSquareDiffusion(
            timesteps,
            beta_schedule,
            device,
            min_dof,
            max_dof,
            dof_schedule,
            gaussian_approximation_threshold,
        )

    def _degrade(self, x_0_pred, t, mask=None):
        """Degrades the predicted x_0 to x_t using Chi-Square noise.

        ``mask`` is accepted to match the parent ``ColdDiffusion._degrade``
        contract (parent ``p_sample`` calls ``self._degrade(..., mask=mask)``)
        but is unused: chi-square degradation is additive noise via ``q_sample``.
        """
        return self.chi_square_sampler.q_sample(x_start=x_0_pred, t=t)

    def p_losses(
        self,
        denoise_model,
        x_start,
        t,
        noise=None,
        loss_type="l1",
        cond=None,
    ):
        """p_losses.

        Args:
            denoise_model (Any): Description.
            x_start (Any): Description.
            t (Any): Description.
            noise (Any): Description.
            loss_type (Any): Description.
            cond (Any): Description.
        Returns:
            Any: Description.
        """
        x_noisy = self.chi_square_sampler.q_sample(x_start=x_start, t=t, noise=noise)

        # Check if the denoise_model supports the 'cond' parameter
        import inspect

        sig = inspect.signature(denoise_model.forward)
        supports_cond = "cond" in sig.parameters

        # Pass the condition only if the model supports it
        if supports_cond and cond is not None:
            predicted_x_start = denoise_model(x_noisy, t, cond=cond)
        else:
            predicted_x_start = denoise_model(x_noisy, t)

        # Lazy import: losses/__init__ imports models.diffusion modules
        # (levy, resetting) — a module-scope import here would cycle.
        from spectramr.models.losses.elementary import resolve_elementary_loss

        return resolve_elementary_loss(loss_type)(predicted_x_start, x_start)
