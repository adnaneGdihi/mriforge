import torch
import torch.nn.functional as F

from spectramr.models.diffusion.base_diffusion import Diffusion


def ssim_metric(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute SSIM using the unified metric registry.

    Args:
        x: Predicted tensor (normalized to [0, 1])
        y: Target tensor (normalized to [0, 1])

    Returns:
        Scalar SSIM value as tensor
    """
    # Lazy import: a module-scope ``from spectramr.core.metrics import
    # get_metric`` created an import cycle (core.metrics walk-discovery →
    # models.losses → models.diffusion → this module → core.metrics, still
    # partially initialized) that silently aborted the losses-package init,
    # leaving the loss registry half-populated for the whole process.
    from spectramr.core.metrics import get_metric

    try:
        metric = get_metric("ssim", device=x.device)
        ssim_value = metric(x, y)
        if hasattr(ssim_value, "item"):
            ssim_value = ssim_value.item()
        return torch.tensor(ssim_value, device=x.device)
    except Exception:
        return torch.tensor(0.0, device=x.device)


def compute_gradient_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute gradient error as a fallback metric."""
    try:
        # Simple gradient-based error computation
        grad_x = torch.gradient(x)[0]
        grad_y = torch.gradient(y)[0]
        error = F.mse_loss(grad_x, grad_y)
        return error
    except Exception:
        return torch.tensor(0.0, device=x.device)


def compute_gradient_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient entropy as a regularization term."""
    try:
        # Compute spatial gradients
        grad_x = torch.gradient(x)[0]
        grad_y = torch.gradient(x)[1] if x.ndim > 2 else torch.zeros_like(x)

        # Entropy as negative sum of squared gradients (encourages smoothness)
        entropy = -(grad_x**2 + grad_y**2).mean()
        return entropy
    except Exception:
        return torch.tensor(0.0, device=x.device)


class ScoreBasedDiffusion(Diffusion):
    """ScoreBasedDiffusion class."""

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        device=None,
        gradient_lambda=1.0,
        entropy_lambda=0.1,
        ssim_lambda=0.5,
    ):
        # Set device to CPU if CUDA is not available or None
        """__init__.

        Args:
            timesteps (Any): Description.
            beta_schedule (Any): Description.
            device (Any): Description.
            gradient_lambda (Any): Description.
            entropy_lambda (Any): Description.
            ssim_lambda (Any): Description.
        """
        if device is None or (device == "cuda" and not torch.cuda.is_available()):
            device = "cpu"
        self.device = device

        super().__init__(timesteps, beta_schedule, device)
        self.gradient_lambda = gradient_lambda
        self.entropy_lambda = entropy_lambda
        self.ssim_lambda = ssim_lambda

    def _predict_start_from_noise(self, x_t, t, noise):
        """Predicts x_0 from x_t and predicted noise using the DDPM formula.
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_t.shape,
        )
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)

        pred_x_start = (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        return pred_x_start

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        # Ensure all input tensors are on the correct device
        """p_losses.

        Args:
            denoise_model (Any): Description.
            x_start (Any): Description.
            t (Any): Description.
            noise (Any): Description.
            loss_type (Any): Description.
        Returns:
            Any: Description.
        """
        x_start = x_start.to(self.device)
        t = t.to(self.device)
        if noise is not None:
            noise = noise.to(self.device)

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        # Registry-routed dispatch (WS-6): an unknown loss_type raises instead
        # of silently defaulting to L2 (pitfall #9). Lazy import: losses/
        # __init__ imports models.diffusion modules — module scope would cycle.
        from spectramr.models.losses.elementary import resolve_elementary_loss

        noise_loss = resolve_elementary_loss(loss_type)(predicted_noise, noise)

        pred_x_start = self._predict_start_from_noise(x_noisy, t, predicted_noise)

        # Normalize images from [-1, 1] to [0, 1] for SSIM calculation, as
        # expected by the metric function.
        pred_x_start_norm = (pred_x_start.clamp(-1, 1) + 1) / 2
        x_start_norm = (x_start.clamp(-1, 1) + 1) / 2

        ssim_loss = (1.0 - ssim_metric(pred_x_start_norm, x_start_norm)) * self.ssim_lambda
        # Gradient metrics are less sensitive to range and can work on the
        # original [-1, 1] outputs.
        grad_error = compute_gradient_error(pred_x_start, x_start) * self.gradient_lambda
        grad_entropy = compute_gradient_entropy(pred_x_start) * self.entropy_lambda

        total_loss = noise_loss + ssim_loss + grad_error - grad_entropy
        return total_loss
