import torch
from torch import nn

try:
    from .score_based_diffusion import ScoreBasedDiffusion
    from .unet_architectures import ViTHybridUNet
except ImportError:
    from .score_based_diffusion import ScoreBasedDiffusion
    from .unet_architectures import ViTHybridUNet


class ViTScoreDiffusion(nn.Module, ScoreBasedDiffusion):
    """Diffusion wrapper that is also an nn.Module and can wrap a model.

    Supports initialization either with a model as the first argument or with
    standard diffusion args. Provides forward() that delegates to the wrapped
    model and sample() that uses the base p_sample_loop.
    """

    def __init__(
        self,
        model_or_timesteps=None,
        beta_schedule="linear",
        device=None,
        timesteps: int = 1000,
        **kwargs,
    ):
        """__init__.

        Args:
            model_or_timesteps (Any): Description.
            beta_schedule (Any): Description.
            device (Any): Description.
            timesteps (int): Description.
        """
        nn.Module.__init__(self)
        if isinstance(model_or_timesteps, nn.Module):
            self.model = model_or_timesteps
            # Initialize diffusion with provided timesteps and schedule
            ScoreBasedDiffusion.__init__(
                self,
                timesteps=timesteps,
                beta_schedule=beta_schedule,
                device=(device or ("cuda" if torch.cuda.is_available() else "cpu")),
            )
        else:
            self.model = None
            ScoreBasedDiffusion.__init__(
                self,
                timesteps=(
                    model_or_timesteps if isinstance(model_or_timesteps, int) else timesteps
                ),
                beta_schedule=beta_schedule,
                device=(device or ("cuda" if torch.cuda.is_available() else "cpu")),
            )

    def forward(self, x, t, cond=None):
        """forward.

        Args:
            x (Any): Description.
            t (Any): Description.
            cond (Any): Description.
        Returns:
            Any: Description.

        forward method for ViTScoreDiffusion.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.model is None:
            raise RuntimeError(
                "ViTScoreDiffusion has no wrapped model. Provide a model in the constructor.",
            )
        return self.model(x, t, cond) if cond is not None else self.model(x, t)

    @torch.no_grad()
    def sample(
        self,
        shape,
        device,
        steps: int | None = None,
        eta: float = 1.0,
    ):
        """Generate samples using the base DDPM sampler; steps overrides
        timesteps.
        """
        if self.model is None:
            raise RuntimeError("ViTScoreDiffusion has no wrapped model to sample with.")
        if steps is None:
            return self.p_sample_loop(self.model, shape)
        # Temporarily adjust timesteps for sampling
        orig = self.timesteps
        try:
            self.timesteps = steps
            return self.p_sample_loop(self.model, shape)
        finally:
            self.timesteps = orig


def create_vit_diffusion_model(
    in_channels: int = 1,
    out_channels: int = 1,
    time_dim: int = 256,
    **kwargs,
) -> tuple[ScoreBasedDiffusion, nn.Module]:
    """Factory function to create a ViT-based diffusion model and process.

    Returns:
    A tuple containing the diffusion process and the ViT Hybrid U-Net model.

    """
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Filter diffusion kwargs to accepted keys only
    diffusion_keys = {
        "timesteps",
        "beta_schedule",
        "gradient_lambda",
        "entropy_lambda",
        "ssim_lambda",
    }
    diffusion_kwargs = {k: v for k, v in kwargs.items() if k in diffusion_keys}

    # Create the ViT Hybrid U-Net model
    model = ViTHybridUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        time_dim=time_dim,
        **kwargs,
    ).to(device)

    # Wrap model inside the diffusion module for convenience
    diffusion_process = ViTScoreDiffusion(model, device=device, **diffusion_kwargs)

    return diffusion_process, model


# Export for use in model factory
__all__ = ["ViTScoreDiffusion", "create_vit_diffusion_model"]
