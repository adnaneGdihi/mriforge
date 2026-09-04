"""Chi-Square Diffusion Generator
=============================

Chi-Square diffusion generator implementing the IGenerator interface.
Uses Chi-Square noise instead of Gaussian noise for the diffusion process.
"""

import torch
from torch import nn

from spectramr.models.diffusion.chi_square_diffusion import ChiSquareDiffusion
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="chi_square_diffusion", training_mode="diffusion")
class ChiSquareDiffusionGenerator(nn.Module, IGenerator):
    """Chi-Square diffusion generator implementing the IGenerator interface.
    Uses Chi-Square noise instead of Gaussian noise for more diverse
    generation.
    """

    def __init__(
        self,
        denoising_model: nn.Module | None = None,
        in_channels: int = 3,
        hidden_channels: int = 64,
        image_size: tuple[int, int] = (64, 64),
        num_inference_steps: int = 50,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str | None = None,
        min_dof: int = 1,
        max_dof: int = 50,
        dof_schedule: str = "cosine",
        gaussian_approximation_threshold: int = 30,
    ):
        """Initialize Chi-Square Diffusion Generator.

        Args:
            denoising_model: The neural network model that predicts noise
            timesteps: Number of diffusion timesteps
            beta_schedule: Schedule for beta values ("linear" or "cosine")
            device: Device to run on ("cuda" or "cpu")
            min_dof: Minimum degrees of freedom for Chi-Square distribution
            max_dof: Maximum degrees of freedom for Chi-Square distribution
            dof_schedule: Schedule for degrees of freedom
            gaussian_approximation_threshold: int = 30,
        ):

        """
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        self.in_channels = in_channels
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps

        if denoising_model is None:
            denoising_model = self._build_default_denoiser(
                in_channels,
                hidden_channels,
            )

        self.denoising_model = denoising_model
        self.diffusion = ChiSquareDiffusion(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            device=device,
            min_dof=min_dof,
            max_dof=max_dof,
            dof_schedule=dof_schedule,
            gaussian_approximation_threshold=gaussian_approximation_threshold,
        )

        # Move model to device
        self.denoising_model.to(self.diffusion.device)

    @property
    def name(self) -> str:
        """Return the model name."""
        return "ChiSquareDiffusionGenerator"

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training: add noise and predict x_0.

        Args:
            x: Input clean image of shape (B, C, H, W)
            t: Optional timestep tensor or normalized values

        Returns:
            Predicted clean image x_0 of shape (B, C, H, W)

        """
        # Shape guards
        assert x.dim() == 4, f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D"
        assert x.shape[2] >= 32 and x.shape[3] >= 32, f"Input spatial dims too small: {x.shape[2:]}"

        device = self.diffusion.device
        x = x.to(device)
        batch_size = x.shape[0]

        if t is None:
            t_indices = torch.randint(
                0,
                self.diffusion.timesteps,
                (batch_size,),
                device=device,
            )
        else:
            if not torch.is_tensor(t):
                raise TypeError("t must be a torch.Tensor")
            t = t.to(device)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.shape[0] != batch_size and t.shape[0] != 1:
                raise ValueError(
                    "t must have the same batch size as x or be broadcastable.",
                )
            if torch.any(t < 0):
                raise ValueError("t values must be non-negative")
            if t.dtype.is_floating_point:
                if torch.any(t > 1):
                    raise ValueError("normalized timesteps must be <= 1")
                scale = float(self.diffusion.timesteps - 1)
                t_indices = torch.clamp(
                    (t * scale).round().long(),
                    min=0,
                    max=self.diffusion.timesteps - 1,
                )
            else:
                if torch.any(t >= self.diffusion.timesteps):
                    raise ValueError("t indices out of range")
                t_indices = t.long()
            if t_indices.shape[0] == 1 and batch_size > 1:
                t_indices = t_indices.repeat(batch_size)

        noise = torch.randn_like(x)

        x_t = self.diffusion.q_sample(x, t_indices, noise)

        # Predict x_0
        try:
            pred_x_start = self.denoising_model(x_t, t_indices)
        except TypeError:
            pred_x_start = self.denoising_model(x_t)

        # Output shape guard
        assert pred_x_start.shape == x.shape, (
            f"Output shape {pred_x_start.shape} doesn't match input shape {x.shape}"
        )

        return pred_x_start

    def generate(
        self,
        batch_size: int = 1,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples using Chi-Square diffusion.

        Args:
            batch_size: Number of samples to generate
            height: Optional height override
            width: Optional width override
            num_inference_steps: Optional number of reverse steps
            initial_noise: Optional initial noise tensor
            cond: Optional conditioning information

        Returns:
            Generated tensor

        """
        steps = num_inference_steps or self.num_inference_steps
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        height = height or self.image_size[0]
        width = width or self.image_size[1]

        device = self.diffusion.device

        if initial_noise is None:
            noise = torch.randn(
                batch_size,
                self.in_channels,
                height,
                width,
                device=device,
            )
        else:
            noise = initial_noise.to(device)
            if noise.ndim != 4:
                raise ValueError("initial_noise must be a 4D tensor")
            if noise.shape[1] != self.in_channels:
                raise ValueError(
                    "Channel count of initial_noise does not match generator.",
                )

        step_indices = self._compute_step_indices(steps)
        img = noise
        for t_index in step_indices:
            t_tensor = torch.full(
                (img.shape[0],),
                t_index,
                device=device,
                dtype=torch.long,
            )
            img = self.diffusion.p_sample(
                self.denoising_model,
                img,
                t_tensor,
                t_index,
                cond=cond,
            )

        return img

    def get_output_shape(
        self,
        input_shape: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        if input_shape is None:
            return (self.in_channels, *self.image_size)
        return input_shape

    def get_parameter_count(self) -> int:
        """Count total parameters in the denoising model."""
        return sum(p.numel() for p in self.denoising_model.parameters() if p.requires_grad)

    def _build_default_denoiser(
        self,
        in_channels: int,
        hidden_channels: int,
    ) -> nn.Module:
        """_build_default_denoiser.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
        Returns:
            nn.Module: Description.
        """

        class _DefaultDenoiser(nn.Module):
            """_DefaultDenoiser class."""

            def __init__(self, in_ch: int, hid_ch: int) -> None:
                """__init__.

                Args:
                    in_ch (int): Description.
                    hid_ch (int): Description.
                """
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hid_ch, hid_ch, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hid_ch, in_ch, kernel_size=3, padding=1),
                )

            def forward(
                self,
                x: torch.Tensor,
                t: torch.Tensor | None = None,
                cond: torch.Tensor | None = None,
            ) -> torch.Tensor:
                """forward.

                        Args:
                            x (torch.Tensor): Description.
                            t (Optional[torch.Tensor]): Description.
                            cond (Optional[torch.Tensor]): Description.
                        Returns:
                            torch.Tensor: Description.

                forward method for _DefaultDenoiser.

                Executes PyTorch tensor operations.

                Args:
                    x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
                    t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
                    cond (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
                return self.net(x)

        return _DefaultDenoiser(in_channels, hidden_channels)

    def _compute_step_indices(self, steps: int) -> list[int]:
        """_compute_step_indices.

        Args:
            steps (int): Description.
        Returns:
            list[int]: Description.
        """
        total = self.diffusion.timesteps
        if steps >= total:
            return list(range(total - 1, -1, -1))

        interval = max(total // steps, 1)
        indices = list(range(total - 1, -1, -interval))
        if not indices:
            indices = [total - 1]
        if indices[-1] != 0:
            indices.append(0)
        if len(indices) > steps:
            indices = indices[:steps]
        while len(indices) < steps:
            indices.append(0)
        return indices
