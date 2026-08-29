"""Rectified Flow Physics Implementation.

Rectified Flow (InstaFlow/Reflow) learns straight paths in latent space to
map noise to data with fewer ODE steps than diffusion.

v = x1 - x0
dx/dt = v(x, t)
"""

import torch
import torch.nn as nn


class RectifiedFlow(nn.Module):
    """Rectified Flow Solver."""

    def __init__(self, model: nn.Module = None):
        """__init__.

        Args:
            model (nn.Module): Description.
        """
        super().__init__()
        self.model = model

    def get_velocity_target(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Compute velocity target for training.

        Args:
            x_0: Source distribution (Noise)
            x_1: Target distribution (Data)

        Returns:
            Target velocity vector (x_1 - x_0)
        """
        # Straight line interpolation target
        return x_1 - x_0

    def step_euler(self, x_t: torch.Tensor, velocity_pred: torch.Tensor, dt: float) -> torch.Tensor:
        """Perform one Euler integration step.

        x_{t+dt} = x_t + v(x_t, t) * dt
        """
        return x_t + velocity_pred * dt

    def sample(self, x_init: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Generate sample by integrating ODE."""
        if self.model is None:
            raise ValueError("Model required for sampling")

        dt = 1.0 / steps
        x = x_init

        for i in range(steps):
            t = i * dt
            t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])

            # Predict velocity field
            v_pred = self.model(x, t_tensor)

            # Integrate
            x = self.step_euler(x, v_pred, dt)

        return x
