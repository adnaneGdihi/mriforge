"""Neural ODE Dynamics for Cine MRI.

Models temporal evolution of MRI states (k-space or image) as a continuous-time ODE.
dx/dt = f(x, t, theta)
"""

from collections.abc import Callable

import torch
import torch.nn as nn

# Try importing torchdiffeq
try:
    from torchdiffeq import odeint_adjoint as odeint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    # Fallback to standard odeint or manual
    try:
        from torchdiffeq import odeint

        TORCHDIFFEQ_AVAILABLE = True
    except ImportError:
        TORCHDIFFEQ_AVAILABLE = False


class NeuralODEDynamics(nn.Module):
    """Differentiable ODE Solver Layer using Adjoint Method."""

    def __init__(self, drift_net: nn.Module, solver: str = "dopri5"):
        """
        Args:
            drift_net: Neural Network computing d(state)/d(time).
                       Contract (SSOT): drift_net MUST implement forward(state, time) -> d(state)/d(time).
                       Internally wrapped so torchdiffeq's func(time, state) maps to drift_net(state, time)
                       (see line `return self.drift_net(z, t)`). A module ordered (time, state) is NOT supported.
            solver: ODE solver method ('euler', 'rk4', 'dopri5')
        """
        super().__init__()
        self.drift_net = drift_net
        self.solver = solver

    def forward(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Integrate from t[0] to t[-1].

        Args:
            z0: Latent state at t=0 (Batch, LatentDim, ...)
            t: Tensor of time points to evaluate (T,)

        Returns:
            States at times t: (Batch, T, LatentDim, ...)

        forward method for NeuralODEDynamics.

        Executes PyTorch tensor operations.

        Args:
            z0 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""

        # odeint requires func(t, z) signature and must be nn.Module for adjoint
        class ODEFunc(nn.Module):
            """ODEFunc class."""

            def __init__(self, drift_net):
                """__init__.

                Args:
                    drift_net (Any): Description.
                """
                super().__init__()
                self.drift_net = drift_net

            def forward(self, t, z):
                # drift_net expected to take (z, t) or just z.
                # If drift_net is a standard layer, it usually takes x.
                # If it's time-conditioned, it takes (x, t).
                # Blueprint assumes drift_net(z, t).
                # We map the time tensor to the batch.
                # t is a scalar during integration step.
                """forward.

                        Args:
                            t (Any): Description.
                            z (Any): Description.
                        Returns:
                            Any: Description.

                forward method for ODEFunc.

                Executes PyTorch tensor operations.

                Args:
                    t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
                    z (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor with shape matching the operation.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
                t_expand = (
                    t.expand(z.shape[0], 1) if isinstance(t, torch.Tensor) and t.ndim == 0 else t
                )
                # Some drift nets might handle scalar t.
                return self.drift_net(z, t)

        func = ODEFunc(self.drift_net)

        if TORCHDIFFEQ_AVAILABLE:
            # Returns (T, Batch, LatentDim, ...)
            z_t = odeint(func, z0, t, method=self.solver, rtol=1e-4, atol=1e-5)
            # Permute to (Batch, T, ...)
            # Dims:
            # 0: T
            # 1: Batch
            # 2+: Latent dims
            permute_dims = [1, 0] + list(range(2, z_t.ndim))
            return z_t.permute(*permute_dims)
        else:
            return self._manual_integration(func, z0, t)

    def _manual_integration(
        self, func: Callable, x0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Manual simple integration if torchdiffeq missing."""
        results = [x0]
        curr_x = x0

        for i in range(len(t) - 1):
            t_curr = t[i]
            t_next = t[i + 1]
            dt = t_next - t_curr

            # Simple Euler
            dxdt = func(t_curr, curr_x)
            curr_x = curr_x + dxdt * dt
            results.append(curr_x)

        return torch.stack(results).permute(1, 0, *range(2, x0.ndim + 1))
