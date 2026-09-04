"""Differential Mamba: Continuous-Time State Space Model.

Innovation VII (SOTA 2.0): Neural ODE integration for SSM.

Mathematical Foundation:
    Standard SSM: h[k] = A h[k-1] + B x[k]
    Continuous SSM: h'(t) = A h(t) + B x(t)

    We solve this using an ODE Solver: h(t) = ODESolve(f, h(0), t)
    where f(t, h) = A h(t) + B x(t)

    To handle discrete input x[k] at continuous time t, we interpolate x(t).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.registry import register_model

# Try to import jaxtyping
try:
    from jaxtyping import Complex, Float
except ImportError:
    from typing import Annotated, TypeVar

    T = TypeVar("T")
    Float = Annotated[T, "Float"]
    Complex = Annotated[T, "Complex"]


class LinearInterpolation(nn.Module):
    """Linearly interpolates a discrete sequence x[k] to continuous time x(t)."""

    def __init__(self, seq_len: int):
        """__init__.

        Args:
            seq_len (int): Description.
        """
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            t: Continuous time in [0, L-1]

        forward method for LinearInterpolation.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (float): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Clamp t
        t = max(0.0, min(float(self.seq_len - 1), t))

        idx_low = int(t)
        idx_high = min(idx_low + 1, self.seq_len - 1)
        alpha = t - idx_low

        x_low = x[:, idx_low, :]
        x_high = x[:, idx_high, :]

        return (1 - alpha) * x_low + alpha * x_high


class DiffSSMDeriv(nn.Module):
    """Derivative function for the ODE solver: dh/dt = A h + B x(t)."""

    def __init__(self, A: torch.Tensor, B_matrix: torch.Tensor, x_seq: torch.Tensor):
        """__init__.

        Args:
            A (torch.Tensor): Description.
            B_matrix (torch.Tensor): Description.
            x_seq (torch.Tensor): Description.
        """
        super().__init__()
        self.A = A  # [D_state, D_state] or diagonal [D_state]
        self.B = B_matrix  # [D_state, D_input]
        self.x_seq = x_seq  # [Batch, L, D_input]
        self.interp = LinearInterpolation(x_seq.shape[1])

    def forward(self, t: float | torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Scalar time. A plain float is preferred — PERF (2026-07-01):
                the old contract required a scalar *tensor* and called
                ``t.item()``, a device→host sync per evaluation; the time
                grid is static Python floats, so no tensor is ever needed.
                Scalar tensors are still accepted for torchdiffeq-style
                callers (they pay the sync they always did).
            h: Current state [Batch, D_state]

        Returns:
            torch.Tensor: dh/dt, shape [Batch, D_state].
        """
        t_val = t.item() if isinstance(t, torch.Tensor) else float(t)

        # Interpolate input x(t)
        xt = self.interp(self.x_seq, t_val)  # [B, D_in]

        # Calculate derivative
        # dh/dt = A @ h + B @ xt (if dense)
        # Assuming A is parameter, h is state

        # If A is diagonal (element-wise multiplication)
        if self.A.dim() == 1:
            ah = self.A * h
        else:
            ah = h @ self.A.T

        bxt = xt @ self.B.T

        return ah + bxt


class DiffSSM(nn.Module):
    """Differential SSM Layer using Euler Solver (for simplicity/speed)."""

    def __init__(self, d_model: int, d_state: int = 16, dt_step: float = 0.5):
        """__init__.

        Args:
            d_model (int): Description.
            d_state (int): Description.
            dt_step (float): Description.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_step = dt_step

        # SSM Parameters
        # A matrix (diagonal). Initialize to negative values for stability (stable system)
        # Random normal - 1.0 ensures mean is around -1.0
        self.A = nn.Parameter(torch.randn(d_state) - 0.5)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Linear(d_model, d_model)

        # Input/Output projections
        self.in_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Float[torch.Tensor, "B L D"]) -> Float[torch.Tensor, "B L D"]:
        """forward.

        Args:
            x (Float[torch.Tensor, 'B L D']): Description.
        Returns:
            Float[torch.Tensor, 'B L D']: Description.

        forward method for DiffSSM.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[torch.Tensor, 'B L D']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, D = x.shape

        # Initial state
        h = torch.zeros(B, self.d_state, device=x.device)

        # Integration loop — a manual explicit-Euler "odeint" that records the
        # state at integer time points t = 0, 1, ..., L-1. The nonlinear tanh
        # clamp makes the recurrence inherently sequential (NOT expressible as
        # a standard selective scan), so the loop stays; PERF (2026-07-01)
        # removed everything else from it:
        #   * the per-substep scalar-tensor time round-trip (an H2D alloc plus
        #     a D2H sync inside DiffSSMDeriv per substep) — the time grid is
        #     static Python floats, so L*steps_per_token stalls are gone;
        #   * the per-substep B-projection matmul: B is linear, so
        #     B(interp(x, t)) == interp(B(x), t) — project the whole sequence
        #     once and interpolate the projections (elementwise only);
        #   * the per-token C/D linear launches: collect the integer-time
        #     states and batch the readout over the sequence after the loop.

        # Optimization: We only need outputs at integer steps.
        # If dt_step=1.0, this becomes a ResNet/RNN
        # If dt_step=0.1, we take 10 substeps per token
        steps_per_token = int(1.0 / self.dt_step)

        xb = self.B(x)  # [B, L, d_state] — input projected once
        interp = LinearInterpolation(L)
        a_is_diag = self.A.dim() == 1

        # Reproduce the old ``current_t += dt_step`` accumulation exactly
        # (including its float drift) so substep times match the previous
        # loop bitwise.
        t_grid: list[float] = []
        current_t = 0.0
        for _ in range(L * steps_per_token):
            t_grid.append(current_t)
            current_t += self.dt_step

        h_states = []
        step = 0
        for _i in range(L):
            # Record state at integer time t=i (readout batched below).
            h_states.append(h)

            # Evolve state from t=i to t=i+1 (Euler: h += dt * tanh(f(t, h)))
            for _ in range(steps_per_token):
                bxt = interp(xb, t_grid[step])
                ah = self.A * h if a_is_diag else h @ self.A.T
                # tanh clamp derivatives to prevent explosion
                dh = torch.tanh(ah + bxt)
                h = h + self.dt_step * dh
                step += 1

        h_all = torch.stack(h_states, dim=1)  # [B, L, d_state]
        # y[i] = C h[i] + D x[i] — full-sequence linears (2 launches, not 2L)
        return self.C(h_all) + self.D(x)


@register_model(name="diff_mamba", training_mode="reconstruction")
class DiffMamba(nn.Module):
    """Differential Mamba Model w/ Continuous Depth/State."""

    def __init__(self, in_channels: int, base_dim: int, num_layers: int = 4):
        """__init__.

        Args:
            in_channels (int): Description.
            base_dim (int): Description.
            num_layers (int): Description.
        """
        super().__init__()
        self.encoder = nn.Linear(in_channels, base_dim)

        self.layers = nn.ModuleList(
            [DiffSSM(base_dim, d_state=base_dim) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(base_dim)
        self.decoder = nn.Linear(base_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, L, C]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DiffMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            spatial_shape = (H, W)
        else:
            spatial_shape = None

        x = self.encoder(x)

        for layer in self.layers:
            # Residual connection
            x = x + layer(x)  # DiffSSM handles its own dynamics but let's add ResNet style

        x = self.norm(x)
        x = self.decoder(x)

        if spatial_shape is not None:
            B, L, C = x.shape
            H, W = spatial_shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x


def create_diff_mamba(in_channels: int = 2, base_dim: int = 64, num_layers: int = 4) -> DiffMamba:
    """create_diff_mamba.

    Args:
        in_channels (int): Description.
        base_dim (int): Description.
        num_layers (int): Description.
    Returns:
        DiffMamba: Description.
    """
    return DiffMamba(in_channels, base_dim, num_layers)
