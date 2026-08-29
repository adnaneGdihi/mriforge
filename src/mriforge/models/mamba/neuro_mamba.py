"""Neuromorphic Mamba: Spiking State Space Model.

Innovation VIII (SOTA 2.0): Spiking Mamba for Edge MRI.

Mathematical Foundation:
    Leaky Integrate-and-Fire (LIF) Neuron:
    V[t] = beta * V[t-1] + I[t] - S[t-1] * V_th
    S[t] = Heaviside(V[t] - V_th)

    Spiking SSM:
    The hidden state h[t] follows LIF dynamics where input I[t] comes from SSM recurrence.
    h_mem[t] = A * h_spike[t-1] + B * x[t]
    h_spike[t] = LIF(h_mem[t])

    Surrogate Gradient:
    dS/dV approx sigmoid'(V) or similar smooth function to enable backprop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.registry import register_model

# Try to import jaxtyping
try:
    from jaxtyping import Float
except ImportError:
    from typing import Annotated, TypeVar

    T = TypeVar("T")
    Float = Annotated[T, "Float"]


class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient for Heaviside step function."""

    @staticmethod
    def forward(ctx, input, thresh=1.0, alpha=2.0):
        """forward.

        Args:
            ctx (Any): Description.
            input (Any): Description.
            thresh (Any): Description.
            alpha (Any): Description.
        Returns:
            Any: Description.
        """
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.alpha = alpha
        return (input > thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        """backward.

        Args:
            ctx (Any): Description.
            grad_output (Any): Description.
        Returns:
            Any: Description.
        """
        (input,) = ctx.saved_tensors
        thresh = ctx.thresh
        alpha = ctx.alpha

        # Sigmoid derivative or ATan derivative approximation
        # f(x) = 1 / (1 + exp(-alpha * (x - thresh)))
        # f'(x) = alpha * f(x) * (1 - f(x))
        # Here we use a fast sigmoid approx or ATan

        grad_input = grad_output.clone()
        sgax = (input - thresh) * alpha
        grad_input = (
            grad_input * (alpha / (1 + torch.exp(-sgax))) * (1 - 1 / (1 + torch.exp(-sgax)))
        )

        # Simple box derivative is also common: |x-thresh| < 0.5

        return grad_input, None, None


def surrogate_spike(x, thresh=1.0, alpha=2.0):
    """surrogate_spike.

    Args:
        x (Any): Description.
        thresh (Any): Description.
        alpha (Any): Description.
    Returns:
        Any: Description.
    """
    return SurrogateSpike.apply(x, thresh, alpha)


class LIFCell(nn.Module):
    """Leaky Integrate-and-Fire Cell."""

    def __init__(self, decay: float = 0.9, thresh: float = 1.0):
        """__init__.

        Args:
            decay (float): Description.
            thresh (float): Description.
        """
        super().__init__()
        self.decay = decay
        self.thresh = thresh

    def forward(self, input: torch.Tensor, mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Input current I[t]
            mem: Membrane potential V[t-1]

        Returns:
            spike: S[t]
            mem_next: V[t]

        forward method for LIFCell.

        Executes PyTorch tensor operations.

        Args:
            input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mem (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Dynamics: V[t] = beta * V[t-1] + Input
        # If spike, reset. We use soft reset (subtract thresh).

        mem_next = self.decay * mem + input

        spike = surrogate_spike(mem_next, self.thresh)

        # Soft reset
        mem_next = mem_next - spike * self.thresh

        return spike, mem_next


class SpikingSSM(nn.Module):
    """Spiking State Space Model Layer."""

    def __init__(self, d_model: int, d_state: int = 16):
        """__init__.

        Args:
            d_model (int): Description.
            d_state (int): Description.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # SSM parameters
        self.in_proj = nn.Linear(d_model, d_state)
        # Recurrent weight A is implicitly handled by LIF decay,
        # but we can add an explicit mixing matrix W_rec if we want "SSM" structure beyond scalar decay.
        # Standard Mamba A is diagonal. LIF decay corresponds to diagonal A.
        # So LIF decay `beta` IS `A`.
        # However, Mamba has B*x and C*h.

        # We'll implement a formal Spiking SSM:
        # h[t] = A * h[t-1] + B * x[t]  <- This is the membrane potential accumulation
        # y[t] = Spike(h[t])

        # Or Recurrent SNN:
        # h[t] = LIF(W_rec * h[t-1] + W_in * x[t])
        # This maps better to hardware.

        self.rec_weight = nn.Linear(d_state, d_state)  # A matrix (dense or diagonal)
        self.lif = LIFCell(decay=0.9)  # Membrane leak

        self.out_proj = nn.Linear(d_state, d_model)

    def forward(self, x: Float[torch.Tensor, "B L D"]) -> Float[torch.Tensor, "B L D"]:
        """forward.

        Args:
            x (Float[torch.Tensor, 'B L D']): Description.
        Returns:
            Float[torch.Tensor, 'B L D']: Description.

        forward method for SpikingSSM.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[torch.Tensor, 'B L D']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, D = x.shape

        # Project input
        u = self.in_proj(x)  # [B, L, d_state]

        # Init state
        spike_state = torch.zeros(B, self.d_state, device=x.device)
        mem_state = torch.zeros(B, self.d_state, device=x.device)

        outputs = []

        for t in range(L):
            # Recurrent input
            # I[t] = W_in * x[t] + W_rec * spike[t-1]
            # u[:, t] is W_in * x[t]

            recurrence = self.rec_weight(spike_state)
            current = u[:, t] + recurrence

            spike, mem_state = self.lif(current, mem_state)
            spike_state = spike

            outputs.append(spike)

        stacked_spikes = torch.stack(outputs, dim=1)  # [B, L, d_state]

        out = self.out_proj(stacked_spikes)

        return out + x  # Residual


@register_model(name="neuro_mamba", training_mode="reconstruction")
class NeuroMamba(nn.Module):
    """Neuromorphic Mamba Model."""

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
            [SpikingSSM(base_dim, d_state=base_dim) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(base_dim)
        self.decoder = nn.Linear(base_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for NeuroMamba.

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
            x = layer(x)

        x = self.norm(x)
        x = self.decoder(x)

        if spatial_shape is not None:
            B, L, C = x.shape
            H, W = spatial_shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x


def create_neuro_mamba(in_channels: int = 2, base_dim: int = 64, num_layers: int = 4) -> NeuroMamba:
    """create_neuro_mamba.

    Args:
        in_channels (int): Description.
        base_dim (int): Description.
        num_layers (int): Description.
    Returns:
        NeuroMamba: Description.
    """
    return NeuroMamba(in_channels, base_dim, num_layers)
