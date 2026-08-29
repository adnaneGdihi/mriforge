"""Bloch-Mamba v2: Parallel Scan Physics-Embedded SSM.

Innovation VI (SOTA 2.0): Physically Parameterized State Space Model.

Mathematical Foundation:
    The continuous time Bloch equation for transverse magnetization (rotating frame):
    dM_xy/dt = -R2 * M_xy - i * gamma * delta_B0 * M_xy
             = -(R2 + i*omega) * M_xy
             = A * M_xy

    This is a diagonal linear ODE: h'(t) = A(t)h(t) + B(t)u(t)
    Discretized (ZOH):
    h[k] = exp(A * dt) * h[k-1] + ...

    We use a parallel associative scan to solve this efficiently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float

# from torch.func import associative_scan (Not available in this torch version)
from mriforge.models.registry import register_model


class BlochParameterEstimator(nn.Module):
    """Predicts physical parameters (T1, T2, B0) from input features."""

    def __init__(self, d_model: int, d_out: int = 1) -> None:
        """__init__.

        Args:
            d_model (int): Description.
            d_out (int): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3 * d_out),  # T1, T2, B0
        )
        self.d_out = d_out

    def forward(
        self, x: Float[torch.Tensor, "B L D"]
    ) -> tuple[
        Float[torch.Tensor, "B L D_out"],
        Float[torch.Tensor, "B L D_out"],
        Float[torch.Tensor, "B L D_out"],
    ]:
        """
        Args:
            x: Input features
        Returns:
            T1, T2, B0 maps (enforced to be positive/physical)

        forward method for BlochParameterEstimator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[Float[torch.Tensor, 'B L D_out'], Float[torch.Tensor, 'B L D_out'], Float[torch.Tensor, 'B L D_out']]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        params = self.net(x)
        B, L, _ = params.shape
        params = params.view(B, L, 3, self.d_out)

        t1_raw, t2_raw, b0_raw = params.unbind(dim=2)

        T1 = F.softplus(t1_raw) + 1.0  # Min 1ms
        T2 = F.softplus(t2_raw) + 1.0  # Min 1ms
        B0 = b0_raw  # Hz

        return T1, T2, B0


class BlochSSM(nn.Module):
    """Physically parameterized State Space Model."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt: float = 0.001,  # 1ms
    ) -> None:
        """__init__.

        Args:
            d_model (int): Description.
            d_state (int): Description.
            dt (float): Description.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.param_estimator = BlochParameterEstimator(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_state, bias=True)
        self.min_dt = dt

        self.in_proj = nn.Linear(d_model, d_state * 2)
        self.out_proj = nn.Linear(d_state * 2, d_model)

    def parallel_scan_step(
        self, log_A: Complex[torch.Tensor, "B L N"], B_u: Complex[torch.Tensor, "B L N"]
    ) -> Complex[torch.Tensor, "B L N"]:
        """
        Sequential fallback for: h[t] = A[t]*h[t-1] + B_u[t]
        TODO: Optimization - Implement Parallel Associative Scan when torch.func available.
        """
        B, L, N = log_A.shape
        A = torch.exp(log_A)

        # Sequential loop (PyTorch Script compatible ideally)
        # We assume L is reasonable for MRI (frequency encode dim ~256-512)
        # For Mamba (L ~ 1M), this is slow. For MRI Rec (~512), it's acceptable.
        # PERF (2026-07-01): the old code allocated a full ``zeros_like(B_u)``
        # + in-place first-row write only to overwrite ``h`` with the stacked
        # list below — dead alloc/copy removed; ``torch.stack`` copies anyway,
        # so values and grads are identical.
        current_h = B_u[:, 0]

        # Using a list accumulation might be faster than inplace tensor update in eager mode
        h_list = [current_h]

        for t in range(1, L):
            # h[t] = A[t] * h[t-1] + B_u[t]
            current_h = A[:, t] * current_h + B_u[:, t]
            h_list.append(current_h)

        h = torch.stack(h_list, dim=1)
        return h

    def forward(self, x: Float[torch.Tensor, "B L D"]) -> Float[torch.Tensor, "B L D"]:
        """forward.

        Args:
            x (Float[torch.Tensor, 'B L D']): Description.
        Returns:
            Float[torch.Tensor, 'B L D']: Description.

        forward method for BlochSSM.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[torch.Tensor, 'B L D']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, D = x.shape

        # 1. Physics Parameters
        T1, T2, delta_B0 = self.param_estimator(x)  # [B, L, d_state]

        # 2. Time step
        dt = F.softplus(self.dt_proj(x)) + self.min_dt  # [B, L, d_state]

        # 3. Discretization of Bloch Equations (ZOH)
        # Gamma in MHz/T is ~42.58, but we assume normalized inputs here or learnable scaling
        gamma = 2 * 3.14159 * 42.58  # rad/s/T roughly, but usually normalized
        # Let's assume params predict directly relevant rates

        R1 = 1.0 / T1
        R2 = 1.0 / T2
        omega = delta_B0  # Already in rad/s or equivalent

        # Transverse relaxation + precession: dM/dt = -(R2 + i*w)M
        # A_cont = -(R2 + i*omega)
        # A_disc = exp(A_cont * dt)

        # To avoid strictly complex types everywhere, we use PyTorch complex tensors
        log_A_real = -R2 * dt
        log_A_imag = -omega * dt
        log_A = torch.complex(log_A_real, log_A_imag)

        # 4. Input B*u
        # Project input to complex state
        u = self.in_proj(x)
        u_real, u_imag = u.chunk(2, dim=-1)
        B_u = torch.complex(u_real, u_imag)
        # Scale by dt (ZOH approx B_bar = A_inv * (A_bar - I) * B, approx B*dt for small dt)
        B_u = B_u * dt

        # 5. Associative Scan
        h = self.parallel_scan_step(log_A, B_u)

        # 6. Output mapping
        # Concatenate real and imag parts
        y_real = h.real
        y_imag = h.imag
        y = torch.cat([y_real, y_imag], dim=-1)

        out = self.out_proj(y)

        # Residual connection matches input dim
        return out + x


@register_model(
    name="bloch_mamba_v2",
    training_mode="reconstruction",
    # Maps (B, C, H, W) → 1D SFC sequence → Bloch-physics-embedded
    # Mamba → reshape back. Per the docstring image-domain in/out;
    # the physics is embedded in the state transitions, not the data
    # representation. 2D only.
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class BlochMambaV2(nn.Module):
    """Full Stack Bloch-Mamba v2.

    Maps (B, C, H, W) → 1D sequence via space-filling curve →
    Physics-parameterized SSM layers → reshape back.

    Supports:
    - Hilbert-curve unfolding (spatially-coherent serialization)
    - Bidirectional scanning for symmetric receptive fields
    - Configurable via model_kwargs
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int | None = None,
        base_dim: int = 64,
        num_layers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bidirectional = kwargs.get("bidirectional", True)

        # Accept kwargs from model_kwargs
        dim = kwargs.get("dim", base_dim)
        d_state = kwargs.get("d_state", dim // 2)
        num_layers = kwargs.get("num_layers", num_layers)

        self.encoder = nn.Linear(in_channels, dim)

        self.layers = nn.ModuleList([BlochSSM(dim, d_state=d_state) for _ in range(num_layers)])

        if self.bidirectional:
            self.rev_layers = nn.ModuleList(
                [BlochSSM(dim, d_state=d_state) for _ in range(num_layers)]
            )
            self.fuse = nn.Linear(dim * 2, dim)

        self.decoder = nn.Linear(dim, out_channels)
        self.norm = nn.LayerNorm(dim)

        # Cache for Hilbert curve indices, keyed by (h, w, device).
        # PERF (2026-07-01): the old cache was keyed on (h, w) only and stored
        # the tensor on CPU, so every cache HIT paid a fresh host→device copy
        # (``.to(device)``) — a recurring H2D transfer on every forward. The
        # cache now stores the device-resident tensor and moves it exactly
        # once per (shape, device).
        self._hilbert_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def _hilbert_indices(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Generate Hilbert-curve traversal order for (h, w) grid.

        Uses Z-order (Morton) as a practical approximation when dimensions
        aren't powers of two. Returns 1D index tensor of length h*w.
        """
        key = (h, w, device)
        if key in self._hilbert_cache:
            return self._hilbert_cache[key]

        # Morton Z-curve (bit-interleaving approximation of Hilbert)
        n = h * w
        ys = torch.arange(h, device=device).unsqueeze(1).expand(h, w).reshape(-1)
        xs = torch.arange(w, device=device).unsqueeze(0).expand(h, w).reshape(-1)

        # Interleave bits for Z-order curve
        z_order = torch.zeros(n, dtype=torch.long, device=device)
        for bit in range(
            max(
                h.bit_length() if isinstance(h, int) else 16,
                w.bit_length() if isinstance(w, int) else 16,
            )
        ):
            z_order |= ((xs >> bit) & 1) << (2 * bit)
            z_order |= ((ys >> bit) & 1) << (2 * bit + 1)

        # Sort by Z-order to get spatially coherent traversal
        indices = torch.argsort(z_order)
        self._hilbert_cache[key] = indices
        return indices

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass: (B, C, H, W) → (B, out_C, H, W).

        Args:
            x: Input tensor (B, C, H, W) or (B, L, C).
            **kwargs: Optional forward kwargs from strategy.

        Returns:
            Reconstructed tensor matching input spatial dimensions.
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            # Hilbert-curve unfolding for spatial coherence
            indices = self._hilbert_indices(H, W, x.device)
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            x_flat = x_flat[:, indices]  # Reorder by space-filling curve
            spatial_shape = (H, W)
        else:
            indices = None
            x_flat = x
            spatial_shape = None

        h = self.encoder(x_flat)

        # Forward scan
        h_fwd = h
        for layer in self.layers:
            h_fwd = layer(h_fwd)

        if self.bidirectional:
            # Reverse scan
            h_rev = h.flip(dims=[1])
            for layer in self.rev_layers:
                h_rev = layer(h_rev)
            h_rev = h_rev.flip(dims=[1])
            # Fuse forward + reverse
            h = self.fuse(torch.cat([h_fwd, h_rev], dim=-1))
        else:
            h = h_fwd

        h = self.norm(h)
        out = self.decoder(h)

        if spatial_shape is not None:
            B_out, L, C_out = out.shape
            H, W = spatial_shape
            # Invert Hilbert-curve reordering
            inv_indices = torch.argsort(indices)
            out = out[:, inv_indices]
            out = out.reshape(B_out, H, W, C_out).permute(0, 3, 1, 2)

        return out

    @property
    def name(self) -> str:
        return "bloch_mamba_v2"


def create_bloch_mamba_v2(
    in_channels: int = 2,
    out_channels: int = 2,
    base_dim: int = 64,
    num_layers: int = 4,
    **kwargs,
) -> BlochMambaV2:
    """Factory function for BlochMambaV2."""
    return BlochMambaV2(in_channels, out_channels, base_dim, num_layers, **kwargs)
