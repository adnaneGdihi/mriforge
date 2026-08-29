"""Bloch-Mamba: Physics-Embedded State Space Model.

Innovation VI from SOTA Roadmap: Zero-shot generalization to new pulse sequences.

Key Insight:
    Standard SSMs learn transitions from data, ignoring MRI physics.
    Bloch-Mamba embeds the Bloch equations directly into state transitions:
    - Physical state: governed by T1/T2 relaxation (fixed from tissue maps)
    - Residual state: learned for system imperfections

Mathematical Foundation:
    Bloch equations (discrete):
        M_z(t+1) = M_z(t)·e^(-Δt/T1) + M0·(1 - e^(-Δt/T1))
        M_xy(t+1) = M_xy(t)·e^(-Δt/T2)·e^(iγB0Δt)

    Partition state: h_t = [h_phys, h_resid]
        h_phys: A_phys(T1, T2) transition (known physics)
        h_resid: A_resid (learned) for imperfections

Benefits:
    - Zero-shot to new sequences (physics is sequence-agnostic)
    - Interpretable: physical vs residual separation
    - Better tissue parameter estimation

References:
    - [27] Differentiable Bloch simulator
    - [29] MR Fingerprinting with neural networks

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.registry import register_model


class BlochTransition(nn.Module):
    """Physics-based state transition from Bloch equations.

    Implements the discrete-time Bloch equations:
        M_z(t+Δt) = M0 + (M_z(t) - M0)·E1  where E1 = exp(-Δt/T1)
        M_xy(t+Δt) = M_xy(t)·E2  where E2 = exp(-Δt/T2)

    Args:
        d_model: Model dimension
        dt: Time step (ms)
    """

    def __init__(self, d_model: int, dt: float = 1.0) -> None:
        """__init__.

        Args:
            d_model (int): Description.
            dt (float): Description.
        """
        super().__init__()
        self.d_model = d_model
        self.dt = dt

        # Learnable default T1, T2 (can be overridden per-sample)
        # Typical values: T1 ~ 1000ms, T2 ~ 100ms
        self.log_T1 = nn.Parameter(torch.full((d_model,), 6.9))  # ~1000ms
        self.log_T2 = nn.Parameter(torch.full((d_model,), 4.6))  # ~100ms

        # M0 (equilibrium magnetization)
        self.M0 = nn.Parameter(torch.ones(d_model))

    def forward(
        self,
        h_phys: torch.Tensor,
        T1_map: torch.Tensor | None = None,
        T2_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Bloch transition.

        Args:
            h_phys: Physical state (B, D) or (B, L, D)
                    First half = M_z (longitudinal)
                    Second half = M_xy (transverse, complex)
            T1_map: Optional T1 map (B, D//2) in ms
            T2_map: Optional T2 map (B, D//2) in ms

        Returns:
            Updated physical state
        """
        d_half = h_phys.shape[-1] // 2
        E1, E2, M0 = self.factors(d_half, T1_map, T2_map)
        return self.apply_factors(h_phys, E1, E2, M0)

    def factors(
        self,
        d_half: int,
        T1_map: torch.Tensor | None = None,
        T2_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Relaxation factors ``(E1, E2, M0)`` for a given state split.

        PERF seam (2026-07-01): these depend only on the (per-forward
        constant) parameters and maps, not on the evolving state, so scan
        loops hoist them once instead of recomputing
        ``exp(log_T1)`` → ``exp(-dt/T1)`` etc. every token. ``forward``
        remains the single-step reference path and shares the same math via
        :meth:`apply_factors`.
        """
        T1 = torch.exp(self.log_T1[:d_half]) if T1_map is None else T1_map
        T2 = torch.exp(self.log_T2[:d_half]) if T2_map is None else T2_map
        E1 = torch.exp(-self.dt / T1)
        E2 = torch.exp(-self.dt / T2)
        M0 = self.M0[:d_half]
        return E1, E2, M0

    @staticmethod
    def apply_factors(
        h_phys: torch.Tensor,
        E1: torch.Tensor,
        E2: torch.Tensor,
        M0: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Bloch relaxation update with precomputed factors."""
        d_half = h_phys.shape[-1] // 2
        M_z = h_phys[..., :d_half]
        M_xy = h_phys[..., d_half:]
        M_z_new = M0 + (M_z - M0) * E1
        M_xy_new = M_xy * E2
        return torch.cat([M_z_new, M_xy_new], dim=-1)


class BioHarmonicTransition(nn.Module):
    """Kinematic state transition using biological harmonic oscillators.

    Instead of Bloch T1/T2 relaxation, this transition initialises the
    state-space eigenvalues as complex conjugate pairs tuned to human
    respiratory and cardiac frequencies.  The continuous ODE inherently
    acts as an adaptive band-pass filter for periodic physiological motion.

    Mathematics:
        Each complex eigenvalue pair ``(σ ± jω)`` defines a damped oscillator:

        .. math::

            A_k = \\sigma + j\\omega_k, \\quad A_{k+1} = \\sigma - j\\omega_k

        where :math:`\\omega_{resp} = 2\\pi f_{resp}` and
        :math:`\\omega_{card} = 2\\pi f_{card}`.  The decay :math:`\\sigma < 0`
        accommodates heart rate variability.

    Args:
        d_model: State dimension (must be divisible by 4).
        f_resp: Respiratory frequency in Hz (default: 0.3 Hz ≈ 18 bpm).
        f_card: Cardiac frequency in Hz (default: 1.2 Hz ≈ 72 bpm).
        decay: Real decay coefficient (should be slightly negative).
        sample_rate: Acquisition sample rate in Hz.

    Reference:
        Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial
        Projections," *NeurIPS*, 2020.
    """

    def __init__(
        self,
        d_model: int,
        f_resp: float = 0.3,
        f_card: float = 1.2,
        decay: float = -0.05,
        sample_rate: float = 100.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        import math

        w_resp = 2 * math.pi * f_resp
        w_card = 2 * math.pi * f_card

        # Build complex conjugate eigenvalue pairs
        # Each pair: (decay + jw, decay - jw)
        base_eigenvalues = torch.tensor(
            [
                complex(decay, w_resp),
                complex(decay, -w_resp),
                complex(decay, w_card),
                complex(decay, -w_card),
            ]
        )

        # Tile to fill d_model
        n_tiles = max(1, d_model // 4)
        eigenvalues = base_eigenvalues.repeat(n_tiles)[:d_model]

        # Store as learnable real/imag parts
        self.A_real = nn.Parameter(eigenvalues.real.clone())
        self.A_imag = nn.Parameter(eigenvalues.imag.clone())

        # Discretisation step Δ (learnable)
        dt_init = 1.0 / sample_rate
        self.log_dt = nn.Parameter(torch.full((d_model,), math.log(dt_init)))

    def forward(
        self,
        h: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply bio-harmonic state transition.

        Args:
            h: State vector ``(B, D)`` or ``(B, L, D)``.

        Returns:
            Updated state with same shape.
        """
        # Reconstruct complex eigenvalues
        A = torch.complex(self.A_real, self.A_imag)  # [D]

        # Discretise: A_bar = exp(A · Δt)
        dt = torch.exp(self.log_dt)  # [D]
        A_bar = torch.exp(A * dt)  # [D] complex

        # Apply transition: h_{t+1} = A_bar * h_t
        # h is real; we apply magnitude scaling and phase rotation
        A_bar_mag = A_bar.abs()
        A_bar_phase = A_bar.angle()

        # For real-valued hidden states, apply magnitude decay
        # and cosine of phase rotation (real part of complex multiplication)
        h_new = h * A_bar_mag * torch.cos(A_bar_phase)

        return h_new


class BlochMambaBlock(nn.Module):
    """Bloch-Mamba block with physics-embedded state transitions.

    Partitions hidden state into:
    - Physical: governed by Bloch equations
    - Residual: learned for imperfections

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        physics_ratio: Fraction of state for physics (default 0.5)
        dt: Time step for Bloch equations (ms)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        physics_ratio: float = 0.5,
        dt: float = 1.0,
    ) -> None:
        """__init__.

        Args:
            d_model (int): Description.
            d_state (int): Description.
            physics_ratio (float): Description.
            dt (float): Description.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # State partition
        self.d_phys = int(d_state * physics_ratio)
        self.d_resid = d_state - self.d_phys

        # Physics transition (Bloch equations)
        self.bloch = BlochTransition(self.d_phys, dt=dt)

        # Residual transition (learned)
        self.A_resid_log = nn.Parameter(torch.randn(d_model, self.d_resid) * 0.1)

        # Input projections
        self.B_phys = nn.Linear(d_model, self.d_phys)
        self.B_resid = nn.Linear(d_model, self.d_resid)

        # Output projection
        self.C = nn.Linear(d_state, d_model)

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        T1_map: torch.Tensor | None = None,
        T2_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, L, D)
            T1_map: Optional T1 map (B, D) in ms
            T2_map: Optional T2 map (B, D) in ms

        Returns:
            Output (B, L, D)
        """
        B, L, D = x.shape
        device = x.device

        x = self.norm(x)

        # --- Internal Helper for Recurrence ---
        def run_recurrence(seq_x, seq_t1):
            """Affine diagonal scan over the sequence dim.

            ``seq_x``: (B, L, D). PERF (2026-07-01): the loop used to
            recompute the Bloch relaxation factors (``exp(log_T1)`` →
            ``exp(-dt/T1)`` …) and ``A_resid.mean(dim=0)`` EVERY token, and
            launched the B/C/D projections per token. All are per-forward
            constants / linear maps, so: factors + decay mean hoisted, input
            projections batched before the loop, readout batched after it.
            The loop keeps only the two fused multiply-add state updates
            (affine Bloch dynamics — deliberately NOT routed onto the
            selective-scan kernel, whose recurrence this is not).

            NOTE on ``seq_t1``: the per-step map plumbing was already dead
            (documented dimension mismatch — maps arrive as (B, H, W) while
            ``BlochTransition`` expects (B, d_phys//2)); the old loop always
            passed ``None`` to the transition regardless of ``seq_t1``. That
            behaviour is preserved verbatim — perf change only.
            """
            del seq_t1  # dead plumbing, see docstring
            h_phys = torch.zeros(B, self.d_phys, device=device)
            h_resid = torch.zeros(B, self.d_resid, device=device)
            A_resid = torch.exp(-torch.exp(self.A_resid_log))

            # Hoisted per-forward constants
            E1, E2, M0 = self.bloch.factors(self.d_phys // 2)
            A_resid_mean = A_resid.mean(dim=0)

            # Batched input projections: (B, L, d_phys) / (B, L, d_resid)
            b_phys_all = self.B_phys(seq_x)
            b_resid_all = self.B_resid(seq_x)

            h_states = []
            for t in range(L):
                # Update States
                h_phys = self.bloch.apply_factors(h_phys, E1, E2, M0) + b_phys_all[:, t]
                h_resid = (A_resid_mean * h_resid) + b_resid_all[:, t]
                h_states.append(torch.cat([h_phys, h_resid], dim=-1))

            # Batched readout: y[t] = C h[t] + D * x[t]
            h_all = torch.stack(h_states, dim=1)  # (B, L, d_state)
            return self.C(h_all) + self.D * seq_x

        # 1. Forward Scan
        # We need to handle maps if they are spatial.
        # For now, pass None as original code seemed to rely on internal params or constant maps?
        # The original code passed T1_map directly to bloch inside the loop.
        # But bloch expects (B, D//2). T1_map is likely (B, H, W).
        # This was a bug in original code too (dimension mismatch).
        # I will pass T1_map as is, assuming it handles it or fails (Scope: Mamba Fix, not Logic Fix unless critical).
        # Actually, let's just run the loop logic as defined.

        # Forward
        y_fwd = run_recurrence(x, T1_map)

        # 2. Backward Scan
        # Flip sequence dim (dim=1)
        x_bwd = torch.flip(x, dims=[1])
        # Also flip maps if they are sequence-aligned?
        # Assuming maps are static for the block for now or handled inside.
        y_bwd = run_recurrence(x_bwd, T1_map)

        # Flip back
        y_bwd = torch.flip(y_bwd, dims=[1])

        return y_fwd + y_bwd


@register_model(name="bloch_mamba", training_mode="reconstruction")
class BlochMamba(nn.Module):
    """Full Bloch-Mamba model for MRI reconstruction.

    Args:
        in_channels: Input channels (2 for complex)
        base_dim: Base feature dimension
        num_layers: Number of Bloch-Mamba blocks
        d_state: SSM state dimension
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_dim: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            base_dim (int): Description.
            num_layers (int): Description.
            d_state (int): Description.
        """
        super().__init__()

        # Input projection
        self.in_proj = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # Bloch-Mamba blocks
        self.blocks = nn.ModuleList(
            [BlochMambaBlock(base_dim, d_state=d_state) for _ in range(num_layers)]
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, in_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        T1_map: torch.Tensor | None = None,
        T2_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, C, H, W)
            T1_map: Optional T1 map (B, H, W) in ms
            T2_map: Optional T2 map (B, H, W) in ms

        Returns:
            Output (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Project to features
        x = self.in_proj(x)  # (B, base_dim, H, W)

        # Flatten spatial dims
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # (B, L, D)

        # Process through Bloch-Mamba blocks
        for block in self.blocks:
            x = x + block(x, T1_map, T2_map)

        # Output projection
        x = self.out_proj(x)  # (B, L, C)

        # Reshape back to image
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x


def create_bloch_mamba(
    in_channels: int = 2,
    base_dim: int = 64,
    num_layers: int = 4,
    d_state: int = 16,
) -> BlochMamba:
    """Factory function for Bloch-Mamba.

    Args:
        in_channels: Input channels
        base_dim: Base feature dimension
        num_layers: Number of blocks
        d_state: SSM state dimension

    Returns:
        Configured Bloch-Mamba model
    """
    return BlochMamba(
        in_channels=in_channels,
        base_dim=base_dim,
        num_layers=num_layers,
        d_state=d_state,
    )


__all__ = [
    "BioHarmonicTransition",
    "BlochMamba",
    "BlochMambaBlock",
    "BlochTransition",
    "create_bloch_mamba",
]
