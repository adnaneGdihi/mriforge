"""Differentiable Pulse Sequence Simulator (Diff-SEQ).

Innovation XIII from SOTA Roadmap: End-to-end sequence optimization.

Key Insight:
    Traditional MRI sequence design is manual and disconnected from reconstruction.
    Diff-SEQ implements fully differentiable Bloch simulation, enabling joint
    optimization of acquisition parameters and reconstruction network.

Mathematical Foundation:
    Bloch equations (matrix form):
        dM/dt = A(B)·M + C

    where:
        M = [Mx, My, Mz]^T (magnetization vector)
        A encodes RF excitation, gradients, and relaxation
        C = [0, 0, M0/T1]^T (recovery term)

    Sequence parameters: φ_seq = {α_i (flip angles), TE, TR, G (gradients)}

    Loss: L = L_recon + λ_SAR · SAR(φ_seq)

Benefits:
    - Data-driven sequence optimization
    - SNR efficiency improvements
    - Safety constraint enforcement (SAR)

References:
    - [27] Differentiable Bloch simulator
    - [54] Joint optimization of acquisition and reconstruction

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.registry import register_model


class BlochSimulator(nn.Module):
    """Differentiable Bloch equation simulator.

    Simulates magnetization evolution under RF pulses and gradients.

    Args:
        T1: T1 relaxation time (ms)
        T2: T2 relaxation time (ms)
        M0: Equilibrium magnetization
        dt: Time step (ms)
    """

    def __init__(
        self,
        T1: float = 1000.0,
        T2: float = 100.0,
        M0: float = 1.0,
        dt: float = 0.1,
    ) -> None:
        """__init__.

        Args:
            T1 (float): Description.
            T2 (float): Description.
            M0 (float): Description.
            dt (float): Description.
        """
        super().__init__()
        self.register_buffer("T1", torch.tensor(T1))
        self.register_buffer("T2", torch.tensor(T2))
        self.register_buffer("M0", torch.tensor(M0))
        self.dt = dt

    def forward(
        self,
        M: torch.Tensor,
        flip_angle: torch.Tensor,
        phase: torch.Tensor,
        off_resonance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Simulate one time step of Bloch dynamics.

        Args:
            M: Magnetization (batch, 3) = [Mx, My, Mz]
            flip_angle: RF flip angle (batch,) in radians
            phase: RF phase (batch,) in radians
            off_resonance: B0 off-resonance (batch,) in rad/ms

        Returns:
            Updated magnetization (batch, 3)

        forward method for BlochSimulator.

        Executes PyTorch tensor operations.

        Args:
            M (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            flip_angle (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            phase (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            off_resonance (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        Mx, My, Mz = M[:, 0], M[:, 1], M[:, 2]

        # 1. RF rotation (around axis in xy-plane at angle 'phase')
        if flip_angle is not None and flip_angle.abs().sum() > 0:
            # Rotation axis in xy-plane
            cos_p = torch.cos(phase)
            sin_p = torch.sin(phase)
            cos_a = torch.cos(flip_angle)
            sin_a = torch.sin(flip_angle)

            # Rodrigues' rotation formula for rotation by +flip_angle about the
            # in-plane axis n=(cos_p, sin_p, 0). The four sin_a off-diagonal
            # terms below previously carried flipped signs (the matrix
            # transpose), i.e. a rotation by −flip_angle: a 90°_x pulse on
            # equilibrium (0,0,1) then gave (0,+1,0) instead of the physical
            # (0,−1,0). The signs now match the standard Rodrigues matrix (and
            # biophysics/bloch.py), so RF and free-precession share handedness.
            Mx_new = (
                Mx * (cos_a + cos_p**2 * (1 - cos_a))
                + My * (cos_p * sin_p * (1 - cos_a))
                + Mz * (sin_p * sin_a)
            )
            My_new = (
                Mx * (cos_p * sin_p * (1 - cos_a))
                + My * (cos_a + sin_p**2 * (1 - cos_a))
                + Mz * (-cos_p * sin_a)
            )
            Mz_new = Mx * (-sin_p * sin_a) + My * (cos_p * sin_a) + Mz * cos_a

            Mx, My, Mz = Mx_new, My_new, Mz_new

        # 2. Free precession (off-resonance rotation + relaxation)
        if off_resonance is not None:
            theta = off_resonance * self.dt
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            Mx_prec = Mx * cos_t - My * sin_t
            My_prec = Mx * sin_t + My * cos_t
            Mx, My = Mx_prec, My_prec

        # 3. Relaxation
        E1 = torch.exp(-self.dt / self.T1)
        E2 = torch.exp(-self.dt / self.T2)

        Mx = Mx * E2
        My = My * E2
        Mz = Mz * E1 + self.M0 * (1 - E1)

        return torch.stack([Mx, My, Mz], dim=-1)

    def simulate_sequence(
        self,
        flip_angles: torch.Tensor,
        phases: torch.Tensor,
        TRs: torch.Tensor,
        num_steps_per_TR: int = 10,
    ) -> torch.Tensor:
        """Simulate full pulse sequence.

        Args:
            flip_angles: (num_pulses,) flip angles in radians
            phases: (num_pulses,) RF phases
            TRs: (num_pulses,) repetition times in ms
            num_steps_per_TR: Steps between pulses

        Returns:
            Signal (num_pulses, 2) = [Mx_signal, My_signal]
        """
        batch = flip_angles.shape[0]
        device = flip_angles.device

        # Initialize at equilibrium
        M = torch.zeros(1, 3, device=device)
        M[:, 2] = self.M0  # Mz = M0

        signals = []

        for i in range(batch):
            # Apply RF pulse
            M = self.forward(
                M,
                flip_angles[i : i + 1],
                phases[i : i + 1],
            )

            # Record signal (transverse magnetization)
            signals.append(M[:, :2].squeeze(0))

            # Free evolution during TR
            steps = int(TRs[i].item() / self.dt)
            for _ in range(min(steps, num_steps_per_TR)):
                M = self.forward(M, torch.zeros(1), torch.zeros(1))

        return torch.stack(signals, dim=0)


class LearnableSequence(nn.Module):
    """Learnable MRI pulse sequence parameters.

    Parameterizes flip angles, phases, and timing for optimization.

    Args:
        num_pulses: Number of RF pulses
        init_flip_angle: Initial flip angle (degrees)
        init_TR: Initial TR (ms)
    """

    def __init__(
        self,
        num_pulses: int = 100,
        init_flip_angle: float = 10.0,
        init_TR: float = 10.0,
    ) -> None:
        """__init__.

        Args:
            num_pulses (int): Description.
            init_flip_angle (float): Description.
            init_TR (float): Description.
        """
        super().__init__()

        # Learnable parameters (unconstrained)
        self.flip_angle_logit = nn.Parameter(
            torch.ones(num_pulses) * self._angle_to_logit(init_flip_angle)
        )
        self.phase = nn.Parameter(torch.zeros(num_pulses))
        self.TR_log = nn.Parameter(torch.ones(num_pulses) * math.log(init_TR))

    def _angle_to_logit(self, angle_deg: float) -> float:
        """Convert angle to sigmoid logit."""
        # Map [0, 180] to [0, 1], then to logit
        p = angle_deg / 180.0
        return math.log(p / (1 - p + 1e-6) + 1e-6)

    def get_flip_angles(self) -> torch.Tensor:
        """Get flip angles in radians, constrained to [0, π]."""
        return torch.sigmoid(self.flip_angle_logit) * math.pi

    def get_phases(self) -> torch.Tensor:
        """Get phases in radians."""
        return self.phase

    def get_TRs(self) -> torch.Tensor:
        """Get TRs in ms, constrained positive."""
        return torch.exp(self.TR_log)

    def compute_SAR(self) -> torch.Tensor:
        """Estimate SAR (proportional to sum of flip^2)."""
        flip_angles = self.get_flip_angles()
        return (flip_angles**2).sum()


@register_model(name="diff_seq", training_mode="reconstruction")
class DiffSEQ(nn.Module):
    """Differentiable Pulse Sequence Optimization.

    Jointly optimizes sequence parameters and reconstruction.

    Args:
        num_pulses: Number of pulses in sequence
        recon_net: Reconstruction network
        T1: T1 for simulation (ms)
        T2: T2 for simulation (ms)
        sar_weight: Weight for SAR penalty
    """

    def __init__(
        self,
        num_pulses: int = 100,
        recon_net: nn.Module | None = None,
        T1: float = 1000.0,
        T2: float = 100.0,
        sar_weight: float = 0.01,
    ) -> None:
        """__init__.

        Args:
            num_pulses (int): Description.
            recon_net (nn.Module | None): Description.
            T1 (float): Description.
            T2 (float): Description.
            sar_weight (float): Description.
        """
        super().__init__()

        self.sequence = LearnableSequence(num_pulses)
        self.simulator = BlochSimulator(T1, T2)
        self.sar_weight = sar_weight

        # Simple reconstruction network if not provided
        if recon_net is None:
            self.recon_net = nn.Sequential(
                nn.Linear(num_pulses * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2),  # Output T1, T2 estimates
            )
        else:
            self.recon_net = recon_net

    def simulate(self) -> torch.Tensor:
        """Simulate the current sequence.

        Returns:
            Signal time series (num_pulses, 2)
        """
        return self.simulator.simulate_sequence(
            self.sequence.get_flip_angles(),
            self.sequence.get_phases(),
            self.sequence.get_TRs(),
        )

    def forward(
        self,
        target_T1: torch.Tensor | None = None,
        target_T2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass with optional supervision.

        Args:
            target_T1: Ground truth T1 (optional)
            target_T2: Ground truth T2 (optional)

        Returns:
            (signal, loss_dict)

        forward method for DiffSEQ.

        Executes PyTorch tensor operations.

        Args:
            target_T1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_T2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, dict]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Simulate sequence
        signal = self.simulate()

        # Reconstruct parameters
        recon_input = signal.flatten().unsqueeze(0)
        recon_out = self.recon_net(recon_input)

        # Compute losses
        losses = {}

        # SAR penalty
        sar = self.sequence.compute_SAR()
        losses["sar"] = sar.item()

        # Reconstruction loss (if targets provided)
        if target_T1 is not None and target_T2 is not None:
            target = torch.stack([target_T1, target_T2], dim=-1)
            recon_loss = F.mse_loss(recon_out, target)
            losses["recon"] = recon_loss.item()
            total_loss = recon_loss + self.sar_weight * sar
        else:
            total_loss = self.sar_weight * sar

        losses["total"] = total_loss.item()

        return signal, losses


def create_diff_seq(
    num_pulses: int = 100,
    T1: float = 1000.0,
    T2: float = 100.0,
) -> DiffSEQ:
    """Factory function for Diff-SEQ.

    Args:
        num_pulses: Number of pulses
        T1: Tissue T1 (ms)
        T2: Tissue T2 (ms)

    Returns:
        Configured Diff-SEQ model
    """
    return DiffSEQ(
        num_pulses=num_pulses,
        T1=T1,
        T2=T2,
    )


__all__ = [
    "BlochSimulator",
    "DiffSEQ",
    "LearnableSequence",
    "create_diff_seq",
]
