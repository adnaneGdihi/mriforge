import torch
import torch.nn as nn


class DifferentiableBlochSimulator(nn.Module):
    """
    Paradigm 8: Differentiable Physics Layer.
    Simulates the Bloch equations to enforce physical consistency.
    Forward: (PD, T1, T2) -> MRI Signal
    Backward: Gradients flow from Signal error to Tissue Properties.

    .. math::

        S_{SE} = PD \\cdot (1 - e^{-TR/T1}) \\cdot e^{-TE/T2}

        S_{GRE} = PD \\cdot \\sin(\alpha) \\cdot \frac{1 - e^{-TR/T1}}{1 - \\cos(\alpha)e^{-TR/T1}} \\cdot e^{-TE/T2^*}
    """

    def __init__(self, sequence_type="SE"):
        """__init__.

        Args:
            sequence_type (Any): Description.
        """
        super().__init__()
        self.sequence_type = sequence_type

    def forward(self, tissue_maps, TR, TE, alpha=None):
        """
        tissue_maps: [Batch, 3, H, W] -> (ProtonDensity, T1, T2)
        TR: Repetition Time (ms)
        TE: Echo Time (ms)
        alpha: Flip Angle (degrees), defaults to 90 if None

        forward method for DifferentiableBlochSimulator.

        Executes PyTorch tensor operations.

        Args:
            tissue_maps (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            TR (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            TE (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            alpha (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure positive values
        tissue_maps = torch.abs(tissue_maps)

        PD = tissue_maps[:, 0:1]
        T1 = tissue_maps[:, 1:2]
        T2 = tissue_maps[:, 2:3]

        # Avoid division by zero
        T1 = T1 + 1e-6
        T2 = T2 + 1e-6

        if alpha is None:
            alpha = 90.0

        # Convert alpha to radians if it's a tensor or scalar
        if isinstance(alpha, (int, float)):
            alpha_rad = torch.tensor(alpha * torch.pi / 180.0, device=tissue_maps.device)
        else:
            alpha_rad = alpha * torch.pi / 180.0

        if self.sequence_type == "SE":  # Spin Echo
            # Signal = PD * (1 - exp(-TR/T1)) * exp(-TE/T2)
            # SE typically implies 90-180 pulses, so alpha is less relevant for the basic equation
            # unless we consider partial flip angle SE which is rare.
            # We keep the standard equation.
            E1 = torch.exp(-TR / T1)
            E2 = torch.exp(-TE / T2)
            signal = PD * (1 - E1) * E2

        elif self.sequence_type == "GRE":  # Gradient Recalled Echo
            # Signal = PD * sin(alpha) * (1-E1) / (1 - cos(alpha)*E1) * exp(-TE/T2*)
            # Note: T2 here is used as T2* for simplicity unless explicitly T2* map provided
            E1 = torch.exp(-TR / T1)
            E2 = torch.exp(-TE / T2)

            sin_alpha = torch.sin(alpha_rad)
            cos_alpha = torch.cos(alpha_rad)

            # Ernst Equation
            signal = PD * sin_alpha * (1 - E1) / (1 - cos_alpha * E1 + 1e-8) * E2

        else:
            raise ValueError(f"Unknown sequence type: {self.sequence_type}")

        return signal


class ContinuousBlochODE(nn.Module):
    r"""Continuous Bloch equation ODE integrator with spin-history tracking.

    Integrates the per-voxel magnetization state :math:`(M_z, M_{xy})` over
    a sequence of N time steps, tracking through-plane motion effects.

    When tissue moves out of the imaging slice (through-plane), the new
    tissue entering has **not** experienced previous RF excitations, so its
    :math:`M_z` is at equilibrium :math:`M_0`.  This creates brightness
    fluctuations (spin-history artifacts) that cannot be captured by
    spatial-only simulation.

    Physics:
        .. math::

            \frac{dM_z}{dt} = \frac{M_0 - M_z}{T_1}

            \frac{dM_{xy}}{dt} = -\frac{M_{xy}}{T_2}

    Through-plane reset:
        When deformation field :math:`\Phi_z(t)` moves a voxel beyond
        the slice thickness, its spin state resets to equilibrium.

    Args:
        dt: Integration time step (ms).
        slice_thickness: Slice thickness in voxels.

    Reference:
        Mazaheri et al., "Characterization of spin-history artifact,"
        *Magnetic Resonance in Medicine*, 1998.
    """

    def __init__(self, dt: float = 5.0, slice_thickness: float = 5.0) -> None:
        super().__init__()
        self.dt = dt
        self.slice_thickness = slice_thickness

    def _step(
        self,
        mz: torch.Tensor,
        mxy: torch.Tensor,
        m0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single Euler integration step.

        Args:
            mz: Longitudinal magnetization ``[B, 1, H, W]``.
            mxy: Transverse magnetization ``[B, 1, H, W]``.
            m0: Equilibrium magnetization ``[B, 1, H, W]``.
            t1: T1 relaxation map (ms) ``[B, 1, H, W]``.
            t2: T2 relaxation map (ms) ``[B, 1, H, W]``.

        Returns:
            Updated (mz, mxy).
        """
        t1 = t1.clamp(min=1e-3)
        t2 = t2.clamp(min=1e-3)

        dmz = (m0 - mz) / t1
        dmxy = -mxy / t2

        mz_new = mz + self.dt * dmz
        mxy_new = mxy + self.dt * dmxy

        return mz_new, mxy_new

    def _apply_through_plane_reset(
        self,
        mz: torch.Tensor,
        mxy: torch.Tensor,
        m0: torch.Tensor,
        deformation_z: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset spin state for voxels that left/entered the slice.

        Args:
            mz: Current Mz ``[B, 1, H, W]``.
            mxy: Current Mxy ``[B, 1, H, W]``.
            m0: Equilibrium ``[B, 1, H, W]``.
            deformation_z: Through-plane displacement ``[B, 1, H, W]``.
                ``None`` means no through-plane motion.

        Returns:
            Updated (mz, mxy) with reset voxels.
        """
        if deformation_z is None:
            return mz, mxy

        # Voxels displaced beyond slice thickness → reset to equilibrium
        out_of_slice = deformation_z.abs() > self.slice_thickness
        mz = torch.where(out_of_slice, m0, mz)
        mxy = torch.where(out_of_slice, torch.zeros_like(mxy), mxy)

        return mz, mxy

    def forward(
        self,
        tissue_maps: torch.Tensor,
        n_steps: int = 50,
        tr: float = 500.0,
        deformation_z_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate Bloch equations over time with spin-history.

        Returns the full per-step transverse-magnetization trajectory (one sample
        every ``dt`` ms), NOT a single TE-sampled echo. A ``te`` argument was
        previously accepted but never read (the signal was recorded every step
        regardless) — an unread knob (pitfall #15) — so it has been removed; the
        readout granularity is the integration step ``dt``.

        Args:
            tissue_maps: ``[B, 3, H, W]`` — (PD, T1, T2) in ms.
            n_steps: Number of integration time steps.
            tr: Repetition time (ms) — RF excitation every TR ms.
            deformation_z_sequence: ``[n_steps, B, 1, H, W]`` through-plane
                displacements.  ``None`` for static imaging.

        Returns:
            Signal time series ``[n_steps, B, 1, H, W]``.

        forward method for ContinuousBlochODE.

        Executes PyTorch tensor operations.

        Args:
            tissue_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            n_steps (int): Expected input tensor.
            tr (float): Expected input tensor.
            deformation_z_sequence (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        tissue_maps = torch.abs(tissue_maps)
        m0 = tissue_maps[:, 0:1]
        t1 = tissue_maps[:, 1:2]
        t2 = tissue_maps[:, 2:3]

        # Initial state: equilibrium
        mz = m0.clone()
        mxy = torch.zeros_like(m0)

        signals = []
        time_since_rf = 0.0

        for step in range(n_steps):
            time_since_rf += self.dt

            # Check for RF excitation (every TR ms)
            if time_since_rf >= tr:
                # 90° excitation: Mxy = Mz, Mz = 0
                mxy = mz.clone()
                mz = torch.zeros_like(mz)
                time_since_rf = 0.0

            # Integrate relaxation
            mz, mxy = self._step(mz, mxy, m0, t1, t2)

            # Through-plane motion reset
            def_z = None
            if deformation_z_sequence is not None and step < len(deformation_z_sequence):
                def_z = deformation_z_sequence[step]
            mz, mxy = self._apply_through_plane_reset(mz, mxy, m0, def_z)

            # Record the transverse magnitude at this integration step (the full
            # FID/echo trajectory; readout granularity is dt).
            signals.append(mxy.abs())

        return torch.stack(signals, dim=0)  # [n_steps, B, 1, H, W]


__all__ = ["ContinuousBlochODE", "DifferentiableBlochSimulator"]
