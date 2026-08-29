import torch
import torch.nn as nn


class BlochSimulator(nn.Module):
    """
    Differentiable Bloch Simulator for MRI.

    Maps tissue properties (M0, T1, T2) and sequence parameters (TE, TR, Flip Angle)
    to complex transverse magnetization signal.
    """

    def __init__(self):
        """__init__."""
        super().__init__()

    def forward(
        self,
        m0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        sequence: str = "spin_echo",
        **kwargs,
    ) -> torch.Tensor:
        """
        Simulate MRI signal based on tissue properties and sequence parameters.

        Args:
            m0: Proton density [B, ...].
            t1: T1 map [B, ...].
            t2: T2 map [B, ...].
            sequence: Sequence type ('spin_echo', 'gradient_echo', 'inversion_recovery').
            **kwargs: Sequence parameters:
                - te: Echo Time (ms)
                - tr: Repetition Time (ms)
                - ti: Inversion Time (ms)
                - t2_star: T2* map (ms) [default: t2]
                - alpha_rad: Flip angle (radians)

        Returns:
            Signal intensity [B, ...].

        forward method for BlochSimulator.

        Executes PyTorch tensor operations.

        Args:
            m0 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sequence (str): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if m0.dim() == 5:
            return self.simulate_3d(m0, t1, t2, sequence, **kwargs)

        if sequence == "spin_echo":
            return self.spin_echo(m0, t1, t2, kwargs.get("te", 0.0), kwargs.get("tr", 1000.0))
        elif sequence == "gradient_echo":
            t2_star = kwargs.get("t2_star", t2)
            return self.gradient_echo(
                m0,
                t1,
                t2_star,
                kwargs.get("te", 0.0),
                kwargs.get("tr", 1000.0),
                kwargs.get("alpha_rad", 0.0),
            )
        elif sequence == "inversion_recovery":
            return self.inversion_recovery(m0, t1, kwargs.get("ti", 0.0), kwargs.get("tr", 1000.0))
        else:
            raise NotImplementedError(
                f"Sequence type '{sequence}' not supported by forward(). Use specific methods or 'bloch_step_matrix_exp'."
            )

    def simulate_3d(
        self,
        m0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        sequence: str = "spin_echo",
        **kwargs,
    ) -> torch.Tensor:
        """
        Simulate 3D MRI signal using slice-by-slice iteration.

        Args:
            m0: Proton density [B, C, D, H, W].
            t1: T1 map [B, C, D, H, W].
            t2: T2 map [B, C, D, H, W].
            sequence: Sequence type.
            **kwargs: Sequence parameters.

        Returns:
            Simulated 3D signal intensity [B, C, D, H, W].
        """
        if m0.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B, C, D, H, W], got {m0.dim()}D")

        # Process slice-by-slice along depth dimension (D)
        B, C, D, H, W = m0.shape
        out_slices = []

        for d in range(D):
            m0_slice = m0[:, :, d, :, :]
            t1_slice = t1[:, :, d, :, :]
            t2_slice = t2[:, :, d, :, :]

            # Calling forward directly without dimension expansion to process as 2D
            signal_slice = self.forward(m0_slice, t1_slice, t2_slice, sequence=sequence, **kwargs)
            out_slices.append(signal_slice)

        return torch.stack(out_slices, dim=2)

    @staticmethod
    def spin_echo(
        m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, te: float, tr: float
    ) -> torch.Tensor:
        """
        Analytical Spin Echo signal equation.
        S = M0 * (1 - exp(-TR/T1)) * exp(-TE/T2)

        Args:
            m0: (B, N, 1) Proton density.
            t1: (B, N, 1) Longitudinal relaxation time (ms).
            t2: (B, N, 1) Transverse relaxation time (ms).
            te: Echo Time (ms).
            tr: Repetition Time (ms).

        Returns:
            signal: (B, N, 1) Scalar signal intensity (assumed real for SE basics, or can return complex Mxy).
                    Usually SE is magnitude unless phase is added.
                    We return magnitude here, assuming phase is 0 or modeled separately.
        """
        # Avoid division by zero
        t1 = torch.clamp(t1, min=1e-6)
        t2 = torch.clamp(t2, min=1e-6)

        e1 = torch.exp(-tr / t1)
        e2 = torch.exp(-te / t2)

        signal = m0 * (1 - e1) * e2
        return signal

    @staticmethod
    def inversion_recovery(
        m0: torch.Tensor, t1: torch.Tensor, ti: float, tr: float
    ) -> torch.Tensor:
        """
        Inversion Recovery signal equation.
        S = M0 * (1 - 2*exp(-TI/T1) + exp(-TR/T1))
        """
        t1 = torch.clamp(t1, min=1e-6)
        e_ti = torch.exp(-ti / t1)
        e_tr = torch.exp(-tr / t1)

        signal = m0 * (1 - 2 * e_ti + e_tr)

        # Taking absolute value for magnitude reconstruction,
        # but for complex processing we might keep the sign (real-valued inversion).
        return signal

    @staticmethod
    def gradient_echo(
        m0: torch.Tensor,
        t1: torch.Tensor,
        t2_star: torch.Tensor,
        te: float,
        tr: float,
        alpha_rad: float,
    ) -> torch.Tensor:
        """
        Spoiled Gradient Echo (FLASH/SPGR) signal equation.
        S = M0 * sin(alpha) * (1 - exp(-TR/T1)) / (1 - cos(alpha)*exp(-TR/T1)) * exp(-TE/T2*)
        """
        t1 = torch.clamp(t1, min=1e-6)
        t2_star = torch.clamp(t2_star, min=1e-6)

        e1 = torch.exp(-tr / t1)
        e2 = torch.exp(-te / t2_star)

        sin_a = torch.sin(torch.tensor(alpha_rad, device=m0.device))
        cos_a = torch.cos(torch.tensor(alpha_rad, device=m0.device))

        numerator = m0 * sin_a * (1 - e1)
        denominator = 1 - cos_a * e1

        # Clamp denominator
        denominator = torch.clamp(denominator, min=1e-6)

        signal = (numerator / denominator) * e2
        return signal

    @staticmethod
    def bloch_step_matrix_exp(
        m: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        dt: float,
        df: float = 0.0,
        b1_phase: float = 0.0,
        flip_angle: float = 0.0,
    ) -> torch.Tensor:
        """
        [FIX 14] Matrix Exponential Bloch Solver.

        Solves the Bloch equations analytically for piecewise-constant pulses
        using rotational frame decomposition. This avoids numerical instability
        from stiff ODEs (RF pulses are µs, relaxation is ms/s).

        Args:
            m: Magnetization [B, N, 3] or [B, 3] with (Mx, My, Mz)
            t1: Longitudinal relaxation time [B, N] or [B]
            t2: Transverse relaxation time [B, N] or [B]
            dt: Time step (ms)
            df: Off-resonance frequency (Hz)
            b1_phase: RF field phase (radians)
            flip_angle: RF flip angle (radians)

        Returns:
            Updated magnetization [B, N, 3] or [B, 3]

        Reference: Hargreaves, B. A., et al. (2001). "MRM spin dynamics simulation."
        """
        device = m.device
        dtype = m.dtype

        # Ensure shapes are compatible
        if m.dim() == 2:
            m = m.unsqueeze(1)  # [B, 1, 3]
        if t1.dim() == 1:
            t1 = t1.unsqueeze(-1)
        if t2.dim() == 1:
            t2 = t2.unsqueeze(-1)

        # Clamp to avoid division by zero
        t1 = torch.clamp(t1, min=1e-6)
        t2 = torch.clamp(t2, min=1e-6)

        # 1. Relaxation (Decay terms)
        E1 = torch.exp(-dt / t1)  # [B, N] or [B, 1]
        E2 = torch.exp(-dt / t2)

        # 2. Precession (Off-resonance rotation about Z)
        # Rotation angle = 2*pi*df*dt (rad)
        phi = 2 * torch.pi * df * dt * 1e-3  # Convert dt from ms to s
        cos_phi = torch.cos(torch.tensor(phi, device=device, dtype=dtype))
        sin_phi = torch.sin(torch.tensor(phi, device=device, dtype=dtype))

        # 3. RF Pulse (Rotation about axis in XY plane)
        if flip_angle != 0.0:
            cos_fa = torch.cos(torch.tensor(flip_angle, device=device, dtype=dtype))
            sin_fa = torch.sin(torch.tensor(flip_angle, device=device, dtype=dtype))
            cos_ph = torch.cos(torch.tensor(b1_phase, device=device, dtype=dtype))
            sin_ph = torch.sin(torch.tensor(b1_phase, device=device, dtype=dtype))
        else:
            cos_fa = torch.tensor(1.0, device=device, dtype=dtype)
            sin_fa = torch.tensor(0.0, device=device, dtype=dtype)
            cos_ph = torch.tensor(1.0, device=device, dtype=dtype)
            sin_ph = torch.tensor(0.0, device=device, dtype=dtype)

        # Extract components
        Mx, My, Mz = m[..., 0], m[..., 1], m[..., 2]

        # Apply RF rotation about the transverse axis at azimuth ``b1_phase``,
        # n = (cos φ, sin φ, 0), by ``flip_angle`` (Rodrigues' formula
        # M' = M cosα + (n×M) sinα + n(n·M)(1−cosα)). For φ=0 this reduces to the
        # X-axis rotation; ``b1_phase`` was previously COMPUTED but ignored (the
        # RF axis was hardcoded to X), making the phase knob a silent no-op.
        if flip_angle != 0.0:
            n_dot_m = Mx * cos_ph + My * sin_ph
            Mx_rf = Mx * cos_fa + sin_ph * Mz * sin_fa + cos_ph * n_dot_m * (1 - cos_fa)
            My_rf = My * cos_fa - cos_ph * Mz * sin_fa + sin_ph * n_dot_m * (1 - cos_fa)
            Mz_rf = Mz * cos_fa + (cos_ph * My - sin_ph * Mx) * sin_fa
        else:
            Mx_rf, My_rf, Mz_rf = Mx, My, Mz

        # Apply precession (rotation about Z)
        Mx_pre = Mx_rf * cos_phi - My_rf * sin_phi
        My_pre = Mx_rf * sin_phi + My_rf * cos_phi
        Mz_pre = Mz_rf

        # Apply relaxation
        E2_exp = E2.unsqueeze(-1) if E2.dim() < Mx_pre.dim() else E2
        E1_exp = E1.unsqueeze(-1) if E1.dim() < Mz_pre.dim() else E1

        Mx_new = Mx_pre * E2_exp.squeeze(-1)
        My_new = My_pre * E2_exp.squeeze(-1)
        Mz_new = Mz_pre * E1_exp.squeeze(-1) + (1 - E1_exp.squeeze(-1))  # Recovery to M0=1

        # Recombine
        m_new = torch.stack([Mx_new, My_new, Mz_new], dim=-1)

        return m_new.squeeze(1) if m_new.shape[1] == 1 else m_new
