"""Physics-Informed Neural Network (PINN) module for MRI reconstruction.

This module implements the core logic for integrating Partial Differential Equation (PDE)
constraints into the reconstruction process. It defines the interface for PDEs and
provides implementations for common physical models used in MRI.
"""

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class PDE(nn.Module):
    """Abstract base class for Partial Differential Equations."""

    @abc.abstractmethod
    def compute_residual(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute the PDE residual.

        Args:
            u: The solution variable (e.g., image intensity or magnetization).
            coords: The coordinates (x, y, z, t) where the residual is evaluated.

        Returns:
            The PDE residual (should be zero if the solution satisfies the PDE).
        """
        pass


class WaveEquation(PDE):
    """Wave equation PDE: u_tt - c^2 * laplacian(u) = 0.
    In frequency domain (Helmholtz): laplacian(u) + k^2 * u = 0.
    """

    def __init__(self, c: float = 1.0, k: float = 0.0):
        """Initialize the Wave Equation.

        Args:
            c: Wave propagation speed.
            k: Wavenumber for Helmholtz equation (k = omega/c).
        """
        super().__init__()
        self.c = c
        self.k = k
        # Register as buffer: Persistent on GPU, saved with model, no gradient
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", laplacian)

    def compute_residual(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute the wave equation residual.

        Note: This is a simplified implementation assuming u is a function of coords.
        In a full PINN, we would need automatic differentiation with respect to coords.
        For image-based constraints, we might approximate derivatives using finite differences
        or spectral methods if coords are implicit (grid indices).

        Here, we assume 'u' is a 2D/3D image/volume and we compute spatial derivatives
        using finite differences (kernels). Temporal derivatives would require a time dimension.
        """
        # Compute Laplacian
        lap = self._laplacian_2d(u)

        # If k is non-zero, apply Helmholtz equation: lap(u) + k^2 * u = 0
        if self.k != 0:
            return lap + (self.k**2) * u

        # Otherwise assume static Laplace equation: lap(u) = 0
        return lap

    def _laplacian_2d(self, u: torch.Tensor) -> torch.Tensor:
        """Compute 2D Laplacian using finite differences."""
        # Handle multiple channels
        b, c, h, w = u.shape
        # Expand kernel for channels - Zero Copy (View)
        # No new allocation, just metadata manipulation
        kernel_expanded = self.kernel.expand(c, -1, -1, -1)

        return F.conv2d(u, kernel_expanded, padding=1, groups=c)


class BlochEquation(PDE):
    """Bloch equation for MRI magnetization dynamics.

    Implements steady-state signal equation for Gradient Echo (GRE) sequences:
    S ≈ ρ * sin(α) * (1 - e^(-TR/T1)) / (1 - cos(α) * e^(-TR/T1)) * e^(-TE/T2*)
    """

    def __init__(
        self,
        T1: float = 1000.0,
        T2: float = 100.0,
        TR: float = 50.0,
        TE: float = 10.0,
        flip_angle: float = 15.0,
    ):
        """Initialize the Bloch Equation parameters.

        Args:
            T1: Longitudinal relaxation time (ms)
            T2: Transverse relaxation time (ms)
            TR: Repetition time (ms)
            TE: Echo time (ms)
            flip_angle: Flip angle (degrees)
        """
        super().__init__()
        self.T1 = T1
        self.T2 = T2
        self.TR = TR
        self.TE = TE
        self.alpha = torch.deg2rad(torch.tensor(flip_angle))

    def compute_residual(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute Bloch equation residual (steady-state signal consistency).

        Args:
            u: Predicted signal intensity [B, 1, H, W] (real or complex magnitude)
            coords: Quantitative maps tensor [B, 3, H, W] containing (PD, T1, T2*)
                    If absent or malformed, falls back to spatial smoothness.
        """
        # If no T1/T2 maps provided, fall back to smoothness regularization
        if coords is None or coords.numel() == 0 or coords.shape[1] < 3:
            return self._laplacian_2d(u)

        # Extract quantitative physical maps
        pd = coords[:, 0:1, ...]
        t1 = coords[:, 1:2, ...]
        t2 = coords[:, 2:3, ...]

        # Numerically stable clamping to prevent div/zero
        t1 = torch.clamp(t1, min=1e-3)
        t2 = torch.clamp(t2, min=1e-3)

        # Build exponential relaxation factors
        e1 = torch.exp(-self.TR / t1)
        e2 = torch.exp(-self.TE / t2)

        # Alpha projections
        sin_alpha = torch.sin(self.alpha.to(u.device))
        cos_alpha = torch.cos(self.alpha.to(u.device))

        # Analytical Bloch equation for spoiled GRE steady-state:
        # S = PD * sin(a) * (1 - E1) / (1 - cos(a) * E1) * E2
        steady_state_num = sin_alpha * (1.0 - e1)
        steady_state_den = 1.0 - cos_alpha * e1
        signal_bloch = pd * (steady_state_num / steady_state_den) * e2

        # In physics, we compute the magnitude of the signal
        # so if `u` is complex, we evaluate consistency of its magnitude.
        u_mag = u.abs() if u.is_complex() else u

        # Loss residual enforces the image physics mathematically
        return u_mag - signal_bloch

    def _laplacian_2d(self, u: torch.Tensor) -> torch.Tensor:
        """Compute 2D Laplacian for smoothness regularization as a baseline PINN."""
        u_real = u.real if u.is_complex() else u
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=u_real.dtype, device=u_real.device
        ).view(1, 1, 3, 3)
        b, c, h, w = u_real.shape
        kernel_expanded = kernel.expand(c, -1, -1, -1)
        lap = F.conv2d(u_real, kernel_expanded, padding=1, groups=c)
        if u.is_complex():
            u_imag = u.imag
            kernel_expanded_imag = kernel.expand(c, -1, -1, -1)
            lap_imag = F.conv2d(u_imag, kernel_expanded_imag, padding=1, groups=c)
            return torch.complex(lap, lap_imag)
        return lap


class MaxwellRegularizer(PDE):
    """Enforce electromagnetic constraints on coil sensitivity maps.

    The B1- field (receive sensitivity) must satisfy Maxwell's equations:
    - ∇·B = 0 (divergence-free)
    - ∇²B ≈ 0 (harmonic in source-free tissue regions)

    This regularizer penalizes deviations from the harmonic constraint,
    which physically corresponds to smooth, electromagnetically consistent
    sensitivity profiles.

    Reference: Pruessmann et al., "SENSE" (1999)
    """

    def __init__(self, weight: float = 1.0):
        """Initialize MaxwellRegularizer.

        Args:
            weight: Regularization weight (higher = stronger smoothness)
        """
        super().__init__()
        self.weight = weight
        # Laplacian kernel for harmonic constraint
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", laplacian)

    def compute_residual(
        self, sensitivity_map: torch.Tensor, coords: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute Maxwell constraint residual on sensitivity maps.

        Args:
            sensitivity_map: Coil sensitivity maps [B, C, H, W] where C = num_coils * 2
                            (real/imag pairs) or [B, num_coils, H, W] for complex
            coords: Ignored (spatial coords are implicit in the grid)

        Returns:
            Laplacian residual - should be near zero for valid sensitivities
        """
        # Enforce Harmonic constraint: ∇²S ≈ 0
        b, c, h, w = sensitivity_map.shape
        kernel_expanded = self.kernel.to(sensitivity_map.device).expand(c, -1, -1, -1)
        laplacian = F.conv2d(sensitivity_map, kernel_expanded, padding=1, groups=c)
        return self.weight * laplacian


class PINNModule(nn.Module):
    """Module for computing Physics-Informed losses."""

    def __init__(self, pde: PDE, weight: float = 1.0):
        """__init__.

        Args:
            pde (PDE): Description.
            weight (float): Description.
        """
        super().__init__()
        self.pde = pde
        self.weight = weight

    def forward(self, u: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the weighted PDE residual loss.

        Args:
            u: The predicted solution (image/volume).
            coords: Optional coordinates.

        Returns:
            Scalar loss value.

        forward method for PINNModule.

        Executes PyTorch tensor operations.

        Args:
            u (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = self.pde.compute_residual(u, coords if coords is not None else torch.empty(0))
        return self.weight * torch.mean(residual**2)


def get_pde(name: str, **kwargs) -> PDE:
    """Factory function to get a PDE instance."""
    if name == "wave_equation":
        return WaveEquation(**kwargs)
    elif name == "bloch_equation":
        return BlochEquation(**kwargs)
    elif name == "maxwell_regularizer":
        return MaxwellRegularizer(**kwargs)
    else:
        raise ValueError(
            f"Unknown PDE type: {name}. Available: wave_equation, bloch_equation, maxwell_regularizer"
        )
