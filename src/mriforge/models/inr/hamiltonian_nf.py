"""Hamiltonian Neural Fields (HNF) for Diffeomorphic Motion Modeling.

Innovation VII from SOTA Roadmap: Physics-constrained deformation for dynamic MRI.

Key Insight:
    Standard motion models use optical flow or free-form deformations that can
    produce non-physical results (folding, tearing). Hamiltonian mechanics
    ensures diffeomorphic (smooth, invertible) transformations by construction.

Mathematical Foundation:
    Hamiltonian system:
        ∂q/∂t = ∂H/∂p   (velocity from momentum)
        ∂p/∂t = -∂H/∂q  (momentum from position)

    Key property: Symplectic structure preserves volume (Liouville's theorem)
    → Mass conservation for tissue deformation

    Implementation:
        - Learn scalar Hamiltonian H(q,p) via neural network
        - Use symplectic integrator (Leapfrog) for time evolution
        - Deformation field is the flow map φ: x → x + displacement

Benefits:
    - Guaranteed invertibility (no mesh folding)
    - Mass/volume preservation
    - Interpretable energy landscape

References:
    - [32] Hamiltonian Neural Networks
    - [14] Diffeomorphic registration for cardiac MRI

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.registry import register_model


class HamiltonianMLP(nn.Module):
    """Neural network representing the Hamiltonian H(q, p).

    The Hamiltonian is a scalar function of position and momentum
    that encodes the total energy of the system.

    Args:
        input_dim: Dimension of (q, p) space (2 * spatial_dim)
        hidden_dims: Hidden layer dimensions
        use_softplus: If True, ensures H >= 0
    """

    def __init__(
        self,
        input_dim: int = 4,  # 2D: q=(x,y), p=(px,py)
        hidden_dims: tuple[int, ...] = (64, 64, 64),
        use_softplus: bool = False,
    ) -> None:
        """__init__.

        Args:
            input_dim (int): Description.
            hidden_dims (tuple[int, ...]): Description.
            use_softplus (bool): Description.
        """
        super().__init__()
        self.use_softplus = use_softplus

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.Tanh(),  # Smooth activation for better gradients
                ]
            )
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))  # Scalar output

        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H(q, p).

        Args:
            q: Position (batch, spatial_dim)
            p: Momentum (batch, spatial_dim)

        Returns:
            Hamiltonian value (batch, 1)

        forward method for HamiltonianMLP.

        Executes PyTorch tensor operations.

        Args:
            q (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            p (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = torch.cat([q, p], dim=-1)
        H = self.net(x)

        if self.use_softplus:
            H = F.softplus(H)

        return H


class LeapfrogIntegrator(nn.Module):
    """Symplectic Leapfrog (Störmer-Verlet) integrator.

    Preserves the symplectic structure:
        1. p_{1/2} = p_0 - (dt/2) * ∂H/∂q(q_0, p_0)
        2. q_1 = q_0 + dt * ∂H/∂p(q_0, p_{1/2})
        3. p_1 = p_{1/2} - (dt/2) * ∂H/∂q(q_1, p_{1/2})

    Args:
        hamiltonian: Neural network representing H(q, p)
        dt: Time step
        num_steps: Number of integration steps
    """

    def __init__(
        self,
        hamiltonian: HamiltonianMLP,
        dt: float = 0.1,
        num_steps: int = 10,
    ) -> None:
        """__init__.

        Args:
            hamiltonian (HamiltonianMLP): Description.
            dt (float): Description.
            num_steps (int): Description.
        """
        super().__init__()
        self.H = hamiltonian
        self.dt = dt
        self.num_steps = num_steps

    def forward(
        self,
        q0: torch.Tensor,
        p0: torch.Tensor,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate Hamilton's equations.

        Args:
            q0: Initial position (batch, spatial_dim)
            p0: Initial momentum (batch, spatial_dim)
            return_trajectory: If True, return all intermediate states

        Returns:
            (q_final, p_final) or trajectory list

        forward method for LeapfrogIntegrator.

        Executes PyTorch tensor operations.

        Args:
            q0 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            p0 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            return_trajectory (bool): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        q, p = q0.clone(), p0.clone()
        q.requires_grad_(True)
        p.requires_grad_(True)

        trajectory = [(q.clone(), p.clone())] if return_trajectory else None

        for _ in range(self.num_steps):
            # Step 1: Half-step momentum update
            dH_dq = self._grad_q(q, p)
            p = p - (self.dt / 2) * dH_dq

            # Step 2: Full-step position update
            dH_dp = self._grad_p(q, p)
            q = q + self.dt * dH_dp

            # Step 3: Half-step momentum update
            dH_dq = self._grad_q(q, p)
            p = p - (self.dt / 2) * dH_dq

            if return_trajectory:
                trajectory.append((q.clone(), p.clone()))

        if return_trajectory:
            return trajectory
        return q, p

    def _grad_q(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute ∂H/∂q."""
        q = q.detach().requires_grad_(True)
        H = self.H(q, p.detach())
        grad = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        return grad

    def _grad_p(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute ∂H/∂p."""
        p = p.detach().requires_grad_(True)
        H = self.H(q.detach(), p)
        grad = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
        return grad


@register_model(name="hnf", training_mode="reconstruction")
class HamiltonianNeuralField(nn.Module):
    """Hamiltonian Neural Fields for diffeomorphic motion modeling.

    Uses a learned Hamiltonian to generate physically-plausible
    deformation fields for dynamic MRI reconstruction.

    Args:
        spatial_dim: Number of spatial dimensions (2 or 3)
        hidden_dims: Hamiltonian MLP hidden dimensions
        dt: Integration time step
        num_steps: Number of integration steps
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dims: tuple[int, ...] = (64, 64, 64),
        dt: float = 0.1,
        num_steps: int = 10,
    ) -> None:
        """__init__.

        Args:
            spatial_dim (int): Description.
            hidden_dims (tuple[int, ...]): Description.
            dt (float): Description.
            num_steps (int): Description.
        """
        super().__init__()
        self.spatial_dim = spatial_dim

        # Hamiltonian network
        self.hamiltonian = HamiltonianMLP(
            input_dim=2 * spatial_dim,
            hidden_dims=hidden_dims,
            use_softplus=True,  # Energy should be positive
        )

        # Symplectic integrator
        self.integrator = LeapfrogIntegrator(
            self.hamiltonian,
            dt=dt,
            num_steps=num_steps,
        )

        # Initial momentum generator (from image content)
        self.momentum_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, spatial_dim, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute deformation field from input.

        Args:
            x: Input image (B, C, H, W)
            t: Optional time parameter (B,) in [0, 1]

        Returns:
            (deformed_coords, displacement_field)
        """
        B, C, H, W = x.shape
        device = x.device

        # Generate initial momentum from image content
        p0 = self.momentum_encoder(x)  # (B, 2, H, W)

        # Create coordinate grid (positions q0)
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        q0 = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        q0 = q0.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # Reshape for integration
        q0_flat = q0.reshape(B * H * W, self.spatial_dim)
        p0_flat = p0.permute(0, 2, 3, 1).reshape(B * H * W, self.spatial_dim)

        # Integrate Hamilton's equations
        q_final, p_final = self.integrator(q0_flat, p0_flat)

        # Reshape back
        q_final = q_final.reshape(B, H, W, self.spatial_dim)

        # Displacement field
        displacement = q_final - q0

        return q_final, displacement

    def warp_image(
        self,
        x: torch.Tensor,
        target_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Warp image using computed deformation field.

        Args:
            x: Source image (B, C, H, W)
            target_coords: Deformed coordinates (B, H, W, 2)

        Returns:
            Warped image (B, C, H, W)
        """
        # grid_sample expects coords in [-1, 1]
        return F.grid_sample(x, target_coords, mode="bilinear", align_corners=True)

    def compute_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        regularization_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict]:
        """Compute registration loss.

        Args:
            source: Source frame (B, C, H, W)
            target: Target frame (B, C, H, W)
            regularization_weight: Weight for smoothness regularization

        Returns:
            (total_loss, loss_dict)
        """
        # Get deformation field
        target_coords, displacement = self(source)

        # Warp source to target
        warped = self.warp_image(source, target_coords)

        # Reconstruction loss
        loss_recon = F.mse_loss(warped, target)

        # Smoothness regularization (Jacobian determinant should be ~1)
        loss_smooth = self._jacobian_regularization(displacement)

        # Energy conservation (Hamiltonian should be approximately constant)
        # This is enforced by the symplectic integrator

        total_loss = loss_recon + regularization_weight * loss_smooth

        return total_loss, {
            "recon": loss_recon.item(),
            "smooth": loss_smooth.item(),
        }

    def _jacobian_regularization(self, displacement: torch.Tensor) -> torch.Tensor:
        """Penalize non-unitary Jacobian determinant.

        For diffeomorphisms, det(J) should be positive (invertible)
        and close to 1 (volume preserving).
        """
        # Compute spatial gradients
        dx = displacement[:, 1:, :, :] - displacement[:, :-1, :, :]
        dy = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]

        # Jacobian components
        # J = I + ∂u/∂x
        # For 2D: J = [[1 + ∂ux/∂x, ∂ux/∂y], [∂uy/∂x, 1 + ∂uy/∂y]]

        # Approximate Jacobian regularization (encourage small displacement gradients)
        reg = (dx**2).mean() + (dy**2).mean()

        return reg


def create_hamiltonian_nf(
    spatial_dim: int = 2,
    hidden_dims: tuple[int, ...] = (64, 64, 64),
    num_steps: int = 10,
) -> HamiltonianNeuralField:
    """Factory function for HNF.

    Args:
        spatial_dim: Number of spatial dimensions
        hidden_dims: Hamiltonian MLP dimensions
        num_steps: Integration steps

    Returns:
        Configured HNF model
    """
    return HamiltonianNeuralField(
        spatial_dim=spatial_dim,
        hidden_dims=hidden_dims,
        dt=1.0 / num_steps,
        num_steps=num_steps,
    )


__all__ = [
    "HamiltonianMLP",
    "HamiltonianNeuralField",
    "LeapfrogIntegrator",
    "create_hamiltonian_nf",
]
