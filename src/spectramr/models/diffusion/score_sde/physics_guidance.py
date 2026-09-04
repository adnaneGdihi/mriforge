"""Physics Guidance for Score-Based Diffusion.

Implements guidance gradients using differentiable Bloch simulation and
measurement consistency for Inverse Problems.
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spectramr.infrastructure.physics.bloch_simulation import DifferentiableBlochSimulator
from spectramr.infrastructure.physics.interfaces import IPhysicsOperator
from spectramr.models.diffusion.mrf_diph.bloch_dictionary import BlochDictionary


class MeasurementConsistencyGuidance:
    """Guidance for enforcing y = Ax + n during diffusion sampling.

    Implements the gradient of the likelihood score: ∇_x log p(y | x_t)
    Derived from Song et al. ICLR 2022.
    """

    def __init__(self, operator: IPhysicsOperator, measurement: torch.Tensor, noise_std: float):
        """
        Args:
            operator: Linear Forward Operator A (e.g., MRI Fourier Transform).
            measurement: Observed data y.
            noise_std: Standard deviation of measurement noise (sigma_n).
        """
        self.A = operator
        self.y = measurement
        self.sigma_n = noise_std

    def get_guidance_gradient(self, x_t: torch.Tensor, time_t: float) -> torch.Tensor:
        """
        Calculates ∇_x log p(y | x_t) using a Gaussian approximation.
        grad = A^H * (y - A(x_t)) / (sigma_n^2 + sigma_t^2)

        Args:
            x_t: Current image estimate at diffusion time t.
            time_t: Current diffusion time scalar (sigma_t).

        Returns:
            Gradient tensor of same shape as x_t.
        """
        # 1. Forward projection of current estimate
        # A(x_t)
        Ax_t = self.A.forward(x_t)

        # 2. Residual calculation
        # y - A(x_t)
        residual = self.y - Ax_t

        # 3. Adjoint projection (Backprojection)
        # A^H * (y - A(x_t))
        grad_numerator = self.A.adjoint(residual)

        # 4. Variance scaling (Measurement noise + Diffusion noise)
        # Assuming scalar sigma_t for simplicity, in SDE this varies
        variance = self.sigma_n**2 + time_t**2

        return grad_numerator / variance

    def apply_data_consistency(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Alternative: Hard projection (POCS) for low-noise regimes.
        x' = x_t + A^H(y - A x_t)

        Note: The formula x_t - A^H(A x_t - y) is equivalent to x_t + A^H(y - A x_t)
        """
        # A(x_t)
        Ax_t = self.A.forward(x_t)

        # y - A(x_t)
        residual = self.y - Ax_t

        # Backproject residual
        correction = self.A.adjoint(residual)

        return x_t + correction


class BlochGuidance(nn.Module):
    """Computes physics-guided gradients for VLF-to-HF translation.

    Guidance score: ∇_x log p(y_vlf | x_hf)
    Approx: -λ * ∇_x || y_vlf - Bloch(Maps(x_hf), P_vlf) ||^2
    """

    def __init__(
        self,
        dictionary: BlochDictionary,
        tr_hf: float = 1000.0,
        te_hf: float = 100.0,
        sequence_type: str = "SE",
    ):
        """__init__.

        Args:
            dictionary (BlochDictionary): Description.
            tr_hf (float): Description.
            te_hf (float): Description.
            sequence_type (str): Description.
        """
        super().__init__()
        self.dictionary = dictionary
        self.tr_hf = tr_hf
        self.te_hf = te_hf
        self.simulator = DifferentiableBlochSimulator(sequence_type=sequence_type)

    def forward(
        self,
        x: Float[Tensor, "B C D H W"],
        y_obs: Float[Tensor, "B C D H W"],
        tr_vlf: float,
        te_vlf: float,
        scale: float = 1.0,
    ) -> Float[Tensor, "B C D H W"]:
        """Compute guidance gradient.

        Args:
            x: Input HF image estimate (x_0_hat).
            y_obs: Observed VLF image.
            tr_vlf: TR of VLF sequence.
            te_vlf: TE of VLF sequence.
            scale: Guidance scale (lambda).

        Returns:
            Gradient tensor of same shape as x.

        forward method for BlochGuidance.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y_obs (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            tr_vlf (float): Expected input tensor.
            te_vlf (float): Expected input tensor.
            scale (float): Expected input tensor.

        Returns:
            Float[Tensor, 'B C D H W']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # We need gradients w.r.t x
        x_in = x.detach().requires_grad_(True)

        # 1. Map HF image to tissue properties (Soft matching for gradients)
        # Dictionary expects [B, C, D, H, W]
        # returns indices [B, 1, D, H, W] but here we need maps
        # match_soft returns weighted sum of dictionary entries: [B, 3, D, H, W]
        # We assume x_in has same contrast properties as dictionary atoms

        # Note: BlochDictionary.match_soft computes similarity between x_in and atoms
        # Atoms are (PD, T1, T2) signals generated for the HF protocol.
        # So we are effectively doing: HF Image -> Tissue Maps
        tissue_maps = self.dictionary.match_soft(
            x_in, TR=self.tr_hf, TE=self.te_hf, temperature=0.1
        )

        # 2. Simulate VLF signal from tissue maps
        # tissue_maps: [B, 3, D, H, W]
        # simulator expects [B, 3, H, W] usually but let's check dims
        # The simulator code uses tissue_maps[:, 0:1] so it handles channel dim 1.
        # It blindly operates on tensors, so D dimension should be preserved if passed correctly.
        y_sim = self.simulator(tissue_maps, TR=tr_vlf, TE=te_vlf)

        # 3. Compute consistency loss
        loss = torch.sum((y_sim - y_obs) ** 2)

        # 4. Compute gradient
        grad = torch.autograd.grad(loss, x_in)[0]

        return -scale * grad
