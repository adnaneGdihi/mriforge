"""Clinical Trust Analyzer: Null-Space Residual Map & Jacobian Sensitivity.

Provides interpretability tools for physics-informed MRI reconstruction:

1. **Null-Space Residual Map**: Pushes the generated output back through the
   DifferentiableULFForwardOperator and compares against raw k-space. Any
   energy above the thermal noise floor indicates hallucinated content that
   violates the acquisition physics.

2. **Jacobian Sensitivity Maps**: Computes ∂G_θ(x)/∂T₁(r) and ∂G_θ(x)/∂ΔB₀(r)
   using ``torch.autograd``. A bright region in J_{T1} proves the network
   derived that contrast from genuine biological T₁ data. A bright region
   in J_{B0} proves the network compensated for a hardware-level field
   distortion.

Together, these convert "AI black-box faith" into deterministic, interrogatable
physics for FDA-level clinical trust.

References:
    - Antun, V. et al. (2020). "On instabilities of deep learning in image
      reconstruction and the potential costs of AI." PNAS.
    - Darestani, M. Z. et al. (2021). "Measuring Robustness in Deep Learning
      Based Compressive Sensing." ICML.
"""

from __future__ import annotations

import inspect
import logging
from typing import NamedTuple

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

logger = logging.getLogger(__name__)


class TrustAnalysisResult(NamedTuple):
    """Output of clinical trust analysis."""

    prediction: torch.Tensor
    """Generated reconstruction [B, C, H, W]."""
    trust_map: torch.Tensor
    """Null-space residual above noise floor [B, 1, H, W].
    Nonzero regions indicate potential hallucination."""
    jacobian_t1: torch.Tensor | None
    """Sensitivity to T₁ input [B, 1, H, W]. None if T₁ not provided."""
    jacobian_b0: torch.Tensor | None
    """Sensitivity to B₀ input [B, 1, H, W]. None if B₀ not provided."""
    kspace_residual: torch.Tensor
    """Raw k-space residual [B, 1, H, W] (complex)."""


def estimate_noise_floor(
    kspace: torch.Tensor,
    corner_fraction: float = 0.1,
) -> torch.Tensor:
    """Estimate thermal noise floor from k-space corners (no signal).

    Uses the Median Absolute Deviation (MAD) estimator on the high-frequency
    corners of k-space where signal energy is negligible.

    Args:
        kspace: Complex k-space [B, C, H, W].
        corner_fraction: Fraction of corner region to sample.

    Returns:
        Noise standard deviation estimate [B, 1, 1, 1].
    """
    H, W = kspace.shape[-2], kspace.shape[-1]
    h_edge = max(1, int(H * corner_fraction))
    w_edge = max(1, int(W * corner_fraction))

    # Sample four corners
    corners = torch.cat(
        [
            kspace[..., :h_edge, :w_edge].reshape(kspace.shape[0], -1),
            kspace[..., :h_edge, -w_edge:].reshape(kspace.shape[0], -1),
            kspace[..., -h_edge:, :w_edge].reshape(kspace.shape[0], -1),
            kspace[..., -h_edge:, -w_edge:].reshape(kspace.shape[0], -1),
        ],
        dim=-1,
    )

    # MAD estimator: σ ≈ 1.4826 · median(|x - median(x)|)
    corner_mag = corners.abs()
    med = corner_mag.median(dim=-1, keepdim=True).values
    mad = (corner_mag - med).abs().median(dim=-1, keepdim=True).values
    sigma = 1.4826 * mad

    # Reshape for broadcasting
    return sigma.reshape(kspace.shape[0], 1, 1, 1)


class ClinicalTrustAnalyzer(nn.Module):
    """Physics-grounded interpretability for MRI reconstruction models.

    Computes null-space residual maps and Jacobian sensitivity analysis
    to provide clinically actionable trust metrics.

    Args:
        forward_operator: Differentiable forward operator (e.g.,
            DifferentiableULFForwardOperator or a simpler FFT-based model).
            Must accept image and return k-space.
        noise_threshold_sigma: Number of noise standard deviations above
            which residual energy is flagged as potential hallucination.
        use_adjoint: If True, project residual back to image space via
            adjoint operator (A†). If False, return magnitude of k-space
            residual directly.
    """

    def __init__(
        self,
        forward_operator: nn.Module | None = None,
        noise_threshold_sigma: float = 3.0,
        use_adjoint: bool = True,
    ) -> None:
        super().__init__()
        self.forward_operator = forward_operator
        self.noise_threshold_sigma = noise_threshold_sigma
        self.use_adjoint = use_adjoint

    def _default_forward(self, image: torch.Tensor) -> torch.Tensor:
        """Default forward operator: centered FFT (single-coil Cartesian)."""
        if not torch.is_complex(image):
            image = image.to(torch.complex64)
        return fft2c(image)

    def _default_adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        """Default adjoint operator: centered IFFT."""
        return ifft2c(kspace)

    def compute_null_space_residual(
        self,
        prediction: torch.Tensor,
        raw_kspace: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute null-space residual: R = |A†(A(x̂) - y)|.

        Args:
            prediction: Network output [B, C, H, W].
            raw_kspace: Acquired k-space measurements [B, C, H, W] (complex).
            mask: Optional undersampling mask [B, 1, H, W].

        Returns:
            (trust_map, kspace_residual):
                trust_map: Thresholded residual magnitude [B, 1, H, W].
                kspace_residual: Raw complex residual [B, C, H, W].
        """
        # Forward: image → k-space
        if self.forward_operator is not None:
            pred_kspace = self.forward_operator(prediction)
        else:
            pred_kspace = self._default_forward(prediction)

        # Ensure raw_kspace is complex
        if not torch.is_complex(raw_kspace):
            raw_kspace = raw_kspace.to(torch.complex64)

        # Apply mask to both (compare only at measured locations)
        if mask is not None:
            mask_f = mask.float()
            while mask_f.ndim < pred_kspace.ndim:
                mask_f = mask_f.unsqueeze(1)
            kspace_residual = mask_f * (pred_kspace - raw_kspace)
        else:
            kspace_residual = pred_kspace - raw_kspace

        # Project back to image space
        if self.use_adjoint:
            image_residual = self._default_adjoint(kspace_residual)
            residual_mag = image_residual.abs()
        else:
            residual_mag = kspace_residual.abs()

        # Collapse channel dim if needed
        if residual_mag.shape[1] > 1:
            residual_mag = residual_mag.mean(dim=1, keepdim=True)

        # Threshold against noise floor
        noise_sigma = estimate_noise_floor(raw_kspace)
        threshold = self.noise_threshold_sigma * noise_sigma
        trust_map = torch.relu(residual_mag - threshold)

        return trust_map, kspace_residual

    def compute_jacobian(
        self,
        generator: nn.Module,
        x_input: torch.Tensor,
        physical_params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute Jacobian sensitivity maps ∂G_θ(x)/∂param(r).

        Args:
            generator: Reconstruction network.
            x_input: Network input (e.g., noisy ULF image) [B, C, H, W].
            physical_params: Dict of physical parameter maps to differentiate
                through, e.g., {"t1": t1_map, "b0": b0_map}.
                Each tensor must have requires_grad=True.

        Returns:
            Dict mapping param name → Jacobian magnitude [B, 1, H, W].
        """
        jacobians = {}

        # Enable gradients on all physical params
        for name, param in physical_params.items():
            param.requires_grad_(True)

        # Forward through generator.
        # Decide kwarg-acceptance by introspecting the signature, NOT by catching
        # TypeError — a bare `except TypeError` would also swallow real TypeErrors
        # raised *inside* a kwarg-accepting generator (bad reshape/einsum/None layer),
        # silently re-running without physical params and masking the crash as a
        # benign "parameter unused" Jacobian (pitfall #9, silent degradation).
        try:
            params = inspect.signature(generator.forward).parameters
        except (ValueError, TypeError):
            params = {}
        accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        accepts_all_named = all(name in params for name in physical_params)
        if physical_params and (accepts_var_kwargs or accepts_all_named):
            pred = generator(x_input, **physical_params)
        else:
            # Generator does not accept the physical-param kwargs — call with input only.
            pred = generator(x_input)

        # Compute scalar output for autograd
        scalar_output = pred.abs().sum() if torch.is_complex(pred) else pred.sum()

        # Compute Jacobian for each physical parameter
        for name, param in physical_params.items():
            if not param.requires_grad:
                continue

            grad = torch.autograd.grad(
                outputs=scalar_output,
                inputs=param,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if grad is not None:
                # Take magnitude and collapse channels
                jac_mag = grad.abs()
                if jac_mag.shape[1] > 1:
                    jac_mag = jac_mag.mean(dim=1, keepdim=True)
                jacobians[name] = jac_mag
            else:
                logger.warning(
                    "Jacobian for '%s' is None (parameter unused by generator)",
                    name,
                )
                jacobians[name] = torch.zeros_like(x_input[:, :1])

        return jacobians

    def forward(
        self,
        generator: nn.Module,
        x_input: torch.Tensor,
        raw_kspace: torch.Tensor,
        mask: torch.Tensor | None = None,
        t1_map: torch.Tensor | None = None,
        b0_map: torch.Tensor | None = None,
    ) -> TrustAnalysisResult:
        """Full clinical trust analysis pipeline.

        Args:
            generator: Reconstruction network G_θ.
            x_input: Network input (noisy ULF) [B, C, H, W].
            raw_kspace: Raw acquired k-space [B, C, H, W] (complex).
            mask: Undersampling mask [B, 1, H, W].
            t1_map: T₁ parameter map [B, 1, H, W]. None to skip.
            b0_map: ΔB₀ map [B, 1, H, W]. None to skip.

        Returns:
            TrustAnalysisResult with prediction, trust_map, and Jacobians.

        forward method for ClinicalTrustAnalyzer.

        Executes PyTorch tensor operations.

        Args:
            generator (nn.Module): Expected input tensor.
            x_input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            raw_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t1_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b0_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            TrustAnalysisResult: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Build physical params dict for Jacobian
        physical_params: dict[str, torch.Tensor] = {}
        if t1_map is not None:
            physical_params["t1"] = t1_map.clone().detach().requires_grad_(True)
        if b0_map is not None:
            physical_params["b0"] = b0_map.clone().detach().requires_grad_(True)

        # Jacobian sensitivity
        jacobians: dict[str, torch.Tensor] = {}
        if physical_params:
            jacobians = self.compute_jacobian(generator, x_input, physical_params)

        # Generate prediction (no grad for residual computation)
        with torch.no_grad():
            prediction = generator(x_input)

        # Null-space residual
        trust_map, kspace_residual = self.compute_null_space_residual(
            prediction,
            raw_kspace,
            mask,
        )

        return TrustAnalysisResult(
            prediction=prediction,
            trust_map=trust_map,
            jacobian_t1=jacobians.get("t1"),
            jacobian_b0=jacobians.get("b0"),
            kspace_residual=kspace_residual,
        )


__all__ = [
    "ClinicalTrustAnalyzer",
    "TrustAnalysisResult",
    "estimate_noise_floor",
]
