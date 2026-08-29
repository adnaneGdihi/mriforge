"""Model-Based Image Reconstruction (MBIR) Solver.

Implements iterative inversion of the MRI forward operator using
Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
"""

import math

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.forward_operator import MultiCoilForwardOperator


class FISTAMBIRSolver(nn.Module):
    """FISTA solver for L1-regularized Model-Based Image Reconstruction.

    Solves: min_x 0.5 * ||A(x) - y||_2^2 + lambda * ||Psi(x)||_1

    Where A is the MultiCoilForwardOperator (with B0/S-maps), y is measured data.

    Note:
        The 'tv' regularizer uses a surrogate proximal step (see ``_apply_tv``),
        not the exact TV proximal map, so FISTA's convergence guarantees hold
        rigorously only for the identity / l1-soft-threshold regularizers.
    """

    def __init__(
        self,
        forward_operator: MultiCoilForwardOperator,
        num_iterations: int = 50,
        lambda_reg: float = 0.01,
        step_size: float = 0.5,
        regularization: str = "tv",
    ):
        """__init__.

        Args:
            forward_operator (MultiCoilForwardOperator): Description.
            num_iterations (int): Description.
            lambda_reg (float): Description.
            step_size (float): Description.
            regularization (str): Description.
        """
        super().__init__()
        self.A = forward_operator
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.alpha = step_size
        self.regularization = regularization

    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Apply soft thresholding to a complex tensor."""
        mag = torch.abs(x)
        # max(0, |x| - t) * (x / |x|)
        scale = torch.clamp(mag - threshold, min=0.0) / (mag + 1e-8)
        return x * scale

    def _apply_tv(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """One-step TV gradient surrogate (NOT the exact TV proximal operator).

        Computes ``x - D^T(soft_threshold(D x))`` where D is the circular forward-
        difference (D^T is exact, including the wrap term). This is a single
        Landweber/Jacobi-style descent step on the TV-smoothing objective, not
        ``prox_{lambda*TV}(x)``: in the small-threshold regime it reduces to
        ``(I - D^T D) x``, and ``I - D^T D`` is expansive (Laplacian eigenvalues up
        to 4), so it can overshoot/oscillate and does not satisfy the proximal-map
        contract FISTA's convergence theory assumes. For a rigorous TV prox, replace
        with Chambolle dual projected-gradient iterations (tracked as a follow-up).
        """
        # Calculate gradients (differences)
        dh = x - torch.roll(x, shifts=1, dims=-2)
        dw = x - torch.roll(x, shifts=1, dims=-1)

        # Soft threshold gradients
        dh_thresh = self._soft_threshold(dh, threshold)
        dw_thresh = self._soft_threshold(dw, threshold)

        # Integrate back (adjoint of difference)
        div = (dh_thresh - torch.roll(dh_thresh, shifts=-1, dims=-2)) + (
            dw_thresh - torch.roll(dw_thresh, shifts=-1, dims=-1)
        )

        return x - div

    def forward(self, y: torch.Tensor, x_init: torch.Tensor | None = None) -> torch.Tensor:
        """Run FISTA iterations to reconstruct image.

        Args:
            y: Acquired k-space measurements [B, C, H, W]
            x_init: Optional initial guess [B, 1, H, W] (e.g. adjoint recon)

        Returns:
            Reconstructed image [B, 1, H, W]

        forward method for FISTAMBIRSolver.

        Executes PyTorch tensor operations.

        Args:
            y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_init (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x_init is None:
            # Start with simple adjoint reconstruction
            x = self.A.adjoint(y)
        else:
            x = x_init.clone()

        # FISTA's O(1/k^2) guarantee needs the gradient step alpha <= 1/L with
        # L = ||A*A||_2 (the Lipschitz constant of the data-fidelity gradient). A
        # fixed alpha diverges when ||A*A|| > 1/alpha — easy to hit for multi-coil
        # A where ||S|| > 1. Estimate L once via power iteration (loop-invariant)
        # and clamp the effective step size; the configured ``step_size`` is then
        # an upper bound, not an unguarded constant.
        with torch.no_grad():
            v = torch.randn_like(x)
            v = v / (v.norm() + 1e-12)
            lam = v.norm().real  # real-valued Lipschitz estimate
            for _ in range(8):
                av = self.A.adjoint(self.A.forward(v))
                lam = av.norm()  # ||A*A v|| / ||v|| with ||v|| = 1
                v = av / (lam + 1e-12)
        # ``lam`` (a vector norm) is real; keep alpha real so the complex
        # gradient step ``alpha * gradient`` is well-defined.
        alpha = torch.minimum(lam.new_tensor(self.alpha), 1.0 / lam.clamp_min(1e-8))

        z = x.clone()  # Nesterov momentum state
        t = 1.0  # Momentum scaling

        for k in range(self.num_iterations):
            # 1. Evaluate gradient at z
            residual = self.A.forward(z) - y
            gradient = self.A.adjoint(residual)

            # 2. Gradient Descent Step (Lipschitz-safe step size)
            x_next_unproxed = z - alpha * gradient

            # 3. Proximal Step (Regularization)
            if self.regularization == "tv":
                x_next = self._apply_tv(x_next_unproxed, self.lambda_reg * alpha)
            else:
                x_next = self._soft_threshold(x_next_unproxed, self.lambda_reg * alpha)

            # 4. Nesterov Momentum Update
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_next
            z = x_next + momentum * (x_next - x)

            # Update state
            x = x_next
            t = t_next

        return x


def positive_contrast_demodulation(
    kspace: torch.Tensor, delta_omega_hz: float, te_sec: float | torch.Tensor
) -> torch.Tensor:
    """Pre-process K-space to shift target frequency to DC for positive contrast.

    Standard susceptibility markers cause signal voids due to violent phase dispersion
    (Δω). By pre-demodulating the k-space, we artificially refocus the marker spins
    so they reconstruct brightly (Positive Contrast), while intentionally blurring out
    the background anatomy.

    Note:
        Positive-contrast refocusing requires a *per-readout-sample* ``te_sec`` tensor
        whose length equals the readout (last) axis of ``kspace`` — only a time-varying
        demodulation refocuses the marker. A scalar ``te_sec`` applies a single uniform
        phase to all of k-space, which is mathematically a global image phase: it
        preserves magnitude and produces NO contrast shift. Pass a scalar only when a
        global phase is intended.

    Args:
        kspace: Acquired k-space [B, C, H, W] or [B, C, K]
        delta_omega_hz: Expected off-resonance frequency of the marker
        te_sec: Echo time in seconds. Scalar float (uniform global phase) or 1-D tensor
            whose length matches the k-space readout (last) axis (time-segmented).

    Returns:
        Demodulated k-space matching input shape

    Raises:
        ValueError: if ``te_sec`` is a tensor whose shape does not match the k-space
            readout (last) axis (silent mis-broadcast guard).
    """
    if isinstance(te_sec, torch.Tensor):
        te = te_sec.to(kspace.device).squeeze()
        if te.ndim > 1:
            raise ValueError(
                f"te_sec tensor must be scalar or 1-D, got shape {tuple(te_sec.shape)}"
            )
        if te.ndim == 1 and te.shape[0] != kspace.shape[-1]:
            raise ValueError(
                "te_sec length must match the k-space readout (last) axis "
                f"({kspace.shape[-1]}), got {te.shape[0]}. A 1-D te_sec is "
                "demodulated along the trailing axis only."
            )
        phase_shift_rad = 2.0 * math.pi * delta_omega_hz * te
    else:
        phase_shift_rad = torch.tensor(
            2.0 * math.pi * delta_omega_hz * te_sec, device=kspace.device
        )

    demod_factor = torch.exp(1j * phase_shift_rad)
    return kspace * demod_factor
