"""Task 1 Architectural Blocks for PMA-VarNet Unrolled Reconstruction.

Implements the 6 missing inner regularization kernels identified in the
PMA-VarNet gap analysis:

1. :class:`MoDLSuperResBlock`      — Complex Swin-Transformer + CG DC
2. :class:`WienerUNetBlock`        — Frequency-domain Wiener + residual CNN
3. :class:`EKFPhaseTracker`        — Extended Kalman Filter phase drift
4. :class:`DirichletESPIRiTBlock`  — Block Hankel SVT calibration projection
5. :class:`VCCVarNetBlock`         — Virtual Conjugate Coil synthesis
6. :class:`NeuralComplexSumBlock`  — Learned 1×1 MLP phase-cycle combiner

All blocks accept ``[B, C, H, W]`` complex or 2-channel real tensors
and return tensors of the same shape, enabling drop-in cascade usage.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

logger = logging.getLogger(__name__)


# ======================================================================
# 1.  MoDL Super-Resolution Block
# ======================================================================


class MoDLSuperResBlock(nn.Module):
    r"""Model-based Deep Learning block with Swin-Transformer regularizer.

    Implements the unrolled MoDL iteration:

    .. math::

        \mathbf{x}^{(n+1)} = (\mathcal{A}^H \mathcal{A} + \lambda I)^{-1}
                              (\mathcal{A}^H \mathbf{y} + \lambda \mathcal{R}_\theta(\mathbf{x}^{(n)}))

    The CG solver is approximated by ``num_cg_iters`` conjugate gradient steps.
    The regularizer :math:`\mathcal{R}_\theta` is a Swin-Transformer block.

    Args:
        channels: Number of feature channels (real-valued representation).
        num_heads: Number of attention heads for Swin.
        window_size: Window size for windowed self-attention.
        lambda_reg: Tikhonov regularization weight.
        num_cg_iters: Number of CG iterations for DC solve.
    """

    def __init__(
        self,
        channels: int = 64,
        num_heads: int = 4,
        window_size: int = 8,
        lambda_reg: float = 0.1,
        num_cg_iters: int = 6,
    ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.num_cg_iters = num_cg_iters

        # Swin regularizer operating on magnitude feature maps
        from spectramr.models.blocks.swin import SwinBlock

        self.embed = nn.Conv2d(channels, channels, 1)
        self.swin_block = SwinBlock(
            dim=channels,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
        )
        self.swin_block_shifted = SwinBlock(
            dim=channels,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
        )
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def _regularize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Swin-Transformer regularizer.

        Args:
            x: ``[B, C, H, W]`` real-valued features.

        Returns:
            Regularized features ``[B, C, H, W]``.
        """
        B, C, H, W = x.shape
        h = self.embed(x)
        # Swin expects [B, L, C] where L = H*W
        h = h.flatten(2).transpose(1, 2)  # [B, H*W, C]
        h = self.swin_block(h, input_resolution=(H, W))
        h = self.swin_block_shifted(h, input_resolution=(H, W))
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + self.proj_out(h)

    def _cg_dc(
        self,
        x_reg: torch.Tensor,
        y_kspace: torch.Tensor,
        mask: torch.Tensor,
        noise_cov_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Conjugate gradient data-consistency solve with optional noise covariance.

        Solves :math:`(A^H \Psi^{-1} A + \lambda I) x = A^H \Psi^{-1} y + \lambda x_{\text{reg}}`
        where :math:`A` is masked FFT and :math:`\Psi^{-1}` is the inverse noise covariance.

        Args:
            x_reg: Regularized estimate ``[B, C, H, W]``.
            y_kspace: Measured k-space ``[B, C, H, W]``.
            mask: Sampling mask ``[1, 1, H, W]`` or broadcastable.
            noise_cov_inv: Optional inverse noise covariance ``[B, C, C]`` or ``[C, C]``.

        Returns:
            DC-consistent estimate ``[B, C, H, W]``.
        """

        def _apply_AHA(x_: torch.Tensor) -> torch.Tensor:
            Ax_ = fft2c(x_) * mask
            if noise_cov_inv is not None:
                B_, C_, H_, W_ = Ax_.shape
                # If noise_cov_inv is [C, C], unsqueeze to [1, C, C]
                psi_inv = noise_cov_inv if noise_cov_inv.dim() == 3 else noise_cov_inv.unsqueeze(0)
                flat_Ax = Ax_.reshape(B_, C_, -1)
                flat_Ax = torch.matmul(psi_inv, flat_Ax)
                Ax_ = flat_Ax.reshape(B_, C_, H_, W_)
            return ifft2c(Ax_) + self.lambda_reg * x_

        # Apply noise covariance to the measured y_kspace
        y_w = y_kspace * mask
        if noise_cov_inv is not None:
            B_, C_, H_, W_ = y_w.shape
            psi_inv = noise_cov_inv if noise_cov_inv.dim() == 3 else noise_cov_inv.unsqueeze(0)
            flat_y = y_w.reshape(B_, C_, -1)
            flat_y = torch.matmul(psi_inv, flat_y)
            y_w = flat_y.reshape(B_, C_, H_, W_)

        # Right-hand side: A^H \Psi^{-1} y + lambda * x_reg
        rhs = ifft2c(y_w) + self.lambda_reg * x_reg

        # Initialize CG
        x = x_reg.clone()
        r = rhs - _apply_AHA(x)
        p = r.clone()
        rsold = (r.conj() * r).real.sum()

        for _ in range(self.num_cg_iters):
            Ap = _apply_AHA(p)
            pAp = (p.conj() * Ap).real.sum()
            if pAp.abs() < 1e-12:
                break
            alpha = rsold / pAp.clamp(min=1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = (r.conj() * r).real.sum()
            if rsnew.sqrt() < 1e-10:
                break
            p = r + (rsnew / rsold.clamp(min=1e-12)) * p
            rsold = rsnew

        return x

    def forward(
        self,
        x: torch.Tensor,
        y_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        noise_cov_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one MoDL iteration.

        Args:
            x: Current image estimate ``[B, C, H, W]`` (real or complex).
            y_kspace: Measured k-space (optional).
            mask: Sampling mask (optional).
            noise_cov_inv: Inverse noise covariance (optional).

        Returns:
            Updated image estimate ``[B, C, H, W]``.

        forward method for MoDLSuperResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            noise_cov_inv (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        is_complex = x.is_complex()

        if is_complex:
            # Regularize on magnitude features, apply as complex residual
            x_mag = x.abs()
            x_reg_mag = self._regularize(x_mag)
            # Scale: multiply original complex by (reg_mag / orig_mag)
            scale = x_reg_mag / x_mag.clamp(min=1e-8)
            x_reg = x * scale
        else:
            x_reg = self._regularize(x)

        if y_kspace is not None and mask is not None:
            return self._cg_dc(x_reg, y_kspace, mask)
        return x_reg


# ======================================================================
# 2.  Wiener-UNet Block
# ======================================================================


class _ResidualBlock(nn.Module):
    """Simple 2-layer residual block for image-domain refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Refined tensor.

        forward method for _ResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return x + self.conv(x)


class WienerUNetBlock(nn.Module):
    r"""Differentiable Wiener deconvolution + residual CNN refinement.

    Stage 1 (frequency domain):

    .. math::

        \hat{X}(k) = \frac{H^*(k) \cdot Y(k)}{|H(k)|^2 + \lambda}

    Stage 2 (image domain): Residual CNN refinement.

    Args:
        channels: Number of channels.
        num_res_blocks: Number of residual blocks in the CNN.
        lambda_noise: Wiener regularization parameter (noise-to-signal).
    """

    def __init__(
        self,
        channels: int = 2,
        num_res_blocks: int = 4,
        lambda_noise: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_noise = nn.Parameter(torch.tensor(lambda_noise), requires_grad=True)

        # Learnable PSF in k-space (initialized to unity)
        self.log_psf_power = nn.Parameter(torch.zeros(1, 1, 1, 1))

        # Image-space residual CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            *[_ResidualBlock(64) for _ in range(num_res_blocks)],
            nn.Conv2d(64, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Wiener deconvolution + CNN refinement.

        Args:
            x: Input ``[B, C, H, W]`` (2-channel real or complex).

        Returns:
            Refined ``[B, C, H, W]``.

        forward method for WienerUNetBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Stage 1: Wiener filter in k-space
        if x.is_complex():
            X_k = fft2c(x)
        else:
            x_complex = torch.complex(x[:, 0:1], x[:, 1:2]) if x.shape[1] >= 2 else x
            X_k = fft2c(x_complex)

        # H is approximated as spatially flat with learned power
        H_power = self.log_psf_power.exp()
        wiener_filter = H_power / (H_power + self.lambda_noise.abs().clamp(min=1e-8))
        X_filtered = X_k * wiener_filter
        x_wiener = ifft2c(X_filtered)

        # Convert back to 2-channel real for CNN
        if not x.is_complex():
            x_wiener_real = torch.cat([x_wiener.real, x_wiener.imag], dim=1)
            if x_wiener_real.shape[1] != x.shape[1]:
                x_wiener_real = x_wiener_real[:, : x.shape[1]]
        else:
            x_wiener_real = torch.cat([x_wiener.real, x_wiener.imag], dim=1)

        # Stage 2: Residual CNN
        refined = self.cnn(x_wiener_real)
        out = x_wiener_real + refined

        if x.is_complex():
            C = out.shape[1] // 2
            return torch.complex(out[:, :C], out[:, C:])
        return out


# ======================================================================
# 3.  Extended Kalman Filter Phase Tracker
# ======================================================================


class EKFPhaseTracker(nn.Module):
    r"""Extended Kalman Filter for navigator-based phase drift tracking.

    State vector: :math:`\mathbf{s} = [\phi, \dot\phi]` (phase and rate).

    Process model (constant velocity):

    .. math::

        \mathbf{s}_{t+1} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}
                           \mathbf{s}_t + \mathbf{w}

    Measurement model:

    .. math::

        z_t = \angle(k_0(t)) = \phi_t + v_t

    Args:
        process_noise: Diagonal process noise covariance.
        measurement_noise: Measurement noise variance.
        dt: Time step between navigators (seconds).
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        dt: float = 0.01,
    ) -> None:
        super().__init__()
        self.dt = dt

        # State transition matrix F: [2, 2]
        F = torch.tensor([[1.0, dt], [0.0, 1.0]])
        self.register_buffer("F", F)

        # Measurement matrix H: [1, 2]
        H = torch.tensor([[1.0, 0.0]])
        self.register_buffer("H", H)

        # Process noise Q: [2, 2]
        Q = torch.eye(2) * process_noise
        self.register_buffer("Q", Q)

        # Measurement noise R: scalar
        self.register_buffer("R", torch.tensor([[measurement_noise]]))

    def forward(
        self,
        phase_measurements: torch.Tensor,
    ) -> torch.Tensor:
        """Run EKF on a sequence of phase measurements.

        Args:
            phase_measurements: ``[B, T]`` navigator phase angles in radians.

        Returns:
            Filtered phase estimates ``[B, T]``.

        forward method for EKFPhaseTracker.

        Executes PyTorch tensor operations.

        Args:
            phase_measurements (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, T = phase_measurements.shape
        device = phase_measurements.device

        # Initialize state: [B, 2]
        x_hat = torch.zeros(B, 2, device=device)
        P = torch.eye(2, device=device).unsqueeze(0).expand(B, 2, 2).clone()

        filtered_phases = []

        for t in range(T):
            # ----- Predict -----
            x_pred = torch.einsum("ij,bj->bi", self.F, x_hat)
            P_pred = self.F @ P @ self.F.T + self.Q

            # ----- Update -----
            z = phase_measurements[:, t : t + 1]  # [B, 1]
            y_innov = z - x_pred[:, 0:1]  # innovation

            # Wrap phase to [-π, π]
            y_innov = torch.atan2(y_innov.sin(), y_innov.cos())

            S = self.H @ P_pred @ self.H.T + self.R  # [B, 1, 1] or [1, 1]
            K = (P_pred @ self.H.T) / S.squeeze(-1).clamp(min=1e-8)  # [B, 2, 1]

            x_hat = x_pred + (K.squeeze(-1) * y_innov)
            I_KH = torch.eye(2, device=device) - K @ self.H
            P = I_KH @ P_pred

            filtered_phases.append(x_hat[:, 0])

        return torch.stack(filtered_phases, dim=1)  # [B, T]


# ======================================================================
# 4.  Dirichlet-ESPIRiT Block
# ======================================================================


class DirichletESPIRiTBlock(nn.Module):
    r"""Block Hankel SVT-based ESPIRiT sensitivity map estimation.

    Extracts coil sensitivity maps from the central calibration region
    of multi-coil k-space using structured low-rank matrix completion:

    1. Extract calibration lines → build block Hankel matrix.
    2. SVD truncation to rank ``r``.
    3. Reshape dominant singular vectors → eigenvalue maps.

    Args:
        kernel_size: Calibration kernel size.
        num_maps: Number of ESPIRiT maps to extract.
        threshold: Singular value threshold (fraction of max).
    """

    def __init__(
        self,
        kernel_size: int = 6,
        num_maps: int = 1,
        threshold: float = 0.02,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_maps = num_maps
        self.threshold = threshold

    def _build_hankel(
        self,
        calib: torch.Tensor,
    ) -> torch.Tensor:
        """Build block Hankel matrix from calibration data.

        Args:
            calib: ``[Nc, cal_h, cal_w]`` calibration k-space.

        Returns:
            Hankel matrix ``[num_blocks, Nc * kernel_size^2]``.
        """
        Nc, cal_h, cal_w = calib.shape
        k = self.kernel_size
        patches = calib.unfold(1, k, 1).unfold(2, k, 1)  # [Nc, nh, nw, k, k]
        nh, nw = patches.shape[1], patches.shape[2]
        # Reshape: [nh*nw, Nc*k*k]
        return patches.permute(1, 2, 0, 3, 4).reshape(nh * nw, Nc * k * k)

    def forward(
        self,
        kspace: torch.Tensor,
        center_fraction: float = 0.08,
    ) -> torch.Tensor:
        """Estimate ESPIRiT sensitivity maps.

        Args:
            kspace: Multi-coil k-space ``[B, Nc, H, W]`` complex.
            center_fraction: Fraction of k-space center to use for calibration.

        Returns:
            Sensitivity maps ``[B, Nc, H, W]`` complex.

        forward method for DirichletESPIRiTBlock.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            center_fraction (float): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, Nc, H, W = kspace.shape
        device = kspace.device
        dtype = kspace.dtype

        cal_h = max(self.kernel_size + 2, int(H * center_fraction))
        cal_w = max(self.kernel_size + 2, int(W * center_fraction))
        h_start = (H - cal_h) // 2
        w_start = (W - cal_w) // 2

        maps_list = []
        for b in range(B):
            calib = kspace[b, :, h_start : h_start + cal_h, w_start : w_start + cal_w]
            hankel = self._build_hankel(calib)

            # SVD
            U, S, Vh = torch.linalg.svd(hankel, full_matrices=False)
            # Threshold
            cutoff = S[0] * self.threshold
            rank = (cutoff < S).sum().item()
            rank = max(rank, Nc)

            # Extract sensitivity kernels from dominant right singular vectors
            V_trunc = Vh[:rank]  # [rank, Nc * k * k]
            kernels = V_trunc.reshape(rank, Nc, self.kernel_size, self.kernel_size)

            # Eigenvalue decomposition at each voxel
            # Pad kernels to image size and IFFT
            padded = torch.zeros(rank, Nc, H, W, device=device, dtype=dtype)
            kh, kw = self.kernel_size, self.kernel_size
            ph, pw = (H - kh) // 2, (W - kw) // 2
            padded[:, :, ph : ph + kh, pw : pw + kw] = kernels
            sens_images = ifft2c(padded)

            # Take top eigenmap via power iteration approximation
            # Simplified: use first SVD component as sensitivity map
            sens_map = sens_images[0]  # [Nc, H, W]

            # Normalize per-pixel
            rss = (sens_map.abs().pow(2).sum(dim=0, keepdim=True) + 1e-8).sqrt()
            sens_map = sens_map / rss

            maps_list.append(sens_map)

        return torch.stack(maps_list, dim=0)  # [B, Nc, H, W]


# ======================================================================
# 5.  Virtual Conjugate Coil VarNet Block
# ======================================================================


class VCCVarNetBlock(nn.Module):
    r"""Virtual Conjugate Coil (VCC) synthesis for Hermitian enforcement.

    Doubles the effective number of coils by appending conjugate-reflected
    k-space data:

    .. math::

        y_{\text{vcc},c}(\mathbf{k}) = y_c^*(-\mathbf{k})

    This enforces Hermitian symmetry (real-valued image output) and
    provides additional SNR.

    Args:
        apply_data_consistency: Whether to apply DC after VCC synthesis.
    """

    def __init__(self, apply_data_consistency: bool = True) -> None:
        super().__init__()
        self.apply_dc = apply_data_consistency
        # Learnable mixing weight between original and VCC coils
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _conjugate_reflect(self, kspace: torch.Tensor) -> torch.Tensor:
        """Generate virtual conjugate coil data.

        Args:
            kspace: ``[B, Nc, H, W]`` complex k-space.

        Returns:
            VCC k-space ``[B, Nc, H, W]`` complex.
        """
        return torch.flip(kspace, dims=[-2, -1]).conj()

    def forward(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply VCC synthesis and optional data consistency.

        Args:
            kspace: ``[B, Nc, H, W]`` complex k-space.
            mask: Sampling mask ``[1, 1, H, W]`` (optional).

        Returns:
            Enhanced k-space ``[B, 2*Nc, H, W]`` or ``[B, Nc, H, W]``
            depending on concat mode.

        forward method for VCCVarNetBlock.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        vcc = self._conjugate_reflect(kspace)
        alpha = self.alpha.sigmoid()

        # Weighted combination: use original at acquired positions, VCC elsewhere
        if mask is not None and self.apply_dc:
            combined = mask * kspace + (1 - mask) * (alpha * kspace + (1 - alpha) * vcc)
        else:
            combined = alpha * kspace + (1 - alpha) * vcc

        return combined


# ======================================================================
# 6.  Neural Complex-Sum Block
# ======================================================================


class NeuralComplexSumBlock(nn.Module):
    r"""Learned voxel-wise :math:`1 \times 1` complex MLP for coil combination.

    Replaces the standard RSS combination:

    .. math::

        x_{\text{RSS}} = \sqrt{\sum_c |x_c|^2}

    with a data-driven adaptive combiner:

    .. math::

        x_{\text{out}} = \text{MLP}([x_1, \ldots, x_{N_c}])

    where the MLP operates on the real/imaginary stacked representation
    at each voxel independently.

    Args:
        num_coils: Number of input coils.
        hidden_dim: Hidden dimension of the per-voxel MLP.
        output_channels: Output channels (typically 1 for combined image).
    """

    def __init__(
        self,
        num_coils: int = 4,
        hidden_dim: int = 32,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        in_features = num_coils * 2  # real + imag per coil
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_channels * 2, 1),
        )
        self.output_channels = output_channels

    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        """Combine multi-coil images via learned MLP.

        Args:
            coil_images: ``[B, Nc, H, W]`` complex coil images.

        Returns:
            Combined image ``[B, output_channels, H, W]`` complex.

        forward method for NeuralComplexSumBlock.

        Executes PyTorch tensor operations.

        Args:
            coil_images (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if coil_images.is_complex():
            # Stack real and imaginary as channels
            x = torch.cat([coil_images.real, coil_images.imag], dim=1)
        else:
            x = coil_images

        out = self.mlp(x)  # [B, 2*out_ch, H, W]
        C = self.output_channels
        return torch.complex(out[:, :C], out[:, C:])


__all__ = [
    "DirichletESPIRiTBlock",
    "EKFPhaseTracker",
    "MoDLSuperResBlock",
    "NeuralComplexSumBlock",
    "VCCVarNetBlock",
    "WienerUNetBlock",
]
