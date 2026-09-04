"""Dual-Marker Correction Operators for SNR & Resolution Enhancement.

Physics-based correction modules that leverage a dual-marker system:
- **Synthetic digital cross anchor** with known geometry, contrast, and position
- **Synthetic on-patient tracking markers** for motion/physiology compensation

Each operator is a differentiable ``nn.Module`` usable in the iterative
reconstruction chain from :mod:`vf_operators`.

Operators
---------
- :class:`GNLUnwarpingOperator`       — gradient non-linearity correction
- :class:`BiasFieldCorrector`         — B1 intensity normalization
- :class:`AnalyticalPSFSolver`        — Fourier-division PSF + Wiener deconv
- :class:`EPINyquistGhostCorrector`   — odd/even phase error correction
- :class:`SENSEGFactorCalibrator`     — g-factor from marker coil calibration
- :class:`RespiratoryBinningAverager` — motion-state binning + √N averaging
- :class:`SubVoxelSuperResolution`    — multi-frame sub-pixel SR
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Gradient Non-Linearity (GNL) Unwarping
# ======================================================================


class GNLUnwarpingOperator(nn.Module):
    r"""Correct gradient non-linearity distortion with Jacobian modulation.

    Gradient fields deviate from linearity far from isocenter.  Given the
    known marker positions, the deformation field is estimated and its
    Jacobian determinant modulates signal intensity to conserve signal
    density:

    .. math::

        I_{\text{corrected}}(\mathbf{r}) =
            I_{\text{distorted}}(\mathbf{r} + \mathbf{d}(\mathbf{r}))
            \cdot |\det J(\mathbf{r})|

    where :math:`J` is the Jacobian of the displacement field.

    Args:
        im_size: Spatial dimensions ``(H, W)``.
        poly_order: Order of the polynomial distortion model.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        poly_order: int = 3,
    ) -> None:
        super().__init__()
        self.im_size = im_size
        self.poly_order = poly_order
        H, W = im_size

        # Normalized coordinate grids [-1, 1]
        gy = torch.linspace(-1, 1, H)
        gx = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        # grid_sample expects (x, y) order in last dim
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        self.register_buffer("base_grid", base_grid)

    def estimate_deformation(
        self,
        true_positions: torch.Tensor,
        distorted_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate dense displacement field from sparse marker correspondences.

        Uses thin-plate spline interpolation (approximated via polynomial fit)
        from marker control point pairs.

        Args:
            true_positions: Known physical positions ``[N, 2]`` (y, x).
            distorted_positions: Measured distorted positions ``[N, 2]``.

        Returns:
            Dense displacement field ``[1, 2, H, W]`` in normalised coords.
        """
        H, W = self.im_size
        device = true_positions.device

        # Displacement at control points
        delta = distorted_positions - true_positions  # [N, 2]

        # Fit 2D polynomial to sparse displacements
        # Build Vandermonde-like basis: [1, y, x, y², yx, x², ...]
        ty = true_positions[:, 0]  # [N]
        tx = true_positions[:, 1]  # [N]

        basis_cols = []
        for p in range(self.poly_order + 1):
            for q in range(self.poly_order + 1 - p):
                basis_cols.append(ty**p * tx**q)
        A = torch.stack(basis_cols, dim=-1)  # [N, M]

        # Solve for coefficients via least squares: A @ coeffs = delta
        coeffs, *_ = torch.linalg.lstsq(A, delta)  # [M, 2]

        # Evaluate on dense grid
        gy = torch.linspace(-1, 1, H, device=device)
        gx = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        dense_basis = []
        for p in range(self.poly_order + 1):
            for q in range(self.poly_order + 1 - p):
                dense_basis.append((grid_y**p * grid_x**q).reshape(-1))
        A_dense = torch.stack(dense_basis, dim=-1)  # [H*W, M]

        flow = (A_dense @ coeffs).reshape(H, W, 2)  # [H, W, 2] (dy, dx)
        # Convert to [1, 2, H, W] format (dx, dy order)
        return flow.permute(2, 0, 1).unsqueeze(0).flip(1)  # flip dy,dx → dx,dy

    def compute_jacobian_det(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian determinant of the displacement field.

        Args:
            flow: Displacement ``[1, 2, H, W]`` (dx, dy).

        Returns:
            Jacobian determinant ``[1, 1, H, W]``.
        """
        # Spatial gradients via finite differences
        # flow[0, 0] = dx, flow[0, 1] = dy
        dx_dx = torch.gradient(flow[0, 0], dim=1)[0]  # ∂dx/∂x
        dx_dy = torch.gradient(flow[0, 0], dim=0)[0]  # ∂dx/∂y
        dy_dx = torch.gradient(flow[0, 1], dim=1)[0]  # ∂dy/∂x
        dy_dy = torch.gradient(flow[0, 1], dim=0)[0]  # ∂dy/∂y

        # J = I + ∇d → det(J) = (1 + ∂dx/∂x)(1 + ∂dy/∂y) - (∂dx/∂y)(∂dy/∂x)
        det = (1.0 + dx_dx) * (1.0 + dy_dy) - dx_dy * dy_dx
        return det.abs().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def forward(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Apply GNL unwarping with Jacobian intensity modulation.

        Args:
            image: Distorted image ``[B, C, H, W]``.
            flow: Displacement field ``[1, 2, H, W]`` (normalised coords).

        Returns:
            Corrected image ``[B, C, H, W]``.

        forward method for GNLUnwarpingOperator.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            flow (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = image.shape[0]

        # Build sampling grid: base + flow
        flow_perm = flow.permute(0, 2, 3, 1)  # [1, H, W, 2]
        grid = self.base_grid + flow_perm
        grid = grid.expand(B, -1, -1, -1)

        # Warp
        unwarped = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Jacobian modulation — conserve signal density
        jac_det = self.compute_jacobian_det(flow)  # [1, 1, H, W]
        return unwarped * jac_det


# ======================================================================
# 2. Bias Field Corrector
# ======================================================================


class BiasFieldCorrector(nn.Module):
    r"""Correct B1 receive intensity bias using marker-derived field map.

    Divides the image by a smooth bias field estimated from marker
    signal deviations:

    .. math::

        I_{\text{corrected}} = \frac{I_{\text{raw}}}{B_1^-(\mathbf{r}) + \epsilon}

    The bias field is interpolated from sparse marker measurements to
    produce a smooth, full-FOV correction.

    Args:
        smooth_sigma: Gaussian smoothing sigma for bias field interpolation.
    """

    def __init__(self, smooth_sigma: float = 10.0) -> None:
        super().__init__()
        self.smooth_sigma = smooth_sigma

    @staticmethod
    def _gaussian_smooth(field: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian smoothing to a 2D field.

        Args:
            field: Input ``[B, 1, H, W]``.
            sigma: Gaussian kernel sigma in pixels.

        Returns:
            Smoothed field ``[B, 1, H, W]``.
        """
        k = int(6 * sigma) | 1  # Ensure odd
        x = torch.arange(k, device=field.device, dtype=field.dtype) - k // 2
        g1d = torch.exp(-0.5 * (x / sigma) ** 2)
        g1d = g1d / g1d.sum()
        kernel = g1d.unsqueeze(0) * g1d.unsqueeze(1)  # [k, k]
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
        pad = k // 2
        return F.conv2d(field, kernel, padding=pad)

    def forward(
        self,
        image: torch.Tensor,
        bias_field: torch.Tensor,
    ) -> torch.Tensor:
        """Apply bias field correction.

        Args:
            image: Raw image ``[B, C, H, W]``.
            bias_field: Sparse bias field ``[B, 1, H, W]`` (0 = unknown).

        Returns:
            Corrected image ``[B, C, H, W]``.

        forward method for BiasFieldCorrector.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            bias_field (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Smooth the sparse bias field to get full-FOV estimate
        nonzero = (bias_field.abs() > 1e-6).float()
        filled = self._gaussian_smooth(bias_field * nonzero, self.smooth_sigma)
        weight = self._gaussian_smooth(nonzero, self.smooth_sigma).clamp(min=1e-6)
        smooth_bias = filled / weight

        # Normalise bias to have unit mean
        mean_b = smooth_bias.mean().clamp(min=1e-6)
        smooth_bias = smooth_bias / mean_b

        return image / smooth_bias.clamp(min=1e-4)


# ======================================================================
# 3. Analytical PSF Solver
# ======================================================================


class AnalyticalPSFSolver(nn.Module):
    r"""Estimate PSF via Fourier division using the known cross prior.

    Because the cross marker has a mathematically known sharp geometry
    :math:`O_{\text{cross}}`, the PSF is extracted by Fourier deconvolution:

    .. math::

        \text{PSF} = \mathcal{F}^{-1}\left(
            \frac{\mathcal{F}(I_{\text{marker}})}
                 {\mathcal{F}(O_{\text{cross}}) + \epsilon}
        \right)

    Wiener regularisation prevents noise amplification:

    .. math::

        \text{PSF} = \mathcal{F}^{-1}\left(
            \frac{\mathcal{F}(I) \cdot \mathcal{F}(O)^*}
                 {|\mathcal{F}(O)|^2 + \lambda}
        \right)

    Args:
        cross_size: Size of the cross arm in pixels.
        cross_width: Width of the cross arm in pixels.
        wiener_lambda: Wiener regularisation parameter.
    """

    def __init__(
        self,
        cross_size: int = 20,
        cross_width: int = 3,
        wiener_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.cross_size = cross_size
        self.cross_width = cross_width
        self.wiener_lambda = wiener_lambda

    def _build_cross_prior(
        self,
        H: int,
        W: int,
        center: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build ideal binary cross image at specified center.

        Args:
            H: Image height.
            W: Image width.
            center: (cy, cx) center of cross.
            device: Device.

        Returns:
            Ideal cross ``[1, 1, H, W]``.
        """
        cross = torch.zeros(1, 1, H, W, device=device)
        cy, cx = center
        hw = self.cross_width // 2
        hs = self.cross_size // 2

        # Vertical arm
        y_start = max(0, cy - hs)
        y_end = min(H, cy + hs + 1)
        x_start = max(0, cx - hw)
        x_end = min(W, cx + hw + 1)
        cross[:, :, y_start:y_end, x_start:x_end] = 1.0

        # Horizontal arm
        y_start = max(0, cy - hw)
        y_end = min(H, cy + hw + 1)
        x_start = max(0, cx - hs)
        x_end = min(W, cx + hs + 1)
        cross[:, :, y_start:y_end, x_start:x_end] = 1.0

        return cross

    def estimate_psf(
        self,
        marker_image: torch.Tensor,
        cross_center: tuple[int, int],
    ) -> torch.Tensor:
        """Estimate PSF from blurred marker vs. ideal cross prior.

        Args:
            marker_image: Acquired marker region ``[1, 1, H, W]``.
            cross_center: (cy, cx) position of the cross.

        Returns:
            Estimated PSF ``[1, 1, H, W]``.
        """
        _, _, H, W = marker_image.shape
        device = marker_image.device

        # Build ideal cross
        cross = self._build_cross_prior(H, W, cross_center, device)

        # Fourier division with Wiener regularisation
        F_image = torch.fft.fft2(marker_image)
        F_cross = torch.fft.fft2(cross)

        # Wiener: PSF = F⁻¹(F(I)·F(O)* / (|F(O)|² + λ))
        numerator = F_image * F_cross.conj()
        denominator = F_cross.abs().pow(2) + self.wiener_lambda
        F_psf = numerator / denominator

        psf = torch.fft.ifft2(F_psf).real
        # Normalise to unit sum
        psf = psf / (psf.sum() + 1e-8)

        return psf

    def deconvolve(
        self,
        image: torch.Tensor,
        psf: torch.Tensor,
        wiener_lambda: float | None = None,
    ) -> torch.Tensor:
        """Apply Wiener deconvolution to an image using estimated PSF.

        Args:
            image: Blurred image ``[B, C, H, W]``.
            psf: Estimated PSF ``[1, 1, H, W]``.
            wiener_lambda: Override regularisation parameter.

        Returns:
            Deblurred image ``[B, C, H, W]``.
        """
        lam = wiener_lambda if wiener_lambda is not None else self.wiener_lambda
        B, C, H, W = image.shape

        F_psf = torch.fft.fft2(psf, s=(H, W))
        result_channels = []
        for c in range(C):
            F_img = torch.fft.fft2(image[:, c : c + 1])
            F_deconv = F_img * F_psf.conj() / (F_psf.abs().pow(2) + lam)
            result_channels.append(torch.fft.ifft2(F_deconv).real)

        return torch.cat(result_channels, dim=1)


# ======================================================================
# 4. EPI Nyquist Ghost Corrector
# ======================================================================


class EPINyquistGhostCorrector(nn.Module):
    r"""Correct N/2 Nyquist ghosting in EPI using marker phase calibration.

    EPI alternating readout polarity causes odd/even k-space line phase
    mismatch.  The fixed marker's known position enables exact
    measurement of the 0th and 1st order phase errors:

    .. math::

        \phi(k_x) = \phi_0 + \phi_1 \cdot k_x
        \quad \text{(applied to alternate, odd-indexed PE lines)}

    Args:
        im_size: Spatial dimensions ``(H, W)``.
    """

    def __init__(self, im_size: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.im_size = im_size

    def estimate_phase_errors(
        self,
        kspace: torch.Tensor,
        marker_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate 0th and 1st order phase errors from marker ghost.

        The ghost of the marker appears shifted by FOV/2 in the PE direction.
        By measuring the phase difference between the primary marker signal
        and its ghost, we extract the systematic phase errors.

        Args:
            kspace: Complex k-space ``[B, 1, H, W]``.
            marker_mask: Marker location mask ``[1, 1, H, W]``.

        Returns:
            Tuple of (phi_0, phi_1) each ``[B]``.
        """
        H, W = self.im_size
        image = torch.fft.ifft2(kspace, dim=(-2, -1))

        # Primary marker signal
        primary = image * marker_mask

        # Ghost is at FOV/2 shift in PE (vertical) direction
        ghost_mask = torch.roll(marker_mask, H // 2, dims=-2)
        # Align the ghost back onto the marker support so primary and ghost are
        # compared at the same spatial locations (column-by-column in readout).
        ghost = torch.roll(image * ghost_mask, -(H // 2), dims=-2)

        # Phase difference between ghost and primary
        p_sum = (primary * marker_mask).sum(dim=(-2, -1))
        g_sum = (ghost * marker_mask).sum(dim=(-2, -1))

        # phi_0 = angle(ghost / primary)
        ratio = g_sum / (p_sum + 1e-10)
        phi_0 = torch.angle(ratio).squeeze()  # [B] or scalar

        # phi_1: the readout-linear phase slope (the DOMINANT Nyquist-ghost term;
        # previously hard-coded to zero, so only the constant offset was ever
        # corrected). Collapse the marker support to a 1-D readout profile for
        # primary and ghost, then estimate the slope of angle(ghost/primary) in
        # k_x via the Ahn-Cho estimator: the phase of the sum of adjacent-sample
        # products is the mean per-sample phase increment (wrap-robust). The
        # forward model uses k_x in [-0.5, 0.5] (step 1/(W-1)), so the slope is
        # the per-sample increment times (W-1).
        prim_ro = primary.sum(dim=-2)  # [B, 1, W]
        ghost_ro = ghost.sum(dim=-2)  # [B, 1, W]
        ratio_ro = ghost_ro / (prim_ro + 1e-10)
        incr = (ratio_ro[..., 1:] * ratio_ro[..., :-1].conj()).sum(dim=-1)
        phi_1 = (torch.angle(incr) * (W - 1)).reshape(phi_0.shape)

        return phi_0, phi_1

    def forward(
        self,
        kspace: torch.Tensor,
        phi_0: torch.Tensor,
        phi_1: torch.Tensor,
    ) -> torch.Tensor:
        """Apply phase correction to remove N/2 ghost.

        Args:
            kspace: Complex k-space ``[B, 1, H, W]``.
            phi_0: 0th order phase error ``[B]``.
            phi_1: 1st order phase error ``[B]``.

        Returns:
            Ghost-corrected k-space ``[B, 1, H, W]``.

        forward method for EPINyquistGhostCorrector.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            phi_0 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            phi_1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = kspace.shape
        kx = torch.linspace(-0.5, 0.5, W, device=kspace.device)

        # Phase correction for the odd-indexed PE lines (EPI Nyquist ghost
        # arises from the alternating readout polarity between even/odd echoes;
        # the correction is applied to the ``1::2`` set below).
        correction = torch.exp(
            -1j * (phi_0.view(B, 1, 1, 1) + phi_1.view(B, 1, 1, 1) * kx.view(1, 1, 1, W))
        )

        # Apply to odd-indexed PE lines only (matches the class docstring; the
        # previous "even" comments contradicted the ``1::2`` slice).
        corrected = kspace.clone()
        corrected[:, :, 1::2, :] = kspace[:, :, 1::2, :] * correction

        return corrected


# ======================================================================
# 5. SENSE G-Factor Calibrator
# ======================================================================


class SENSEGFactorCalibrator(nn.Module):
    r"""Calibrate SENSE g-factor map from marker coil sensitivity profiles.

    The marker's known, noise-free signal provides perfect coil sensitivity
    calibration.  The g-factor (geometry factor) quantifies noise
    amplification in parallel imaging:

    .. math::

        g(\mathbf{r}) = \sqrt{
            (\mathbf{S}^H \Psi^{-1} \mathbf{S})^{-1}_{jj}
            \cdot (\mathbf{S}^H \Psi^{-1} \mathbf{S})_{jj}
        }

    where :math:`\mathbf{S}` is the sensitivity matrix and :math:`\Psi`
    is the noise covariance.

    Args:
        num_coils: Number of receive coils.
        acceleration: Acceleration factor R.
    """

    def __init__(
        self,
        num_coils: int = 8,
        acceleration: int = 4,
    ) -> None:
        super().__init__()
        self.num_coils = num_coils
        self.acceleration = acceleration

    def compute_g_factor(
        self,
        sensitivity_maps: torch.Tensor,
        noise_cov: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute SENSE g-factor map.

        Args:
            sensitivity_maps: Coil sensitivities ``[num_coils, H, W]`` complex.
            noise_cov: Noise covariance ``[num_coils, num_coils]`` complex.
                        If None, assumes identity (uncorrelated noise).

        Returns:
            G-factor map ``[1, 1, H, W]``.
        """
        C, H, W = sensitivity_maps.shape
        R = self.acceleration
        device = sensitivity_maps.device

        if noise_cov is None:
            psi_inv = torch.eye(C, device=device, dtype=sensitivity_maps.dtype)
        else:
            psi_inv = torch.linalg.inv(noise_cov)

        g_map = torch.ones(H, W, device=device)

        if H % R == 0:
            # Vectorized over all aliased groups and columns at once. Aliased
            # pixel ``(g, r)`` is row ``r*G + g`` with ``G = H // R``; reshaping
            # H -> (R, G) groups them. This replaces an O((H/R)*W) Python loop of
            # tiny [R, R] inversions (~16k for 256^2, R=4) with one batched solve.
            G = H // R
            smap_r = sensitivity_maps.reshape(C, R, G, W)
            S = smap_r.permute(2, 3, 0, 1)  # [G, W, C, R]
            E = S.conj().transpose(-2, -1) @ psi_inv @ S  # [G, W, R, R]
            E_inv = torch.linalg.pinv(E)  # robust on singular groups
            e_diag = E.diagonal(dim1=-2, dim2=-1)  # [G, W, R]
            einv_diag = E_inv.diagonal(dim1=-2, dim2=-1)  # [G, W, R]
            g_val = torch.sqrt((einv_diag * e_diag).real.abs()).clamp(min=1.0)
            # g_val[g, w, r] -> g_map[r*G + g, w]
            g_map = g_val.permute(2, 0, 1).reshape(H, W)
            return g_map.unsqueeze(0).unsqueeze(0)

        # Fallback for non-divisible H (incomplete aliased groups left at g=1).
        for offset in range(H // R):
            indices = [offset + r * (H // R) for r in range(R) if offset + r * (H // R) < H]
            if len(indices) < 2:
                continue
            for w in range(W):
                S = torch.stack([sensitivity_maps[:, idx, w] for idx in indices], dim=1)  # [C, R]
                E = S.conj().T @ psi_inv @ S  # [R, R]
                try:
                    E_inv = torch.linalg.inv(E)
                except torch.linalg.LinAlgError:
                    continue
                for r_idx, pixel_idx in enumerate(indices):
                    g_map[pixel_idx, w] = torch.sqrt(
                        (E_inv[r_idx, r_idx] * E[r_idx, r_idx]).real.abs()
                    ).clamp(min=1.0)

        return g_map.unsqueeze(0).unsqueeze(0)


# ======================================================================
# 6. Respiratory Binning & Coherent Averaging
# ======================================================================


class RespiratoryBinningAverager(nn.Module):
    r"""Bin k-space data by respiratory phase and coherently average.

    The on-patient marker trajectory provides a continuous respiratory
    waveform.  All acquired data is binned into discrete breathing
    phases, and non-rigid registration aligns all bins to a reference
    state before averaging:

    .. math::

        I_{\text{final}} = \frac{1}{N} \sum_{n=1}^{N}
            \mathcal{W}_n^{-1}(I_n)
        \quad \Rightarrow \quad
        \text{SNR}_{\text{final}} = \sqrt{N} \cdot \text{SNR}_1

    Args:
        num_bins: Number of respiratory phase bins.
    """

    def __init__(self, num_bins: int = 4) -> None:
        super().__init__()
        self.num_bins = num_bins

    def bin_by_phase(
        self,
        kspace_lines: torch.Tensor,
        respiratory_trace: torch.Tensor,
    ) -> list[list[int]]:
        """Assign k-space lines to respiratory bins.

        Args:
            kspace_lines: ``[N_lines, W]`` complex k-space.
            respiratory_trace: ``[N_lines]`` breathing amplitude [0, 1].

        Returns:
            List of N_bins lists, each containing line indices.
        """
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=respiratory_trace.device)
        bins: list[list[int]] = [[] for _ in range(self.num_bins)]

        for i, amp in enumerate(respiratory_trace):
            bin_idx = min(
                int((amp * self.num_bins).clamp(max=self.num_bins - 1).item()),
                self.num_bins - 1,
            )
            bins[bin_idx].append(i)

        return bins

    def forward(
        self,
        frames: torch.Tensor,
        reference_idx: int = 0,
    ) -> torch.Tensor:
        """Average multiple respiratory-binned frames.

        In a simplified model (without explicit non-rigid registration),
        directly averages the frames.  For full pipeline, compose with
        an external elastic registration module.

        Args:
            frames: ``[N_bins, C, H, W]`` images from each respiratory bin.
            reference_idx: Which bin is the reference state.

        Returns:
            Averaged image ``[1, C, H, W]`` with √N SNR boost.

        forward method for RespiratoryBinningAverager.

        Executes PyTorch tensor operations.

        Args:
            frames (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            reference_idx (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Simple coherent average (non-rigid registration would be external)
        return frames.mean(dim=0, keepdim=True)


# ======================================================================
# 7. Sub-Voxel Super-Resolution
# ======================================================================


class SubVoxelSuperResolution(nn.Module):
    r"""Multi-frame super-resolution from sub-pixel tracked shifts.

    Multiple low-resolution frames with known sub-pixel translations
    (from marker tracking) are combined to reconstruct a higher-
    resolution image.  Formulated as an inverse problem:

    .. math::

        \min_{\mathbf{x}} \sum_{n=1}^{N}
            \| D \cdot W_n \cdot \mathbf{x} - \mathbf{y}_n \|_2^2
            + \lambda \|\nabla \mathbf{x}\|_1

    where :math:`D` is downsampling, :math:`W_n` is the sub-pixel
    shift for frame n, and :math:`\nabla` is total variation.

    Args:
        upscale_factor: Resolution enhancement factor (e.g. 2 → 2× resolution).
        num_iters: Iterative solver iterations.
        tv_lambda: Total variation regularisation weight.
    """

    def __init__(
        self,
        upscale_factor: int = 2,
        num_iters: int = 20,
        tv_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.num_iters = num_iters
        self.tv_lambda = tv_lambda

    @staticmethod
    def _total_variation(x: torch.Tensor) -> torch.Tensor:
        """Compute isotropic total variation.

        Args:
            x: Image ``[B, C, H, W]``.

        Returns:
            Scalar TV value.
        """
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum()
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()
        return dy + dx

    def _shift_and_downsample(
        self,
        hr: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Forward model: shift HR image, then downsample.

        Args:
            hr: High-res estimate ``[1, C, H_hr, W_hr]``.
            shift: Sub-pixel shift ``[2]`` (dy, dx) in HR pixels.

        Returns:
            Simulated LR frame ``[1, C, H_lr, W_lr]``.
        """
        _, C, H, W = hr.shape
        s = self.upscale_factor

        # Build shifted sampling grid
        grid_y = torch.linspace(-1, 1, H, device=hr.device)
        grid_x = torch.linspace(-1, 1, W, device=hr.device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")

        # Apply sub-pixel shift (convert to normalised coordinates)
        gx = gx + shift[1] * 2.0 / W
        gy = gy + shift[0] * 2.0 / H
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

        shifted = F.grid_sample(
            hr, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        # Downsample via average pooling
        return F.avg_pool2d(shifted, kernel_size=s, stride=s)

    def forward(
        self,
        lr_frames: torch.Tensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct super-resolved image from multi-frame LR inputs.

        Args:
            lr_frames: ``[N, C, H_lr, W_lr]`` low-resolution frames.
            shifts: ``[N, 2]`` sub-pixel shifts (dy, dx) in LR pixels.

        Returns:
            Super-resolved image ``[1, C, H_hr, W_hr]``.

        forward method for SubVoxelSuperResolution.

        Executes PyTorch tensor operations.

        Args:
            lr_frames (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            shifts (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        N, C, H_lr, W_lr = lr_frames.shape
        s = self.upscale_factor
        H_hr, W_hr = H_lr * s, W_lr * s

        # Convert shifts from LR to HR pixel space
        shifts_hr = shifts * s

        # Initialize HR estimate via bilinear upsampling of first frame
        x = F.interpolate(
            lr_frames[0:1],
            size=(H_hr, W_hr),
            mode="bilinear",
            align_corners=True,
        )

        # Gradient descent. The data-fidelity and TV gradients are computed
        # MANUALLY below (no .backward() call), so autograd tracking is pure
        # overhead — leaving x with requires_grad built a ~num_iters-deep graph
        # that was retained across the loop only to be discarded by .detach().
        # Run the whole solve under no_grad.
        lr = 0.1
        with torch.no_grad():
            for _it in range(self.num_iters):
                # Data fidelity gradient
                grad = torch.zeros_like(x)
                for n in range(N):
                    simulated = self._shift_and_downsample(x, shifts_hr[n])
                    residual = simulated - lr_frames[n : n + 1]
                    # Backproject residual to HR space
                    grad = grad + F.interpolate(
                        residual,
                        size=(H_hr, W_hr),
                        mode="bilinear",
                        align_corners=True,
                    )

                grad = grad / N

                # TV regularisation gradient (approximate)
                tv_grad = torch.zeros_like(x)
                if self.tv_lambda > 0:
                    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
                    dx_img = x[:, :, :, 1:] - x[:, :, :, :-1]
                    tv_grad[:, :, :-1, :] += -dy.sign()
                    tv_grad[:, :, 1:, :] += dy.sign()
                    tv_grad[:, :, :, :-1] += -dx_img.sign()
                    tv_grad[:, :, :, 1:] += dx_img.sign()

                x = x - lr * (grad + self.tv_lambda * tv_grad)

        return x.detach()


# ======================================================================
# Alias: K-Space Phase Autofocus = RigidKinematicOperator
# ======================================================================

# Idea 7 from the proposal is already implemented as RigidKinematicOperator.
# Provide a semantic alias for discoverability.
from spectramr.infrastructure.physics.vf_operators import RigidKinematicOperator

KSpacePhaseAutofocus = RigidKinematicOperator


__all__ = [
    "AnalyticalPSFSolver",
    "BiasFieldCorrector",
    "EPINyquistGhostCorrector",
    "GNLUnwarpingOperator",
    "KSpacePhaseAutofocus",
    "RespiratoryBinningAverager",
    "SENSEGFactorCalibrator",
    "SubVoxelSuperResolution",
]
