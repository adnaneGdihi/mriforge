"""Quantitative Susceptibility Mapping (QSM) Pipeline.

Implements the complete QSM processing chain:
1. Field map estimation from multi-echo phase data
2. Laplacian-based spatial phase unwrapping (Schofield & Zhu 2003)
3. V-SHARP background field removal (Li et al. 2011)
4. Dipole inversion via TKD and iterative LSQR (Wharton et al. 2010)

Reference:
    Wang et al., "Quantitative Susceptibility Mapping (QSM): Decoding
    MRI Data for a Tissue Magnetic Biomarker," MRM (2015).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.dipole import get_dipole_kernel_3d

# ---------------------------------------------------------------------------
# 1. Field-map estimation
# ---------------------------------------------------------------------------


def compute_field_map(
    phase_te1: torch.Tensor,
    phase_te2: torch.Tensor,
    te1: float,
    te2: float,
) -> torch.Tensor:
    r"""Compute raw B0 field map from dual-echo phase difference.

    .. math::
        \Delta\omega(\mathbf{r}) = \frac{\phi(TE_2) - \phi(TE_1)}
                                        {TE_2 - TE_1}

    The output is the off-resonance **angular frequency** in radians/second —
    there is **no** :math:`\gamma` division (that would give Tesla; the code
    returns ``delta_phi / delta_te``). The caller converts to Hz by dividing
    by 2π.

    Args:
        phase_te1: Wrapped phase at TE₁ [B, 1, D, H, W] (radians)
        phase_te2: Wrapped phase at TE₂ [B, 1, D, H, W] (radians)
        te1: First echo time in seconds
        te2: Second echo time in seconds

    Returns:
        Field map in **rad/s**, same shape as inputs
    """
    if te2 <= te1:
        raise ValueError(f"te2 ({te2}) must be > te1 ({te1})")

    delta_te = te2 - te1
    # Phase difference (wrapped)
    delta_phi = torch.angle(torch.exp(1j * phase_te2) * torch.exp(-1j * phase_te1))
    # Convert from rad to rad/s
    return delta_phi / delta_te


# ---------------------------------------------------------------------------
# 2. Laplacian phase unwrapping
# ---------------------------------------------------------------------------


def laplacian_phase_unwrap(
    wrapped_phase: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Laplacian-based spatial phase unwrapping.

    Solves the Poisson equation :math:`\nabla^2 \phi_{unwrapped}
    = \nabla^2 \phi_{wrapped}` using FFT, exploiting the fact that
    the Laplacian of the true (unwrapped) phase is smooth.

    Reference:
        Schofield & Zhu, "Fast phase unwrapping algorithm for
        interferometric applications," Optics Letters (2003).

    Args:
        wrapped_phase: Wrapped phase [B, 1, D, H, W] (radians)
        mask: Optional binary brain mask [B, 1, D, H, W]

    Returns:
        Unwrapped phase, same shape
    """
    # Compute Laplacian of wrapped phase via cos/sin decomposition
    # ∇²φ = Im[ (∇²e^{iφ}) * e^{-iφ} ]  (Schofield & Zhu trick)
    exp_phi = torch.exp(1j * wrapped_phase.to(torch.complex64))

    # 3D Laplacian via finite differences (isotropic stencil)
    laplacian_exp = _laplacian_3d(exp_phi)
    laplacian_phi = (laplacian_exp * exp_phi.conj()).imag

    # Solve Poisson equation in k-space: φ = F⁻¹{ F{∇²φ} / L(k) }
    spatial_dims = (-3, -2, -1)
    laplacian_phi_k = torch.fft.fftn(laplacian_phi, dim=spatial_dims)

    # Build Laplacian eigenvalues in k-space
    D, H, W = wrapped_phase.shape[-3:]
    kz = torch.fft.fftfreq(D, device=wrapped_phase.device)
    ky = torch.fft.fftfreq(H, device=wrapped_phase.device)
    kx = torch.fft.fftfreq(W, device=wrapped_phase.device)
    kz_g, ky_g, kx_g = torch.meshgrid(kz, ky, kx, indexing="ij")

    # Eigenvalues of the 3D discrete Laplacian:
    # λ = 2(cos(2πkx) + cos(2πky) + cos(2πkz) - 3)
    eigenvalues = 2.0 * (
        torch.cos(2.0 * torch.pi * kx_g)
        + torch.cos(2.0 * torch.pi * ky_g)
        + torch.cos(2.0 * torch.pi * kz_g)
        - 3.0
    )
    # Expand to match batch dims
    while eigenvalues.dim() < wrapped_phase.dim():
        eigenvalues = eigenvalues.unsqueeze(0)

    # Regularize DC (eigenvalue = 0)
    eigenvalues[eigenvalues.abs() < 1e-6] = 1.0

    unwrapped = torch.fft.ifftn(laplacian_phi_k / eigenvalues, dim=spatial_dims).real

    if mask is not None:
        unwrapped = unwrapped * mask.float()

    return unwrapped


def _laplacian_3d(x: torch.Tensor) -> torch.Tensor:
    """3D discrete Laplacian via 7-point stencil (finite differences)."""
    lap = -6.0 * x
    lap = lap + torch.roll(x, 1, dims=-3) + torch.roll(x, -1, dims=-3)
    lap = lap + torch.roll(x, 1, dims=-2) + torch.roll(x, -1, dims=-2)
    lap = lap + torch.roll(x, 1, dims=-1) + torch.roll(x, -1, dims=-1)
    return lap


# ---------------------------------------------------------------------------
# 3. Background field removal (V-SHARP)
# ---------------------------------------------------------------------------


def vsharp_background_removal(
    field_map: torch.Tensor,
    mask: torch.Tensor,
    radii: list[int] | None = None,
) -> torch.Tensor:
    r"""Variable-radius Spherical Mean Value (V-SHARP) background removal.

    For each radius *r*, the spherical mean value (SMV) filter extracts
    the harmonic (background) component.  The local field is recovered
    by subtracting the SMV from the total field.

    Reference:
        Li et al., "Quantitative susceptibility mapping of human brain
        reflects spatial variation in tissue composition," NeuroImage (2011).

    Args:
        field_map: Total field map [B, 1, D, H, W] (rad/s)
        mask: Brain mask [B, 1, D, H, W]
        radii: List of SMV kernel radii in voxels (default: [12, 9, 6, 3])

    Returns:
        Local field map (background removed) [B, 1, D, H, W]
    """
    if radii is None:
        radii = [12, 9, 6, 3]

    device = field_map.device
    local_field = torch.zeros_like(field_map)
    remaining_mask = mask.float().clone()

    for radius in sorted(radii, reverse=True):
        # Build 3D spherical kernel
        kernel = _spherical_kernel(radius, device)

        # SMV convolution (normalized)
        smv_field = _smv_convolve(field_map * remaining_mask, kernel)
        smv_mask = _smv_convolve(remaining_mask, kernel)

        # Voxels where SMV is valid (fully inside the eroded mask)
        valid = smv_mask > 0.999
        valid_new = valid & (remaining_mask > 0.5) & (local_field.abs() < 1e-10)

        # Local field = total - background (SMV)
        local_at_r = field_map - smv_field / (smv_mask + 1e-8)
        local_field[valid_new] = local_at_r[valid_new]

        # Erode mask for next iteration
        remaining_mask[valid] = 0.0

    return local_field


def _spherical_kernel(radius: int, device: torch.device) -> torch.Tensor:
    """Create a normalized 3D spherical averaging kernel."""
    size = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=torch.float32) - radius
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    sphere = (x**2 + y**2 + z**2) <= radius**2
    kernel = sphere.float()
    kernel = kernel / kernel.sum()
    return kernel.reshape(1, 1, size, size, size)


def _smv_convolve(
    volume: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Apply 3D convolution with a spherical kernel (same-padding)."""
    pad = kernel.shape[-1] // 2
    padded = F.pad(volume, [pad] * 6, mode="replicate")
    return F.conv3d(padded, kernel, padding=0)


# ---------------------------------------------------------------------------
# 4. Dipole inversion — Truncated K-space Division (TKD)
# ---------------------------------------------------------------------------


def tkd_dipole_inversion(
    local_field: torch.Tensor,
    mask: torch.Tensor,
    voxel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float = 0.1,
) -> torch.Tensor:
    r"""Truncated K-space Division (TKD) for dipole inversion.

    Solves :math:`\chi = D^{-1}(k) \cdot F\{\Delta B_0^{local}\}` where
    the dipole kernel D(k) is thresholded to avoid the ill-conditioned
    cone at the magic angle (54.7°).

    Args:
        local_field: Background-removed field map [B, 1, D, H, W]
        mask: Brain mask [B, 1, D, H, W]
        voxel_sizes: (dz, dy, dx) in mm
        threshold: Truncation threshold for D(k)

    Returns:
        Susceptibility map χ [B, 1, D, H, W] in same units as local_field
    """
    spatial_dims = (-3, -2, -1)

    # Get dipole kernel
    D_k = get_dipole_kernel_3d(local_field.shape, voxel_sizes).to(local_field.device)

    # Truncated inverse
    D_inv = torch.zeros_like(D_k)
    valid = D_k.abs() >= threshold
    D_inv[valid] = 1.0 / D_k[valid]

    # Apply inversion in k-space
    field_k = torch.fft.fftn(local_field * mask.float(), dim=spatial_dims)
    chi_k = field_k * D_inv
    chi = torch.fft.ifftn(chi_k, dim=spatial_dims).real

    return chi * mask.float()


# ---------------------------------------------------------------------------
# 5. Dipole inversion — Iterative LSQR with Tikhonov regularization
# ---------------------------------------------------------------------------


def ilsqr_dipole_inversion(
    local_field: torch.Tensor,
    mask: torch.Tensor,
    voxel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lambda_reg: float = 0.001,
    max_iter: int = 50,
) -> torch.Tensor:
    r"""Iterative LSQR-based dipole inversion with Tikhonov regularization.

    Solves:

    .. math::
        \hat{\chi} = \arg\min_\chi \| W(D\chi - f) \|_2^2
                     + \lambda \| \nabla\chi \|_2^2

    using conjugate gradient descent, where *D* is the dipole kernel,
    *W* is the mask weighting, and *f* is the local field.

    Args:
        local_field: Background-removed field [B, 1, D, H, W]
        mask: Brain mask [B, 1, D, H, W]
        voxel_sizes: (dz, dy, dx) in mm
        lambda_reg: Tikhonov regularization weight
        max_iter: Maximum CG iterations

    Returns:
        Susceptibility map χ [B, 1, D, H, W]
    """
    spatial_dims = (-3, -2, -1)
    W = mask.float()

    D_k = get_dipole_kernel_3d(local_field.shape, voxel_sizes).to(local_field.device)

    def forward_dipole(chi: torch.Tensor) -> torch.Tensor:
        """Apply forward dipole convolution: D * chi."""
        chi_k = torch.fft.fftn(chi, dim=spatial_dims)
        return torch.fft.ifftn(chi_k * D_k, dim=spatial_dims).real

    def adjoint_dipole(field: torch.Tensor) -> torch.Tensor:
        """Apply adjoint dipole convolution: D^H * field."""
        # D(k) is real-valued, so D^H = D
        return forward_dipole(field)

    def gradient_penalty(chi: torch.Tensor) -> torch.Tensor:
        """Compute -∇²χ (negative Laplacian as gradient regularization)."""
        return -_laplacian_3d(chi).real

    # Initialize with TKD solution
    chi = tkd_dipole_inversion(local_field, mask, voxel_sizes, threshold=0.1)

    # Conjugate gradient on the SYMMETRIC normal operator of min ‖W(Dχ−f)‖².
    # The normal equations are  (Dᴴ W² D + λ(-∇²)) χ = Dᴴ W² f, so
    #   A = Dᴴ W² D + λ(-∇²)   (self-adjoint: Dᴴ=D, W² and -∇² symmetric)
    #   b = Dᴴ W² f
    # A previous version used  W Dᴴ W D  (an extra OUTER W): since W and D do
    # not commute, (W Dᴴ W D)ᵀ = D W D W ≠ W Dᴴ W D, so it is NOT symmetric and
    # CG is not a valid solver for it. Squaring the inner weight and dropping the
    # outer W restores symmetry (for a binary mask W²=W, so this also fixes the
    # weighting without changing it there).
    W2 = W * W
    b = adjoint_dipole(W2 * local_field)
    r = b - (adjoint_dipole(W2 * forward_dipole(chi)) + lambda_reg * gradient_penalty(chi))
    p = r.clone()
    rs_old = (r * r).sum()

    for _iteration in range(max_iter):
        Ap = adjoint_dipole(W2 * forward_dipole(p)) + lambda_reg * gradient_penalty(p)
        pAp = (p * Ap).sum()
        if pAp.abs() < 1e-12:
            break

        alpha = rs_old / pAp
        chi = chi + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()

        if rs_new < 1e-10:
            break

        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new

    return chi * mask.float()
