"""Quasi-conformal and conformal geometry primitives.

Shared infrastructure for the six SFC / conformal directions of
``TODO/audit_plan_novel.md`` §§194-344. All operators are
differentiable and live in the physics SSOT alongside ``fft_ops``,
``data_consistency``, ``sampling``.

Primitives:
    * :class:`LinearBeltramiSolver` — recovers a quasi-conformal map
      :math:`\\varphi` from its Beltrami coefficient :math:`\\mu` by
      solving :math:`\\partial_{\\bar z}\\varphi = \\mu\\,\\partial_z\\varphi`
      via FFT-preconditioned fixed-point iteration on the unit grid.
    * :func:`conformal_modulus_quadrilateral` — an image-content
      distortion measure (mean image-graph metric over the quad). NB
      it is **not** a true conformal modulus: the harmonic field is
      solved on the canonical square and the quad geometry never enters
      it. See the function's warning.
    * :func:`radial_riemann_map` / :func:`disk_to_disk` — analytic
      reference maps for sampling regions where Φ admits a closed form
      (radial sampling, rotation+scale of the disk).
    * :func:`teichmuller_geodesic` — closed-form geodesic *path*
      between two Beltrami coefficients in the unit ball, from
      Gardiner-Lakic [9, Ch. 6]. NB the linear ``r`` parametrisation
      traces the correct geodesic image but is **not** constant
      (hyperbolic/Teichmüller) speed — that needs ``r = tanh(s·d)``.
    * :func:`enforce_beltrami_constraint` — projects an unconstrained
      tensor to ``|μ| ≤ k < 1`` via a smooth ``tanh`` clamp.

References (inline citation numbers match the plan):
    [1] L. V. Ahlfors, *Lectures on Quasiconformal Mappings*, 2nd ed.,
        Amer. Math. Soc., 2006.
    [3] L. V. Ahlfors, *Complex Analysis*, 3rd ed., McGraw-Hill, 1979.
    [4] T. A. Driscoll & L. N. Trefethen, *Schwarz-Christoffel
        Mapping*, Cambridge Univ. Press, 2002.
    [5] D. E. Marshall & S. Rohde, "Convergence of a variant of the
        zipper algorithm for conformal mapping", *SIAM J. Numer.
        Anal.*, 45(6), 2007, 2577-2609.
    [7] N. Papamichael & N. Stylianopoulos, *Numerical Conformal
        Mapping*, World Scientific, 2010.
    [8] L. M. Lui et al., "Teichmüller mapping (T-map) and its
        applications to landmark matching registration", *SIAM J.
        Imag. Sci.*, 7(1), 2014, 391-426.
    [9] F. P. Gardiner & N. Lakic, *Quasiconformal Teichmüller
        Theory*, Math. Surveys Monogr. 76, AMS, 2000.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.nn.functional import grid_sample

logger = logging.getLogger(__name__)


def enforce_beltrami_constraint(raw: torch.Tensor, *, k_max: float = 0.95) -> torch.Tensor:
    """Smooth projection of a complex tensor to ``|μ| ≤ k_max < 1``.

    Pre-activation: ``raw`` is a complex tensor with unconstrained
    magnitude. Output preserves the argument of ``raw`` and squashes
    the magnitude through ``k_max * tanh``. Ahlfors [1, V.B] requires
    ``k_max < 1`` for the resulting map to be a homeomorphism.
    """
    mag = raw.abs().clamp_min(1e-12)
    phase = raw / mag
    return k_max * torch.tanh(mag) * phase


@dataclass(frozen=True)
class LinearBeltramiSolverResult:
    phi: torch.Tensor  # complex [B, H, W] — the recovered q-c map values on the grid
    iterations: int
    residual: float


class LinearBeltramiSolver(torch.nn.Module):
    """FFT-preconditioned fixed-point Beltrami solver.

    Given a Beltrami coefficient :math:`\\mu` on the unit disk, solves
    :math:`\\partial_{\\bar z}\\varphi = \\mu\\,\\partial_z\\varphi`
    with boundary condition :math:`\\varphi(z) = z` on
    :math:`\\partial\\mathbb{D}`. Decomposes :math:`\\varphi(z) = z +
    h(z)`, so the equation becomes :math:`\\partial_{\\bar z} h =
    \\mu\\,(1 + \\partial_z h)` (using :math:`\\partial_z z = 1`,
    :math:`\\partial_{\\bar z} z = 0`). Each fixed-point step recovers
    :math:`h` by applying the **anti-derivative**
    :math:`\\partial_{\\bar z}^{-1}` (Cauchy transform; spectral
    multiplier :math:`1/\\mathrm{sym}(\\partial_{\\bar z})`, DC mode
    projected out, since the mean of :math:`h` is fixed by the boundary
    condition) to :math:`\\mu(1 + \\partial_z h)`.

    Note this is *not* the Beurling transform :math:`B = \\partial_z
    \\partial_{\\bar z}^{-1}` (unit modulus): applying :math:`B` would
    return :math:`\\partial_z h`, not :math:`h`. Applying
    :math:`\\partial_z` to the update shows the derivative
    :math:`g = \\partial_z h` obeys the Beurling fixed point
    :math:`g \\mapsto B[\\mu(1+g)]` with :math:`\\|B\\|_{L^2}=1`, hence
    the contraction rate is ``‖μ‖_∞`` per iteration; for ``k_max = 0.9``
    that is ``log(tol)/log(0.9) ≈ 22`` iterations at ``tol = 1e-2``.
    See [1, V.B], [8, §3.2].
    """

    def __init__(
        self,
        *,
        max_iter: int = 30,
        tol: float = 1e-3,
    ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

    def _spectral_dz_dzbar(self, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(s_z, s_zbar)`` Wirtinger spectral multipliers.

        Multipliers are for the canonical ``[-1, 1]^2`` grid (physical
        spacing ``2/N``, domain length 2), so they are unit-consistent
        with the boundary term ``∂_z z = 1``. With the FFT convention
        ``∂_x → i k_x`` (torch ``ifft`` reconstructs ``e^{+i k x}``):

            ``s_z    = ½(k_y + i k_x)``   for ``∂_z  = ½(∂_x − i∂_y)``
            ``s_zbar = ½(−k_y + i k_x)``  for ``∂_z̄ = ½(∂_x + i∂_y)``
        """
        _, H, W = mu.shape
        device = mu.device
        # Angular wavenumbers on the physical [-1, 1] grid (length 2).
        ky = torch.fft.fftfreq(H, d=2.0 / H, device=device) * (2 * torch.pi)
        kx = torch.fft.fftfreq(W, d=2.0 / W, device=device) * (2 * torch.pi)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        s_z = 0.5 * (KY + 1j * KX)
        s_zbar = 0.5 * (-KY + 1j * KX)
        return s_z, s_zbar

    def forward(self, mu: torch.Tensor) -> LinearBeltramiSolverResult:
        """Solve for :math:`\\varphi` given Beltrami coefficient :math:`\\mu`.

        Args:
            mu: Complex tensor ``[B, H, W]`` with ``|mu| < 1`` pointwise.

        Returns:
            Result containing complex ``phi[B, H, W]``, iterations used,
            and the final residual norm.
        """
        if mu.dim() != 3 or not mu.is_complex():
            raise ValueError("LinearBeltramiSolver expects complex mu of shape [B, H, W]")
        B, H, W = mu.shape
        device, dtype = mu.device, mu.dtype

        # z = x + iy on the canonical [-1, 1] grid.
        ys = torch.linspace(-1, 1, H, device=device, dtype=torch.float32)
        xs = torch.linspace(-1, 1, W, device=device, dtype=torch.float32)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        z = (X + 1j * Y).to(dtype).unsqueeze(0).expand(B, -1, -1)

        s_z, s_zbar = self._spectral_dz_dzbar(mu)
        # Anti-derivative ∂_z̄⁻¹ multiplier (Cauchy transform on the torus):
        # invert the ∂_z̄ symbol and project out the DC nullspace (the mean of
        # h is fixed by the boundary condition φ = z). NB: this is the inverse
        # of ∂_z̄, NOT the Beurling transform ∂_z∂_z̄⁻¹ — see the class docstring.
        inv_zbar = torch.zeros_like(s_zbar)
        nonzero = s_zbar.abs() > 1e-12
        inv_zbar[nonzero] = s_zbar[nonzero].reciprocal()

        h = torch.zeros_like(z)
        residual = float("inf")
        it = 0
        for it in range(1, self.max_iter + 1):
            # ∂_z h via spectral derivative (h is the periodic perturbation).
            dz_h = torch.fft.ifft2(s_z * torch.fft.fft2(h))
            # Beltrami equation for the perturbation: ∂_z̄ h = μ (1 + ∂_z h).
            rhs = mu * (1.0 + dz_h)
            # Recover h = ∂_z̄⁻¹[rhs] (applying the Beurling transform here would
            # instead yield ∂_z h, the original bug).
            h_new = torch.fft.ifft2(inv_zbar * torch.fft.fft2(rhs))
            residual = (h_new - h).abs().mean().item()
            h = h_new
            if residual < self.tol:
                break
        return LinearBeltramiSolverResult(phi=z + h, iterations=it, residual=residual)


def _quad_metric_weight(
    image: torch.Tensor, quad_corners: torch.Tensor, grid_size: int
) -> torch.Tensor:
    """Sample ``w = sqrt(1 + |grad I|^2)`` on each quad's ``NxN`` region.

    The quad's four corners (cyclic, in normalised ``[-1, 1]`` image
    coordinates) define a bilinear map from the unit square; the image is
    bilinearly sampled on that warped grid (differentiable via
    ``grid_sample``) and the per-pixel image-graph metric weight is returned.
    A flat image gives ``w ≡ 1`` (recovering the canonical, image-independent
    modulus), while structured anatomy gives a spatially-varying weight — so
    two different images over the same quad yield different moduli.
    """
    b = quad_corners.shape[0]
    n = grid_size
    img = image
    if img.is_complex():
        img = img.abs()
    if img.dim() == 5:  # [B, C, D, H, W] -> middle slice
        img = img[:, :, img.shape[2] // 2]
    img = img.mean(dim=1, keepdim=True).to(torch.float64)  # [B, 1, H, W]
    if img.shape[0] == 1 and b > 1:
        img = img.expand(b, -1, -1, -1)

    # Bilinear unit-square -> quad map. Corners cyclic: v1=(0,0) v2=(1,0)
    # v3=(1,1) v4=(0,1). P(s,t) = (1-s)(1-t)v1 + s(1-t)v2 + s t v3 + (1-s) t v4.
    lin = torch.linspace(0.0, 1.0, n, device=image.device, dtype=torch.float64)
    t_grid, s_grid = torch.meshgrid(lin, lin, indexing="ij")  # [N, N]
    s = s_grid.reshape(1, n, n, 1)
    t = t_grid.reshape(1, n, n, 1)
    c = quad_corners.to(torch.float64)  # [B, 4, 2]
    v1, v2, v3, v4 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]  # each [B, 2]
    v1 = v1.reshape(b, 1, 1, 2)
    v2 = v2.reshape(b, 1, 1, 2)
    v3 = v3.reshape(b, 1, 1, 2)
    v4 = v4.reshape(b, 1, 1, 2)
    grid = (
        (1 - s) * (1 - t) * v1 + s * (1 - t) * v2 + s * t * v3 + (1 - s) * t * v4
    )  # [B, N, N, 2] in [-1, 1] (x, y)
    sampled = grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # [B, 1, N, N]
    si = sampled[:, 0]  # [B, N, N]
    gy = (si[:, 2:, :] - si[:, :-2, :]) * 0.5 * (n - 1)
    gx = (si[:, :, 2:] - si[:, :, :-2]) * 0.5 * (n - 1)
    grad_sq = gy[:, :, 1:-1].pow(2) + gx[:, 1:-1, :].pow(2)  # [B, N-2, N-2]
    return torch.sqrt(1.0 + grad_sq)


# Dense modulus solve is O((N^2)^3); cap the FE resolution used for the solve.
_MODULUS_SOLVE_MAX_N = 20


def _quadrilateral_modulus_geometric(quad_corners: torch.Tensor, grid_size: int) -> torch.Tensor:
    r"""Genuine conformal modulus of each quadrilateral (shape-dependent).

    Solves the harmonic Dirichlet problem on the *physical* quad — ``u=0`` on
    the ``v0→v1`` edge, ``u=1`` on the opposite ``v3→v2`` edge, natural
    (Neumann) on the other two — and returns the Dirichlet energy
    :math:`\int_Q|\nabla u|^2\,dA`, which is the conformal modulus (the aspect
    ratio :math:`a/b` for which :math:`Q\cong[0,a]\times[0,b]`).

    Discretised with isoparametric bilinear (Q1) finite elements on the
    bilinear map from the unit square to the quad (1-point quadrature). Exact
    for parallelograms (a rectangle of width ``a``, height ``b`` returns
    ``a/b``); genuinely shape-dependent for general quads. Differentiable in
    ``quad_corners``. Returns ``[B]`` (float64).
    """
    n = int(min(grid_size, _MODULUS_SOLVE_MAX_N))
    b = quad_corners.shape[0]
    device = quad_corners.device
    qc = quad_corners.to(torch.float64)
    v0, v1, v2, v3 = qc[:, 0], qc[:, 1], qc[:, 2], qc[:, 3]  # [B, 2]

    lin = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float64)
    t_g, s_g = torch.meshgrid(lin, lin, indexing="ij")  # [N, N]: i→t (rows), j→s (cols)
    s = s_g.reshape(1, n, n, 1)
    t = t_g.reshape(1, n, n, 1)
    # Bilinear map: P(s,t)=(1-s)(1-t)v0 + s(1-t)v1 + s·t·v2 + (1-s)·t·v3.
    p = (
        (1 - s) * (1 - t) * v0.view(b, 1, 1, 2)
        + s * (1 - t) * v1.view(b, 1, 1, 2)
        + s * t * v2.view(b, 1, 1, 2)
        + (1 - s) * t * v3.view(b, 1, 1, 2)
    )  # [B, N, N, 2]

    # Per-cell corners: a=(i,j) b=(i,j+1) c=(i+1,j) d=(i+1,j+1).
    pa, pb_, pc, pd = p[:, :-1, :-1], p[:, :-1, 1:], p[:, 1:, :-1], p[:, 1:, 1:]
    dxi = 0.5 * ((pb_ - pa) + (pd - pc))  # ∂P/∂ξ (s-dir), [B, Nc, Nc, 2]
    deta = 0.5 * ((pc - pa) + (pd - pb_))  # ∂P/∂η (t-dir)
    det = dxi[..., 0] * deta[..., 1] - deta[..., 0] * dxi[..., 1]  # [B, Nc, Nc]
    det_safe = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    # Reference-gradient of the 4 shape fns at the cell centre, [4, 2]=[ξ, η].
    gref = torch.tensor(
        [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
        device=device,
        dtype=torch.float64,
    )
    # Jac^{-1} (2×2) per cell; Jac=[[dxi_x, deta_x],[dxi_y, deta_y]].
    inv = torch.empty(b, n - 1, n - 1, 2, 2, device=device, dtype=torch.float64)
    inv[..., 0, 0] = deta[..., 1] / det_safe
    inv[..., 0, 1] = -deta[..., 0] / det_safe
    inv[..., 1, 0] = -dxi[..., 1] / det_safe
    inv[..., 1, 1] = dxi[..., 0] / det_safe
    # Physical shape-fn gradients: grad_phys[k] = gref[k] @ Jac^{-1}, [B,Nc,Nc,4,2].
    grad_phys = torch.einsum("ka,bijav->bijkv", gref, inv)
    # Element stiffness Ke[a,b] = |det|·(grad_phys[a]·grad_phys[b]), [B,Nc,Nc,4,4].
    ke = det_safe.abs()[..., None, None] * torch.einsum("bijav,bijcv->bijac", grad_phys, grad_phys)

    # Global node indices of the 4 corners per cell (flattened i*N+j).
    ii, jj = torch.meshgrid(
        torch.arange(n - 1, device=device),
        torch.arange(n - 1, device=device),
        indexing="ij",
    )
    na = (ii * n + jj).reshape(-1)
    nodes = torch.stack([na, na + 1, na + n, na + n + 1], dim=1)  # [Ncells, 4]

    nn = n * n
    dirichlet = torch.zeros(nn, dtype=torch.bool, device=device)
    dirichlet[:n] = True  # t=0 row → u=0
    dirichlet[-n:] = True  # t=1 row → u=1
    u_d = torch.zeros(nn, dtype=torch.float64, device=device)
    u_d[-n:] = 1.0
    free = ~dirichlet

    moduli = []
    for bi in range(b):
        big_k = torch.zeros(nn, nn, dtype=torch.float64, device=device)
        ke_b = ke[bi].reshape(-1, 4, 4)  # [Ncells, 4, 4]
        rows = nodes[:, :, None].expand(-1, 4, 4).reshape(-1)
        cols = nodes[:, None, :].expand(-1, 4, 4).reshape(-1)
        big_k.index_put_((rows, cols), ke_b.reshape(-1), accumulate=True)
        k_ff = big_k[free][:, free]
        k_fd = big_k[free][:, dirichlet]
        rhs = -k_fd @ u_d[dirichlet]
        u = u_d.clone()
        u[free] = torch.linalg.solve(k_ff, rhs)
        moduli.append(u @ (big_k @ u))  # Dirichlet energy = modulus
    return torch.stack(moduli)  # [B]


def conformal_modulus_quadrilateral(
    quad_corners: torch.Tensor,
    *,
    grid_size: int = 32,
    image: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Conformal modulus of a quadrilateral region (optionally image-weighted).

    The modulus of a quadrilateral :math:`Q` is the aspect ratio :math:`a/b`
    for which :math:`Q` is conformally equivalent to :math:`[0,a]\times[0,b]`,
    obtained from the Dirichlet energy of the harmonic function with boundary
    ``0`` on the ``v0→v1`` edge, ``1`` on the opposite edge, and Neumann on the
    sides (:func:`_quadrilateral_modulus_geometric`). This value is genuinely
    shape-dependent: a square returns ``≈1`` and a 3:1 rectangle returns
    ``≈3`` (previously the ``quad_corners`` never entered the solve and every
    quad returned the same number).

    Args:
        quad_corners: ``[B, 4, 2]`` real corner coordinates (cyclic, ``v0..v3``),
            in normalised ``[-1, 1]`` image coordinates when ``image`` is given.
        grid_size: Node resolution for the FE solve (capped at
            :data:`_MODULUS_SOLVE_MAX_N`; the dense solve is ``O((N^2)^3)``).
        image: optional ``[B, C, H, W]`` (or 5-D) image. When supplied, the
            geometric modulus is scaled by the mean image-graph metric
            ``w = sqrt(1 + |∇I|²)`` over the quad, so the result depends on the
            image content — two reconstructions of the same anatomy over the
            same quad give different moduli (this is what
            :class:`~spectramr.models.losses.conformal_geometry.ConformalModulusLoss`
            relies on). When ``None`` the pure geometric modulus is returned.
    """
    if quad_corners.dim() != 3 or quad_corners.shape[-2:] != (4, 2):
        raise ValueError("expected [B, 4, 2] quad_corners tensor")
    mod_geo = _quadrilateral_modulus_geometric(quad_corners, grid_size)  # [B]
    if image is None:
        return mod_geo.to(quad_corners.dtype)
    # Scale the genuine geometric modulus by the mean image-graph metric so the
    # value tracks image content (keeps ConformalModulusLoss non-degenerate).
    w = _quad_metric_weight(image, quad_corners, min(grid_size, _MODULUS_SOLVE_MAX_N))
    return (mod_geo * w.flatten(1).mean(dim=1)).to(quad_corners.dtype)


def radial_riemann_map(z: torch.Tensor, *, radius: float = 1.0) -> torch.Tensor:
    """Conformal map from a centred disk of radius ``r`` to the unit disk.

    Trivial rescaling — the only closed-form Riemann map relevant to
    radial k-space sampling. For polygonal regions use a
    Schwarz-Christoffel solver from [4]; we don't ship one here.
    """
    return z / radius


def disk_to_disk(z: torch.Tensor, *, alpha: complex, theta: float) -> torch.Tensor:
    """Möbius automorphism of the unit disk.

    :math:`\\phi(z) = e^{i\\theta}(z - \\alpha) / (1 - \\bar\\alpha z)`.
    Used to fix the three-parameter ambiguity of the Riemann map
    (Section 5 of the plan, anchor points on the flattened cortex).
    """
    if abs(alpha) >= 1.0:
        raise ValueError("alpha must lie strictly inside the unit disk")
    real_dtype = z.real.dtype
    rot = torch.tensor(theta, dtype=real_dtype, device=z.device)
    e_it = torch.complex(torch.cos(rot), torch.sin(rot)).to(z.dtype)
    a = torch.tensor(alpha, dtype=z.dtype, device=z.device)
    return e_it * (z - a) / (1 - a.conj() * z)


def teichmuller_geodesic(
    mu0: torch.Tensor, mu1: torch.Tensor, r: float | torch.Tensor
) -> torch.Tensor:
    """Geodesic from ``mu0`` to ``mu1`` in the Beltrami unit ball at
    parameter ``r ∈ [0, 1]``.

    Construction (Gardiner-Lakic [9, Ch. 6]): translate ``mu0`` to
    the origin via the Möbius automorphism, scale linearly by ``r``,
    then translate back. Reaches ``mu0`` at ``r=0`` and ``mu1`` at
    ``r=1`` exactly. Stays inside the unit ball throughout when both
    endpoints do.

    NB: this traces the correct geodesic *path* (image), but the linear ``r``
    is **not** a constant-speed (hyperbolic / Teichmüller) parametrisation —
    constant speed would need ``r = tanh(s · d)``. Callers that only need the
    set of points on the geodesic are unaffected.
    """
    if isinstance(r, float):
        r_t = torch.as_tensor(r, dtype=mu0.real.dtype, device=mu0.device)
    else:
        r_t = r
    one_minus = 1.0 - mu0.conj() * mu1
    safe_denom = torch.where(
        one_minus.abs() > 1e-12,
        one_minus,
        torch.full_like(one_minus, 1e-12 + 0j),
    )
    nu1 = (mu1 - mu0) / safe_denom
    nu_r = r_t * nu1
    return (mu0 + nu_r) / (1.0 + mu0.conj() * nu_r)


def teichmuller_dilatation(mu: torch.Tensor) -> torch.Tensor:
    """Maximal dilatation ``K = (1 + ‖μ‖_∞) / (1 - ‖μ‖_∞)``."""
    k = mu.abs().amax(dim=tuple(range(1, mu.dim()))).clamp_max(0.999)
    return (1.0 + k) / (1.0 - k)


__all__ = [
    "LinearBeltramiSolver",
    "LinearBeltramiSolverResult",
    "conformal_modulus_quadrilateral",
    "disk_to_disk",
    "enforce_beltrami_constraint",
    "radial_riemann_map",
    "teichmuller_dilatation",
    "teichmuller_geodesic",
]
