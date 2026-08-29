r"""Structured noise covariance: k-space-diagonal plus low-rank.

The noise operator :math:`\mathcal N_\Sigma` of the identified degradation
operator (Proposal 1) has covariance

.. math::

    \Sigma = \mathcal F^{-1}\operatorname{diag}(\rho)\,\mathcal F + U U^{H},

a k-space-diagonal term (the power spectral density :math:`\rho`) plus a
low-rank correction :math:`U U^{H}` (:math:`U\in\mathbb C^{N\times r}`).
This structure makes the three quantities needed by the exact Gaussian
likelihood cheap and matrix-free:

* :meth:`StructuredCovariance.apply`  — :math:`\Sigma v` via centred FFTs.
* :meth:`StructuredCovariance.solve`  — :math:`\Sigma^{-1} v` via the
  Woodbury identity (only an ``r x r`` solve).
* :meth:`StructuredCovariance.logdet` — :math:`\log\det\Sigma` via the
  matrix-determinant lemma (:math:`\sum\log\rho + \log\det(I_r + U^H
  A^{-1} U)`).

All operations are batched over the leading axis and differentiable.
The centred, orthonormal FFT (:mod:`mriforge.infrastructure.physics.fft_ops`)
is what makes :math:`A=\mathcal F^{-1}\operatorname{diag}(\rho)\mathcal F`
diagonalisable with eigenvalues exactly :math:`\rho` — never substitute a
raw ``torch.fft`` here.
"""

from __future__ import annotations

import torch

from .fft_ops import fft2c, ifft2c

__all__ = [
    "StructuredCovariance",
    "lowrank_from_basis",
    "radial_to_grid",
]


def radial_to_grid(rho_radial: torch.Tensor, h: int, w: int) -> torch.Tensor:
    r"""Broadcast a radial PSD profile to a centred 2-D k-space grid.

    Args:
        rho_radial: ``[B, n_bins]`` non-negative radial profile (bin 0 =
            k-space centre).
        h, w: Target grid size.

    Returns:
        ``[B, 1, H, W]`` PSD grid; each k-space location samples the radial
        profile at its (normalised) distance from the centre via linear
        interpolation between adjacent bins.
    """
    b, n_bins = rho_radial.shape
    device = rho_radial.device
    real_dtype = rho_radial.dtype
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=real_dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=real_dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    radius = torch.sqrt(gx * gx + gy * gy) / (2.0**0.5)  # ∈ [0, 1]
    # Map radius → fractional bin index, then linear interpolation.
    idx = radius * (n_bins - 1)
    lo = idx.floor().clamp(0, n_bins - 1).long()
    hi = (lo + 1).clamp(0, n_bins - 1)
    frac = (idx - lo.to(real_dtype)).reshape(1, h, w)
    lo_flat = lo.reshape(-1)
    hi_flat = hi.reshape(-1)
    rho_lo = rho_radial[:, lo_flat].reshape(b, h, w)
    rho_hi = rho_radial[:, hi_flat].reshape(b, h, w)
    grid = rho_lo * (1.0 - frac) + rho_hi * frac
    return grid.reshape(b, 1, h, w)


def lowrank_from_basis(basis: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    r"""Assemble the low-rank factor ``U`` from shared basis images and per-sample gains.

    Args:
        basis: ``[r, C, H, W]`` complex basis images (shared across the batch).
        gains: ``[B, r]`` complex (or real) per-sample column gains.

    Returns:
        ``U`` of shape ``[B, N, r]`` with ``N = C*H*W``; column ``k`` is
        ``gains[:, k] * vec(basis[k])``.
    """
    r, c, h, w = basis.shape
    n = c * h * w
    basis_flat = basis.reshape(r, n).t()  # [N, r]
    u = basis_flat.unsqueeze(0) * gains.unsqueeze(1).to(basis_flat.dtype)  # [B, N, r]
    return u


class StructuredCovariance:
    r"""Sigma = F^{-1} diag(rho) F + U U^H with matrix-free apply / solve / logdet.

    Args:
        rho_grid: ``[B, 1, H, W]`` or ``[B, C, H, W]`` strictly-positive
            k-space PSD (broadcast over channels when the channel axis is 1).
        u: ``[B, N, r]`` complex low-rank factor with ``N = C*H*W``. ``None``
            or ``r == 0`` ⇒ a pure k-space-diagonal covariance.
        channels: Number of image channels ``C``.
        height, width: Spatial dims ``H``, ``W``.
        eps: Floor added to ``rho_grid`` for numerical positivity.
    """

    def __init__(
        self,
        rho_grid: torch.Tensor,
        u: torch.Tensor | None,
        *,
        channels: int,
        height: int,
        width: int,
        eps: float = 1e-6,
    ) -> None:
        self.batch = rho_grid.shape[0]
        self.channels = channels
        self.height = height
        self.width = width
        self.n = channels * height * width
        self.eps = eps
        self.rho = rho_grid.clamp_min(0.0) + eps  # [B, 1|C, H, W]
        self.u = u if (u is not None and u.shape[-1] > 0) else None
        # Working complex dtype. ``fft2c``/``ifft2c`` compute in complex64
        # internally; we cast their results back to ``_cdtype`` so every
        # einsum / linalg.solve below sees one consistent dtype (mixing
        # complex64 and complex128 trips ``torch.linalg.solve``).
        if (
            self.u is not None and self.u.dtype == torch.complex128
        ) or rho_grid.dtype == torch.float64:
            self._cdtype = torch.complex128
        else:
            self._cdtype = torch.complex64

    # ── shape helpers ────────────────────────────────────────────────
    def _flatten(self, v: torch.Tensor) -> torch.Tensor:
        return v.reshape(self.batch, self.n)

    def _unflatten(self, v: torch.Tensor) -> torch.Tensor:
        return v.reshape(self.batch, self.channels, self.height, self.width)

    # ── diagonal term A = F^{-1} diag(rho) F ──────────────────────────────
    def _apply_diag(self, v: torch.Tensor) -> torch.Tensor:
        out = ifft2c(self.rho.to(torch.complex64) * fft2c(v))
        return out.to(self._cdtype)

    def _solve_diag(self, v: torch.Tensor) -> torch.Tensor:
        out = ifft2c((1.0 / self.rho).to(torch.complex64) * fft2c(v))
        return out.to(self._cdtype)

    # ── public API ───────────────────────────────────────────────────
    def apply(self, v: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\Sigma v` for ``v`` of shape ``[B, C, H, W]``."""
        out = self._apply_diag(v)
        if self.u is not None:
            vf = self._flatten(v).to(self._cdtype)
            uh_v = torch.einsum("bnr,bn->br", self.u.conj(), vf)
            uu_v = torch.einsum("bnr,br->bn", self.u, uh_v)
            out = out + self._unflatten(uu_v).to(out.dtype)
        return out

    def solve(self, v: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\Sigma^{-1} v` via the Woodbury identity."""
        a = self._solve_diag(v)  # A⁻¹ v
        if self.u is None:
            return a
        af = self._flatten(a).to(self._cdtype)
        uf = self.u
        # M = I_r + Uᴴ A⁻¹ U
        ainv_u = self._apply_solve_diag_cols(uf)  # [B, N, r]
        m = torch.einsum("bnr,bns->brs", uf.conj(), ainv_u)
        r = uf.shape[-1]
        eye = torch.eye(r, dtype=self._cdtype, device=v.device).expand(self.batch, r, r)
        m = m + eye
        b = torch.einsum("bnr,bn->br", uf.conj(), af)  # Uᴴ A⁻¹ v
        z = torch.linalg.solve(m, b.unsqueeze(-1)).squeeze(-1)  # M⁻¹ b
        uz = torch.einsum("bnr,br->bn", uf, z)  # U z
        corr = self._solve_diag(self._unflatten(uz).to(v.dtype))  # A⁻¹ U z
        return a - corr

    def logdet(self) -> torch.Tensor:
        r"""Compute :math:`\log\det\Sigma` (real, shape ``[B]``)."""
        rho_full = self.rho.expand(self.batch, self.channels, self.height, self.width)
        logdet_a = torch.log(rho_full).sum(dim=(1, 2, 3))
        if self.u is None:
            return logdet_a
        uf = self.u
        ainv_u = self._apply_solve_diag_cols(uf)
        m = torch.einsum("bnr,bns->brs", uf.conj(), ainv_u)
        r = uf.shape[-1]
        eye = torch.eye(r, dtype=self._cdtype, device=uf.device).expand(self.batch, r, r)
        m = m + eye
        _, logabsdet = torch.linalg.slogdet(m)
        return logdet_a + logabsdet.real

    # ── internals ────────────────────────────────────────────────────
    def _apply_solve_diag_cols(self, u: torch.Tensor) -> torch.Tensor:
        """Apply ``A⁻¹`` to each of the ``r`` columns of ``U`` (``[B, N, r]``)."""
        r = u.shape[-1]
        cols = []
        for k in range(r):
            col_img = self._unflatten(u[:, :, k])
            cols.append(self._flatten(self._solve_diag(col_img)))
        return torch.stack(cols, dim=-1)  # [B, N, r]

    def quadratic_form(self, v: torch.Tensor) -> torch.Tensor:
        r"""Compute the real quadratic form :math:`v^H \Sigma^{-1} v` → ``[B]``."""
        sol = self.solve(v)
        vf = self._flatten(v)
        sf = self._flatten(sol)
        return (vf.conj() * sf).sum(dim=1).real
