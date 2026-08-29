r"""Split Bregman Solver for L1-TV Regularised Image Reconstruction.

Implements the Split Bregman method (Goldstein & Osher, 2009) for solving
the model-based image reconstruction problem from Koolstra et al. (2021):

.. math::

    \hat{\mathbf{m}} = \arg\min_{\mathbf{m}} \left\{
        \frac{\mu}{2} \| E \mathbf{m} - \mathbf{y} \|_2^2
        + \frac{\lambda}{2} \bigl(
            \|\nabla_x \mathbf{m}\|_1 + \|\nabla_y \mathbf{m}\|_1
        \bigr)
    \right\}

The Split Bregman reformulation introduces auxiliary variables
:math:`d_x, d_y` and Bregman iterates :math:`b_x, b_y`:

.. math::

    \text{Inner:} \quad
    \mathbf{m}^{k+1} = \arg\min_{\mathbf{m}} \left\{
        \frac{\mu}{2} \|E\mathbf{m} - \mathbf{y}\|_2^2
        + \frac{\beta}{2} \bigl(
            \|\nabla_x \mathbf{m} - d_x^k + b_x^k\|_2^2
            + \|\nabla_y \mathbf{m} - d_y^k + b_y^k\|_2^2
        \bigr)
    \right\}

    d_x^{k+1} = \text{shrink}(\nabla_x \mathbf{m}^{k+1} + b_x^k, \lambda/\beta)

    b_x^{k+1} = b_x^k + \nabla_x \mathbf{m}^{k+1} - d_x^{k+1}

The inner minimization is a quadratic problem solved via CG.

Paper defaults: 2 outer iterations, 1 inner iteration, CG tol = 1e-2.

References:
    - Goldstein T, Osher S. "The split Bregman method for L1-regularized
      problems." SIAM J. Imaging Sci. 2(2):323–343, 2009.
    - Koolstra K et al., MAGMA (2021) 34:631–642, §2.5.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _shrink(x: torch.Tensor, threshold: float) -> torch.Tensor:
    r"""Complex soft thresholding (shrinkage operator).

    .. math::

        \text{shrink}(x, t) = \frac{x}{|x|} \max(|x| - t, 0)

    Args:
        x: Complex tensor.
        threshold: Shrinkage threshold.

    Returns:
        Thresholded tensor.
    """
    mag = torch.abs(x)
    scale = torch.clamp(mag - threshold, min=0.0) / (mag + 1e-12)
    return x * scale


def _grad_x(m: torch.Tensor) -> torch.Tensor:
    """Forward finite difference in x (columns)."""
    gx = torch.zeros_like(m)
    gx[..., :-1] = m[..., 1:] - m[..., :-1]
    return gx


def _grad_y(m: torch.Tensor) -> torch.Tensor:
    """Forward finite difference in y (rows)."""
    gy = torch.zeros_like(m)
    gy[..., :-1, :] = m[..., 1:, :] - m[..., :-1, :]
    return gy


def _div_x(gx: torch.Tensor) -> torch.Tensor:
    """Hermitian adjoint :math:`D^T` of the forward-difference ``grad_x``.

    Implements the true (positive) transpose so that
    ``div_x(grad_x(.)) = D^T D`` is positive semi-definite.
    """
    div = torch.zeros_like(gx)
    div[..., 1:] += gx[..., :-1]
    div[..., :-1] -= gx[..., :-1]
    return div


def _div_y(gy: torch.Tensor) -> torch.Tensor:
    """Hermitian adjoint :math:`D^T` of the forward-difference ``grad_y``.

    Implements the true (positive) transpose so that
    ``div_y(grad_y(.)) = D^T D`` is positive semi-definite.
    """
    div = torch.zeros_like(gy)
    div[..., 1:, :] += gy[..., :-1, :]
    div[..., :-1, :] -= gy[..., :-1, :]
    return div


def _laplacian(m: torch.Tensor) -> torch.Tensor:
    r""":math:`\nabla^T \nabla = D_x^T D_x + D_y^T D_y` (PSD).

    This is ``div_x(grad_x) + div_y(grad_y)`` with NO leading sign flip;
    it equals the continuous negative Laplacian :math:`-\Delta` and is
    positive semi-definite, which keeps the CG normal operator SPD. Do
    NOT negate it.
    """
    return _div_x(_grad_x(m)) + _div_y(_grad_y(m))


class SplitBregmanSolver:
    r"""Split Bregman solver for model-based MRI reconstruction.

    Solves L1-TV regularised problems with CG inner loop.

    Args:
        forward_op: Callable m → Em (forward encoding).
        adjoint_op: Callable y → E†y (adjoint encoding).
        mu: Data fidelity weight.
        lambda_tv: TV regularisation weight.
        beta: Split Bregman penalty parameter (auto-set to lambda_tv if None).
        outer_iterations: Number of Bregman outer iterations.
        inner_iterations: Number of inner minimisation iterations.
        cg_max_iter: Maximum CG iterations per inner solve.
        cg_tolerance: CG convergence tolerance.
    """

    def __init__(
        self,
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        adjoint_op: Callable[[torch.Tensor], torch.Tensor],
        mu: float = 1.0,
        lambda_tv: float = 1e-10,
        beta: float | None = None,
        outer_iterations: int = 2,
        inner_iterations: int = 1,
        cg_max_iter: int = 50,
        cg_tolerance: float = 1e-2,
    ) -> None:
        self.E = forward_op
        self.EH = adjoint_op
        self.mu = mu
        self.lam = lambda_tv
        self.beta = beta if beta is not None else max(lambda_tv * 10, 1e-4)
        self.n_outer = outer_iterations
        self.n_inner = inner_iterations
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tolerance

    def _apply_normal_plus_reg(self, m: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`(\mu \cdot E^\dagger E + \beta \cdot \nabla^T \nabla) m`.

        The operator :math:`\nabla^T \nabla` (``_laplacian``) is the
        **negative** discrete Laplacian, which is positive semi-definite.
        Combined with the positive-definite normal operator this ensures
        the full LHS is SPD so CG converges.
        """
        data_term = self.mu * self.EH(self.E(m))
        reg_term = self.beta * _laplacian(m)  # ∇ᵀ∇ is already PSD
        return data_term + reg_term

    def _cg_solve(
        self,
        rhs: torch.Tensor,
        m_init: torch.Tensor,
    ) -> torch.Tensor:
        """Conjugate gradient solver for the inner quadratic problem.

        Solves: (μ·E†E + β·(-Δ))·m = rhs.

        Args:
            rhs: Right-hand side.
            m_init: Initial guess.

        Returns:
            Solution m.
        """
        m = m_init.clone()
        r = rhs - self._apply_normal_plus_reg(m)
        p = r.clone()
        rs_old = torch.sum(r.conj() * r).real

        for _ in range(self.cg_max_iter):
            Ap = self._apply_normal_plus_reg(p)
            pAp = torch.sum(p.conj() * Ap).real

            if pAp.abs() < 1e-30:
                break

            alpha = rs_old / pAp
            m = m + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r.conj() * r).real

            if rs_new.sqrt() < self.cg_tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return m

    def solve(
        self,
        y: torch.Tensor,
        m_init: torch.Tensor | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        r"""Run Split Bregman iterations.

        Args:
            y: K-space measurements (flattened or shaped).
            m_init: Initial image estimate. If None, uses E†y.
            image_shape: (H, W) shape for reshaping. Required if m_init
                is None.

        Returns:
            Reconstructed complex image [1, 1, H, W].
        """
        # Initialise
        EHy = self.EH(y)

        if m_init is not None:
            m = m_init.clone()
        else:
            m = EHy.clone()

        # Ensure complex
        if not torch.is_complex(m):
            m = m.to(torch.complex64)

        # Bregman variables
        dx = torch.zeros_like(m)
        dy = torch.zeros_like(m)
        bx = torch.zeros_like(m)
        by = torch.zeros_like(m)

        threshold = self.lam / (self.beta + 1e-12)

        for _ in range(self.n_outer):
            for _ in range(self.n_inner):
                # ── CG inner solve ─────────────────────────────────
                # RHS: μ·E†y + β·(div_x(dx - bx) + div_y(dy - by))
                rhs = self.mu * EHy + self.beta * (_div_x(dx - bx) + _div_y(dy - by))

                m = self._cg_solve(rhs, m)

            # ── Update auxiliary variables (shrinkage) ─────────────
            gx_m = _grad_x(m)
            gy_m = _grad_y(m)

            dx = _shrink(gx_m + bx, threshold)
            dy = _shrink(gy_m + by, threshold)

            # ── Update Bregman iterates ───────────────────────────
            bx = bx + gx_m - dx
            by = by + gy_m - dy

        return m


__all__ = ["SplitBregmanSolver"]
