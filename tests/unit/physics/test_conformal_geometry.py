"""Unit tests for the conformal_geometry physics primitives
(``TODO/audit_plan_novel.md`` SFC plan)."""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.conformal_geometry import (
    LinearBeltramiSolver,
    conformal_modulus_quadrilateral,
    disk_to_disk,
    enforce_beltrami_constraint,
    radial_riemann_map,
    teichmuller_dilatation,
    teichmuller_geodesic,
)


def test_enforce_beltrami_constraint_bounds_magnitude() -> None:
    raw = torch.complex(torch.randn(2, 8, 8), torch.randn(2, 8, 8)) * 10.0
    mu = enforce_beltrami_constraint(raw, k_max=0.9)
    assert mu.abs().amax().item() <= 0.9 + 1e-6


def test_beltrami_solver_converges_on_varying_mu() -> None:
    solver = LinearBeltramiSolver(max_iter=50, tol=1e-5)
    ys = torch.linspace(-0.5, 0.5, 32)
    Y, X = torch.meshgrid(ys, ys, indexing="ij")
    mu = 0.4 * (X + 1j * Y).unsqueeze(0)
    out = solver(mu)
    assert out.phi.shape == (1, 32, 32)
    assert out.phi.is_complex()
    assert out.residual < 1e-4
    assert out.iterations <= 30  # converges well under max_iter


def _central_wirtinger(phi: torch.Tensor, spacing: float):
    """Independent central-difference Wirtinger derivatives ``(∂_z, ∂_z̄)``.

    Deliberately computed by finite differences (not the solver's own
    spectral operators) so the Beltrami-residual check below is not
    circular.
    """

    def dx(f: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(f)
        g[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2 * spacing)
        return g

    def dy(f: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(f)
        g[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * spacing)
        return g

    dz = 0.5 * (dx(phi) - 1j * dy(phi))
    dzbar = 0.5 * (dx(phi) + 1j * dy(phi))
    return dz, dzbar


def test_beltrami_solver_recovers_prescribed_mu() -> None:
    """The recovered map φ must actually *carry* the prescribed μ.

    Regression for the 2026-07 fix: the solver previously applied the
    Beurling transform ``B = ∂_z ∂_z̄⁻¹`` instead of the anti-derivative
    ``∂_z̄⁻¹``, so φ solved a different PDE (Beltrami residual ≈ 11).
    A pure convergence check (``residual < tol``) cannot catch this — it
    only proves the *iteration* converged, not that it solved the right
    equation. Here we verify ``‖∂_z̄φ − μ ∂_zφ‖ / ‖∂_z̄φ‖ ≈ 0`` using an
    independent finite-difference derivative.
    """
    n = 48
    solver = LinearBeltramiSolver(max_iter=200, tol=1e-8)
    ax = torch.linspace(-1, 1, n)
    Y, X = torch.meshgrid(ax, ax, indexing="ij")
    # Smooth, compactly-supported μ (a Gaussian-bump complex swirl), ‖μ‖∞ < 1,
    # so the periodic-FFT solve is accurate in the support interior.
    bump = torch.exp(-((X**2 + Y**2) / (2 * 0.25**2)))
    mu = (0.5 * bump * torch.exp(1j * 2.0 * torch.atan2(Y, X))).unsqueeze(0)
    mu = mu.to(torch.complex64)

    phi = solver(mu).phi
    dz, dzbar = _central_wirtinger(phi, spacing=2.0 / (n - 1))

    interior = torch.zeros(1, n, n, dtype=torch.bool)
    interior[:, 2:-2, 2:-2] = True
    sel = interior & (mu.abs() > 0.02)

    residual = (dzbar[sel] - mu[sel] * dz[sel]).abs().mean() / dz[sel].abs().mean()
    assert residual.item() < 0.05  # ≈0.004 post-fix; was ≈11 pre-fix

    # The map is fold-free: its own Beltrami coefficient stays < 1.
    mu_phi = dzbar[sel] / dz[sel].abs().clamp_min(1e-6)
    assert mu_phi.abs().amax().item() < 1.0


def test_beltrami_solver_zero_mu_is_identity() -> None:
    """μ ≡ 0 ⇒ φ = z exactly (the conformal / no-warp base case)."""
    n = 32
    solver = LinearBeltramiSolver(max_iter=50, tol=1e-6)
    ax = torch.linspace(-1, 1, n)
    Y, X = torch.meshgrid(ax, ax, indexing="ij")
    z = (X + 1j * Y).unsqueeze(0).to(torch.complex64)
    phi = solver(torch.zeros(1, n, n, dtype=torch.complex64)).phi
    assert torch.allclose(phi, z, atol=1e-6)


def test_teichmuller_geodesic_hits_endpoints() -> None:
    mu0 = torch.full((1, 4, 4), 0.1 + 0.0j, dtype=torch.complex64)
    mu1 = torch.full((1, 4, 4), 0.5 + 0.2j, dtype=torch.complex64)
    assert torch.allclose(teichmuller_geodesic(mu0, mu1, 0.0), mu0, atol=1e-5)
    assert torch.allclose(teichmuller_geodesic(mu0, mu1, 1.0), mu1, atol=1e-5)


def test_teichmuller_geodesic_stays_in_unit_ball() -> None:
    mu0 = torch.full((1, 4, 4), 0.3 + 0.4j, dtype=torch.complex64)
    mu1 = torch.full((1, 4, 4), -0.4 + 0.3j, dtype=torch.complex64)
    for r in (0.1, 0.25, 0.5, 0.75, 0.9):
        mid = teichmuller_geodesic(mu0, mu1, r)
        assert mid.abs().amax().item() < 1.0


def test_teichmuller_dilatation_formula() -> None:
    mu = torch.full((1, 4, 4), 0.5 + 0.0j, dtype=torch.complex64)
    K = teichmuller_dilatation(mu)
    # K = (1 + 0.5) / (1 - 0.5) = 3.0
    assert torch.allclose(K, torch.tensor([3.0]), atol=1e-4)


def test_conformal_modulus_returns_scalar_per_batch() -> None:
    quads = torch.tensor(
        [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 3, dtype=torch.float32
    )
    mod = conformal_modulus_quadrilateral(quads, grid_size=12)
    assert mod.shape == (3,)
    # All three quads identical → same modulus value
    assert torch.allclose(mod, mod[0].expand(3), atol=1e-6)


def test_conformal_modulus_is_image_dependent() -> None:
    """With an ``image`` the modulus tracks image content — two different
    images over the same quad give different moduli (the fix that makes
    ConformalModulusLoss non-degenerate; before 2026-06 the modulus was
    image-independent and the loss residual was identically zero)."""
    quad = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]])
    torch.manual_seed(0)
    img_a = torch.randn(1, 1, 32, 32)
    img_b = torch.randn(1, 1, 32, 32)
    mod_a = conformal_modulus_quadrilateral(quad, grid_size=16, image=img_a)
    mod_b = conformal_modulus_quadrilateral(quad, grid_size=16, image=img_b)
    assert not torch.allclose(mod_a, mod_b)


def test_conformal_modulus_is_shape_dependent() -> None:
    """The modulus now genuinely depends on the quad shape (2026-07 de-facade).

    Previously the harmonic field was solved on the canonical square and the
    quad corners never entered, so a unit square and a 10:1 rectangle returned
    the same value. The FE solve on the physical quad makes a wide-and-short
    quad have a much larger modulus (aspect ratio ≈ width/height ≈ 10).
    """
    square = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]])
    thin = torch.tensor([[[-1.0, -0.1], [1.0, -0.1], [1.0, 0.1], [-1.0, 0.1]]])
    m_sq = conformal_modulus_quadrilateral(square, grid_size=16)
    m_thin = conformal_modulus_quadrilateral(thin, grid_size=16)
    assert torch.allclose(m_sq, torch.ones_like(m_sq), atol=0.05)
    assert m_thin.item() > 5.0 * m_sq.item()


def test_conformal_modulus_matches_rectangle_aspect_ratio() -> None:
    """A width-``a`` × height-``b`` rectangle has modulus ``a/b`` (exact for
    parallelograms), confirming this is a genuine conformal modulus."""
    def rect(a: float, b: float) -> torch.Tensor:
        return torch.tensor([[[0.0, 0.0], [a, 0.0], [a, b], [0.0, b]]])

    assert conformal_modulus_quadrilateral(rect(1.0, 1.0)).item() == pytest.approx(
        1.0, abs=0.05
    )
    assert conformal_modulus_quadrilateral(rect(3.0, 1.0)).item() == pytest.approx(
        3.0, abs=0.15
    )


def test_conformal_modulus_image_is_differentiable() -> None:
    quad = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]])
    img = torch.randn(1, 1, 24, 24, requires_grad=True)
    mod = conformal_modulus_quadrilateral(quad, grid_size=16, image=img)
    mod.sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


def test_conformal_modulus_flat_image_weight_is_unity() -> None:
    """A constant image has zero gradient → ``w ≡ 1`` → the weighted modulus
    equals the canonical (image-independent) value."""
    quad = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]])
    flat = torch.full((1, 1, 32, 32), 0.7)
    mod_img = conformal_modulus_quadrilateral(quad, grid_size=16, image=flat)
    mod_canon = conformal_modulus_quadrilateral(quad, grid_size=16)
    assert torch.allclose(mod_img, mod_canon, atol=1e-5)


def test_radial_riemann_map_is_scaling() -> None:
    z = torch.complex(torch.linspace(-1, 1, 4), torch.zeros(4))
    out = radial_riemann_map(z, radius=2.0)
    assert torch.allclose(out, z / 2.0, atol=1e-6)


def test_disk_to_disk_preserves_disk() -> None:
    # Point at origin maps to -alpha · e^{iθ}; magnitude < 1 for |alpha|<1.
    z = torch.zeros(1, dtype=torch.complex64)
    out = disk_to_disk(z, alpha=0.3 + 0.0j, theta=0.0)
    assert out.abs().item() < 1.0
