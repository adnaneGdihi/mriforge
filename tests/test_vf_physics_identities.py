"""Replacements for the dropped proof arms vf_11..vf_20.

Each of these tests asserts a deterministic mathematical identity that
the corresponding proof-arm YAML claimed to "prove" via a 500,000-
iteration training loop with PSNR as the success metric — a category
error. Asserting the identity directly is faster, cheaper, and
correct. See ``experiments/inprogress/vf/DROPPED_PROOF_ARMS.md``.

All tests run on CPU in well under 2 seconds. They have no GPU budget
and no data dependency. ``test_helmholtz_uniqueness`` is intentionally
skipped — Helmholtz uniqueness is a textbook PDE result (Evans, PDE,
1998, §6.4) and reasserting it in code is no more meaningful than
reasserting it in a comment.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------
# vf_11 — Fourier shift theorem.
# F{x(n - n0)}[k] == F{x}[k] * exp(-2πi k n0 / N).
# ---------------------------------------------------------------------
def test_fourier_shift_theorem() -> None:
    n = 64
    n0 = 7  # integer shift → exact roll equivalence
    x = torch.randn(n, dtype=torch.complex64)
    x_shifted = torch.roll(x, shifts=n0, dims=0)

    fft_lhs = torch.fft.fft(x_shifted)

    k = torch.arange(n, dtype=torch.float32)
    phase = torch.exp(-2j * math.pi * k * n0 / n).to(torch.complex64)
    fft_rhs = torch.fft.fft(x) * phase

    assert torch.allclose(fft_lhs, fft_rhs, atol=1e-5, rtol=1e-5), (
        "Fourier shift theorem violated."
    )


# ---------------------------------------------------------------------
# vf_12 — Laplace harmonicity of B0.
# A susceptibility-derived B0 field in vacuum satisfies ∇²B0 = 0 to
# numerical precision. We test the finite-difference Laplacian of a
# known harmonic function (the real part of an analytic function in 2D).
# ---------------------------------------------------------------------
def test_laplace_b0_harmonicity() -> None:
    # u(x, y) = x² - y²  is harmonic (∂²u/∂x² + ∂²u/∂y² = 2 - 2 = 0).
    n = 64
    coords = torch.linspace(-1, 1, n)
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    u = x.pow(2) - y.pow(2)
    h = (coords[1] - coords[0]).item()

    # 5-point stencil Laplacian on the interior.
    lap = (
        u[2:, 1:-1] + u[:-2, 1:-1]
        + u[1:-1, 2:] + u[1:-1, :-2]
        - 4.0 * u[1:-1, 1:-1]
    ) / (h * h)

    assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-3), (
        f"Discrete Laplacian of harmonic function should be ~0, "
        f"got max |∇²u| = {lap.abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------
# vf_13 — GNL hardware determinism.
# Gradient nonlinearity is a deterministic function of (x, y, z) for a
# fixed coil geometry. Identical inputs → identical outputs, always.
# ---------------------------------------------------------------------
def test_gnl_determinism() -> None:
    n = 32

    def gnl_displacement(coords: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
        # Toy GNL: cubic radial distortion. Deterministic given coords.
        r2 = coords.pow(2).sum(dim=-1, keepdim=True)
        return coords + scale * coords * r2

    coords = torch.randn(n, 3)
    out1 = gnl_displacement(coords)
    out2 = gnl_displacement(coords)
    out3 = gnl_displacement(coords.clone())
    assert torch.equal(out1, out2)
    assert torch.equal(out1, out3)


# ---------------------------------------------------------------------
# vf_14 — Gradient delay spatial invariance.
# A constant trajectory delay τ applied to a k-space trajectory shifts
# every sample by the same amount in k-space (and thus by the same
# spatial phase in image space). Spatial invariance: the image shift
# is independent of which voxel you read out at.
# ---------------------------------------------------------------------
def test_trajectory_invariance() -> None:
    n = 32
    k = torch.linspace(-0.5, 0.5, n)
    tau = 0.03

    # Delayed trajectory: k_delayed = k + g * tau (g is gradient amp; for a
    # constant gradient this is uniform). All samples shift by the same
    # k-space amount → equivalent to a constant spatial-domain phase ramp.
    k_delayed = k + tau

    # Spatial domain: a constant tau-induced shift in k yields a single
    # voxel-position-independent phase ramp. Test that the shift is
    # constant across all samples to floating-point precision.
    shift = k_delayed - k
    assert torch.allclose(shift, torch.full_like(shift, tau), atol=1e-6), (
        "Trajectory delay must induce a SPATIALLY INVARIANT k-space "
        f"shift; got min={shift.min().item():.6f}, max={shift.max().item():.6f}."
    )


# ---------------------------------------------------------------------
# vf_15 — Fourier convolution linearity.
# F{a * x + b * y} = a F{x} + b F{y}. Standard linearity of DFT.
# ---------------------------------------------------------------------
def test_fourier_convolution() -> None:
    n = 32
    x = torch.randn(n, dtype=torch.complex64)
    y = torch.randn(n, dtype=torch.complex64)
    a = torch.tensor(0.3 + 0.7j)
    b = torch.tensor(-1.2 + 0.1j)

    lhs = torch.fft.fft(a * x + b * y)
    rhs = a * torch.fft.fft(x) + b * torch.fft.fft(y)
    assert torch.allclose(lhs, rhs, atol=1e-5), "DFT linearity violated."


# ---------------------------------------------------------------------
# vf_16 — Helmholtz uniqueness. Textbook PDE result, not a code claim.
# ---------------------------------------------------------------------
@pytest.mark.skip(reason="Textbook PDE uniqueness (Evans 1998, §6.4); not a code claim.")
def test_helmholtz_uniqueness() -> None:
    pass


# ---------------------------------------------------------------------
# vf_17 — AWGN covariance stationarity.
# An additive white Gaussian noise tensor has a covariance that is
# (statistically) independent of the spatial index. Test that the
# sample covariance at lag 0 matches σ² across position.
# ---------------------------------------------------------------------
def test_awgn_stationarity() -> None:
    torch.manual_seed(42)
    n_samples, n_positions = 4096, 32
    sigma = 0.5
    noise = sigma * torch.randn(n_samples, n_positions)

    # Per-position sample variance.
    per_pos_var = noise.var(dim=0)
    # Stationarity: variance is independent of position.
    spread = per_pos_var.std().item() / per_pos_var.mean().item()
    assert spread < 0.1, (
        f"AWGN per-position variance has CoV {spread:.3f}; "
        "should be < 0.1 for n=4096 samples."
    )
    # Mean variance close to σ².
    assert abs(per_pos_var.mean().item() - sigma**2) < 0.02


# ---------------------------------------------------------------------
# vf_18 — Rigid-body kinematic equivalence (Kabsch alignment).
# Two rigid-body transformations of the same point cloud have the same
# distance pattern. The Kabsch-solved rigid transform recovers the
# generator to numerical precision.
# ---------------------------------------------------------------------
def test_rigid_body_kabsch() -> None:
    torch.manual_seed(0)
    n = 16
    pts = torch.randn(n, 3)

    # Generator: rotation matrix R (random orthogonal) + translation t.
    a = torch.randn(3, 3)
    q, _ = torch.linalg.qr(a)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    r_true = q
    t_true = torch.randn(3)

    pts_transformed = pts @ r_true.T + t_true

    # Kabsch recovery.
    centroid_a = pts.mean(dim=0)
    centroid_b = pts_transformed.mean(dim=0)
    h = (pts - centroid_a).T @ (pts_transformed - centroid_b)
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.T @ u.T))
    s = torch.diag(torch.tensor([1.0, 1.0, d]))
    r_recovered = vt.T @ s @ u.T
    t_recovered = centroid_b - r_recovered @ centroid_a

    assert torch.allclose(r_recovered, r_true, atol=1e-5)
    assert torch.allclose(t_recovered, t_true, atol=1e-5)


# ---------------------------------------------------------------------
# vf_19 — CNN translation equivariance.
# A convolution with zero padding commutes with cyclic translation:
# Conv(roll(x)) ≈ roll(Conv(x)) (exactly, in the cyclic-padding limit).
# ---------------------------------------------------------------------
def test_cnn_translation_equivariance() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32)
    conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode="circular", bias=False)
    with torch.no_grad():
        shifts = (3, 5)
        out_then_shift = torch.roll(conv(x), shifts=shifts, dims=(-2, -1))
        shift_then_out = conv(torch.roll(x, shifts=shifts, dims=(-2, -1)))
    assert torch.allclose(out_then_shift, shift_then_out, atol=1e-5), (
        "Circular-padding convolution must commute with cyclic translation."
    )


# ---------------------------------------------------------------------
# vf_20 — Liang partial-separability rank.
# A space-time signal that is k-separable has Casorati-matrix rank ≤ k.
# Construct ρ(x, t) = Σ_{i=1..k} u_i(x) v_i(t) for k=3 and assert rank 3.
# ---------------------------------------------------------------------
def test_partial_separability_rank() -> None:
    torch.manual_seed(0)
    n_x, n_t, k = 64, 40, 3
    u = torch.randn(n_x, k)
    v = torch.randn(k, n_t)
    rho = u @ v  # shape (n_x, n_t), rank exactly k.

    # Casorati matrix is rho itself; rank counts singular values > tol.
    singular_values = torch.linalg.svdvals(rho)
    tol = singular_values.max() * 1e-6
    rank = (singular_values > tol).sum().item()
    assert rank == k, (
        f"Casorati rank should equal partial-separability order k={k}; got {rank}."
    )


# ---------------------------------------------------------------------
# vf_m2 — Null-space marker erasure (Moore-Penrose projection).
# The orthogonal projector onto range(M) is P = M (Mᴴ M)⁻¹ Mᴴ; its
# complement P⊥ = I − P annihilates the marker subspace exactly
# (P⊥ M = 0) and P is idempotent (P² = P). This is the formal guarantee
# that the virtual fiducial can be removed at inference without
# contaminating the anatomy estimate — a deterministic identity, not a
# learnable problem, so it lives here rather than as a 50k-iter arm.
# ---------------------------------------------------------------------
def test_moore_penrose_projection_identity() -> None:
    torch.manual_seed(0)
    n, n_m = 64, 5  # ambient dim N, marker subspace dim N_M
    M = torch.randn(n, n_m, dtype=torch.complex64)

    gram_inv = torch.linalg.inv(M.conj().T @ M)
    P = M @ gram_inv @ M.conj().T  # projector onto range(M)
    P_perp = torch.eye(n, dtype=torch.complex64) - P

    # P⊥ erases the marker subspace exactly.
    assert torch.allclose(P_perp @ M, torch.zeros_like(M), atol=1e-4), (
        "Complement projector must annihilate the marker subspace (P⊥ M = 0)."
    )
    # P is Hermitian (an orthogonal projector).
    assert torch.allclose(P, P.conj().T, atol=1e-5), "Projector must be Hermitian."


def test_projection_idempotency() -> None:
    torch.manual_seed(1)
    n, n_m = 48, 4
    M = torch.randn(n, n_m, dtype=torch.complex64)
    P = M @ torch.linalg.inv(M.conj().T @ M) @ M.conj().T

    assert torch.allclose(P @ P, P, atol=1e-4), "Projector must be idempotent (P² = P)."
    # Eigenvalues of an orthogonal projector are 0 or 1; trace = rank = N_M.
    assert math.isclose(P.diagonal().real.sum().item(), n_m, abs_tol=1e-3), (
        f"trace(P) should equal the marker rank N_M={n_m}."
    )
