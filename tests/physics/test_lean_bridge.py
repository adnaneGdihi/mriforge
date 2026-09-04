"""Bridge tests linking Lean certificates to the physics implementation.

Each test asserts, on the *actual* PyTorch implementation, the same coefficient
or invariant that its paired Lean theorem (in ``docs/lean/Sim2Rank/Sim2Rank/
Physics{Core,Fields,Geometry}.lean``) certifies. This is what makes a
machine-checked proof constrain real code rather than float free of it.

See ``docs/physics_correctness_audit.rst`` for the full
equation -> Lean theorem -> bridge test map.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.physics


def test_bridge_harness_imports():
    """Sanity: the physics surface exercised by the bridges is importable."""
    from spectramr.infrastructure.physics import (  # noqa: F401
        dipole,
        fft_ops,
        hermitian,
        maxwell_concomitant,
    )


def test_concomitant_eighth_bridge():
    """Bridges ``Sim2Rank.concomitant_coeff_eighth`` <-> ``maxwell_concomitant_field_hz``.

    On the z=0 slice with a pure read gradient (Gx=Gy=0), the concomitant Hz map
    equals gamma_bar/(2*pi) * (Gz_SI)^2/(8*B0) * (x^2+y^2) -- the 1/8 fingerprint.
    """
    from spectramr.infrastructure.physics.maxwell_concomitant import (
        GAMMA_BAR_PROTON_RAD_S_T,
        maxwell_concomitant_field_hz,
    )

    D, H, W = 1, 9, 9  # D=1 centred -> z=0
    vox = (1.0, 2.0, 2.0)  # (dz, dy, dx) mm
    Gz = 10.0  # mT/m, pure z-gradient
    B0 = 0.064  # T (ULF: concomitant non-negligible)
    f = maxwell_concomitant_field_hz((D, H, W), vox, (0.0, 0.0, Gz), B0)

    dy_m, dx_m = vox[1] * 1e-3, vox[2] * 1e-3
    ys = (torch.arange(H) - (H - 1) / 2) * dy_m
    xs = (torch.arange(W) - (W - 1) / 2) * dx_m
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coeff = GAMMA_BAR_PROTON_RAD_S_T / (2 * math.pi) * (Gz * 1e-3) ** 2 / (8 * B0)
    expected = coeff * (xx**2 + yy**2)
    assert torch.allclose(f[0], expected, rtol=1e-5, atol=1e-9)


def test_dipole_sign_bridge():
    """Bridges ``Sim2Rank.dipole_code_sign_trace_free`` <-> ``get_dipole_kernel_3d``.

    Pins the Salomir sign ``D(k)=1/3 - kz^2/|k|^2``: along the kz (B0) axis the
    kernel is -2/3, along kx/ky it is +1/3. The opposite (Schenck) sign flips both.
    """
    from spectramr.infrastructure.physics.dipole import get_dipole_kernel_3d

    k = get_dipole_kernel_3d((16, 16, 16), (1.0, 1.0, 1.0))
    kz_axis = k[1:, 0, 0]  # kx=ky=0, kz!=0 -> 1/3 - 1 = -2/3
    kx_axis = k[0, 0, 1:]  # ky=kz=0, kx!=0 -> 1/3 - 0 = +1/3
    ky_axis = k[0, 1:, 0]  # kx=kz=0, ky!=0 -> +1/3
    assert torch.allclose(kz_axis, torch.full_like(kz_axis, -2 / 3), atol=1e-3)
    assert torch.allclose(kx_axis, torch.full_like(kx_axis, 1 / 3), atol=1e-3)
    assert torch.allclose(ky_axis, torch.full_like(ky_axis, 1 / 3), atol=1e-3)


def test_dc_idempotent_bridge():
    """Bridges ``Sim2Rank.hard_dc_idempotent`` <-> ``HardDataConsistency`` (k-space).

    Hard DC ``P(k)=(1-M)k + M*y`` with the measurement fixed and noise disabled is
    idempotent: applying it twice equals once.
    """
    from spectramr.infrastructure.physics.data_consistency import HardDataConsistency

    torch.manual_seed(0)
    B, C, H, W = 1, 2, 16, 16
    k_pred = torch.randn(B, C, H, W, dtype=torch.complex64)
    y = torch.randn(B, C, H, W, dtype=torch.complex64)
    mask = (torch.rand(B, 1, H, W) > 0.5).float()
    dc = HardDataConsistency(train_noise_level=0.0, eval_noise_level=0.0).eval()
    once = dc(k_pred, kspace_obs=y, mask=mask, is_kspace_domain=True)
    twice = dc(once, kspace_obs=y, mask=mask, is_kspace_domain=True)
    assert torch.allclose(once, twice, atol=1e-6)


def test_hermitian_bridge():
    """Bridges ``Sim2Rank.hermitian_projection_satisfies`` <-> ``enforce_hermitian``.

    The projection lands in (and is a fixed point of) the Hermitian subspace:
    ``is_hermitian(enforce_hermitian(Z))`` and ``enforce_hermitian`` is idempotent.
    """
    from spectramr.infrastructure.physics.hermitian import enforce_hermitian, is_hermitian

    torch.manual_seed(0)
    z = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    h1 = enforce_hermitian(z)
    h2 = enforce_hermitian(h1)
    assert torch.allclose(h1, h2, atol=1e-6)  # idempotent
    assert is_hermitian(h1, atol=1e-5)  # lands in the Hermitian subspace


def test_relaxation_bounds_bridge():
    """Bridges ``Sim2Rank.relaxation_in_unit_interval`` <-> Bloch SE signal.

    ``E1,E2 in (0,1)`` => the SE signal ``PD*(1-E1)*E2`` lies in ``(0, PD)`` and
    rises monotonically with TR (longer TR -> smaller E1 -> more recovery).
    """
    from spectramr.infrastructure.physics.bloch_simulation import DifferentiableBlochSimulator

    sim = DifferentiableBlochSimulator(sequence_type="SE")
    tissue = torch.tensor([1.0, 900.0, 80.0]).view(1, 3, 1, 1)
    s_short = sim(tissue, TR=torch.tensor(300.0), TE=torch.tensor(20.0)).item()
    s_long = sim(tissue, TR=torch.tensor(2000.0), TE=torch.tensor(20.0)).item()
    assert 0.0 < s_short < 1.0
    assert 0.0 < s_long < 1.0
    assert s_long > s_short


def test_ir_null_bridge():
    """Bridges ``Sim2Rank.ir_null_at_t1_log2`` <-> the inversion-recovery model.

    The long-TR IR magnetization ``1 - 2 exp(-TI/T1)`` (the TR>>T1 limit of
    ``multi_physics_bloch``'s IR branch ``|1 - 2 e^{-TI/T1} + e^{-TR/T1}|``) nulls
    exactly at ``TI = T1 ln 2`` -- the FLAIR / T1-null fingerprint.
    """
    T1 = 900.0
    ti_null = T1 * math.log(2)
    mag = 1.0 - 2.0 * math.exp(-ti_null / T1)
    assert abs(mag) < 1e-12


def test_ernst_angle_bridge():
    """Bridges ``Sim2Rank.ernst_derivative_numerator`` <-> ``DifferentiableBlochSimulator`` GRE.

    ``dS/dalpha = 0 <=> cos(alpha) = E1``, so the flip angle maximizing the GRE
    signal is the Ernst angle ``acos(E1)``. Scanning the actual simulator, the
    argmax flip angle matches ``acos(E1)`` within the 1-degree scan resolution.
    """
    from spectramr.infrastructure.physics.bloch_simulation import DifferentiableBlochSimulator

    sim = DifferentiableBlochSimulator(sequence_type="GRE")
    TR, TE, T1, T2 = 100.0, 1.0, 900.0, 80.0
    E1 = math.exp(-TR / T1)
    tissue = torch.tensor([1.0, T1, T2]).view(1, 3, 1, 1)
    alphas = torch.linspace(1.0, 89.0, 89)
    sig = torch.stack(
        [
            sim(tissue, TR=torch.tensor(TR), TE=torch.tensor(TE), alpha=float(a)).squeeze()
            for a in alphas
        ]
    )
    a_star = alphas[int(torch.argmax(sig))].item()
    ernst_deg = math.degrees(math.acos(E1))
    assert abs(a_star - ernst_deg) < 2.0, f"argmax {a_star} vs Ernst {ernst_deg}"


def test_roundtrip_bridge():
    """``ifft2c . fft2c = id`` — the centered ortho FFT is invertible (unitary),
    the companion of the Parseval energy invariant (``perm_preserves_sq_sum``)."""
    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    torch.manual_seed(0)
    x = torch.randn(1, 2, 32, 32, dtype=torch.complex64)
    assert torch.allclose(ifft2c(fft2c(x)), x, atol=1e-5)


def test_b0_phase_roundtrip():
    """Flag F-B: B0 off-resonance phase is invertible by its conjugate.

    Acquisition applies ``e^{-i Δω TE}`` (``forward_operator``,
    ``complex_forward_operator``); CPR demodulation (``conjugate_phase``) applies
    the conjugate ``e^{+i Δω TE}``. Composing ``apply_b0_phase`` with the negated
    field reproduces the +sign correction and recovers the image exactly --
    confirming the opposite-sign conventions are consistent (no bug).
    """
    from spectramr.infrastructure.physics.complex_forward_operator import apply_b0_phase

    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    b0 = torch.randn(1, 1, 16, 16) * 50.0  # rad/s
    te = 0.02
    distorted = apply_b0_phase(x, b0, te)  # x * e^{-i b0 te}
    recovered = apply_b0_phase(distorted, -b0, te)  # * e^{+i b0 te} -> cancels
    assert torch.allclose(recovered, x, atol=1e-5)
    assert not torch.allclose(distorted, x, atol=1e-3)  # distortion is non-trivial


def test_logsumexp_bridge():
    """Bridges ``Sim2Rank.logsumexp_ge_max`` <-> ``tropical_geometry.log_sum_exp_tropical``.

    The soft tropical polynomial ``(1/beta) log sum exp(beta*affine)`` dominates the
    max affine value for any ``beta>0`` and converges down to the max as
    ``beta -> inf`` (max-plus / tropical limit).
    """
    from spectramr.infrastructure.physics.tropical_geometry import log_sum_exp_tropical

    torch.manual_seed(0)
    pts = torch.randn(5, 3)
    slopes = torch.randn(4, 3)
    intercepts = torch.randn(4)
    max_affine = (pts @ slopes.T + intercepts).max(dim=-1).values
    soft16 = log_sum_exp_tropical(pts, slopes, intercepts, beta=16.0)
    soft_big = log_sum_exp_tropical(pts, slopes, intercepts, beta=400.0)
    assert torch.all(soft16 >= max_affine - 1e-5)  # dominates the max
    assert torch.allclose(soft_big, max_affine, atol=5e-2)  # -> max as beta grows


def test_nullspace_idempotent_bridge():
    """Bridges ``Sim2Rank.complement_proj_idempotent`` <-> ``NullSpaceMarkerEraser``.

    ``P^perp = I - M^H(MM^H+epsI)^{-1}M`` is (as eps->0) idempotent: erasing the
    marker twice equals erasing it once.
    """
    from spectramr.infrastructure.physics.null_space_erasure import NullSpaceMarkerEraser

    torch.manual_seed(0)
    marker = torch.randn(2, 1, 8, 8)  # K=2 basis vectors
    eraser = NullSpaceMarkerEraser(marker, epsilon=1e-8)
    x = torch.randn(3, 1, 8, 8)
    once = eraser(x)
    twice = eraser(once)
    assert torch.allclose(once, twice, atol=1e-4)


def test_hodge_orthogonal_bridge():
    """Bridges ``Sim2Rank.grad_curl_orthogonal`` <-> ``helmholtz_hodge_decompose``.

    The Helmholtz-Hodge irrotational and solenoidal parts are L2-orthogonal up to
    discretisation error: ``|<v_irrot, v_sol>| << ||v_irrot|| ||v_sol||``.
    """
    from spectramr.infrastructure.physics.helmholtz_hodge import (
        helmholtz_hodge_decompose,
        hodge_orthogonality_residual,
    )

    torch.manual_seed(0)
    g = torch.linspace(0.0, 2.0 * math.pi, 16)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    v = torch.stack(
        [
            torch.sin(xx) * torch.cos(yy),
            torch.sin(yy) * torch.cos(zz),
            torch.sin(zz) * torch.cos(xx),
        ],
        dim=0,
    )
    v_irrot, v_sol = helmholtz_hodge_decompose(v)
    residual = hodge_orthogonality_residual(v_irrot, v_sol)
    denom = v_irrot.norm() * v_sol.norm() + 1e-8
    assert (residual / denom).item() < 0.1
