"""Unit tests for SE3LieAlgebraMambaBlock (B-3 of the VF campaign).

Covers (per ``TODO/VF_CAMPAIGN_IMPLEMENTATION_PLAN.md`` Idea 3 test list):
irreducible-decomposition correctness, scalar-selectivity equivariance
(SO2 + full SO3 round-trip), bio-harmonic eigenvalue placement, gradient flow,
AMP-safety, deterministic seeding, integration with the framework Mamba
primitive, and no zero-rank degeneracy. Direct-import per the source<->test
pairing rule.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.models.blocks.se3_lie_algebra_mamba import (
    DEFAULT_BIO_HARMONIC_FREQS,
    SE3LieAlgebraMambaBlock,
)
from tests.utils.optional_backends import requires_cuda_for_mamba

TOL = 1e-4


def _rot_z(theta: float) -> torch.Tensor:
    r = torch.eye(3)
    c, s = math.cos(theta), math.sin(theta)
    r[0, 0], r[0, 1] = c, -s
    r[1, 0], r[1, 1] = s, c
    return r


def _random_so3(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.linalg.det(q) < 0:
        q = q.clone()
        q[:, -1] = -q[:, -1]
    return q


@pytest.fixture
def block() -> SE3LieAlgebraMambaBlock:
    torch.manual_seed(0)
    return SE3LieAlgebraMambaBlock(d_state=8, group="SE3", sample_rate_hz=100.0).eval()


@requires_cuda_for_mamba
def test_forward_shape_and_decompose(block: SE3LieAlgebraMambaBlock) -> None:
    xi = torch.randn(3, 16, 6)
    out = block(xi)
    assert out.shape == xi.shape
    omega, v = block.decompose(xi)
    assert omega.shape == (3, 16, 3)
    assert v.shape == (3, 16, 3)
    # decomposition is a pure split → concatenation recovers the input exactly.
    assert torch.equal(torch.cat([omega, v], dim=-1), xi)


@requires_cuda_for_mamba
def test_equivariance_so2_round_trip(block: SE3LieAlgebraMambaBlock) -> None:
    """||Block(Ad_g u) - Ad_g Block(u)|| < tol for an in-plane rotation."""
    torch.manual_seed(1)
    xi = torch.randn(2, 20, 6)
    out = block(xi)
    rot = _rot_z(0.6)
    lhs = block(block.adjoint_rotation(xi, rot))
    rhs = block.adjoint_rotation(out, rot)
    defect = torch.linalg.vector_norm(lhs - rhs).item()
    assert defect < TOL, f"SO2 equivariance defect {defect} >= {TOL}"


@requires_cuda_for_mamba
def test_equivariance_so3_full(block: SE3LieAlgebraMambaBlock) -> None:
    """The block is equivariant for FULL SO(3) rotations, not just SO(2)."""
    torch.manual_seed(2)
    xi = torch.randn(2, 18, 6)
    out = block(xi)
    rot = _random_so3(seed=7)
    lhs = block(block.adjoint_rotation(xi, rot))
    rhs = block.adjoint_rotation(out, rot)
    defect = torch.linalg.vector_norm(lhs - rhs).item()
    assert defect < TOL, f"SO3 equivariance defect {defect} >= {TOL}"


def test_bio_harmonic_eigenvalues_on_imaginary_axis() -> None:
    """A(0) eigenvalues sit at +/- i 2 pi f_k for the physiological freqs."""
    freqs = (0.3, 0.6, 1.2, 2.4)
    blk = SE3LieAlgebraMambaBlock(
        d_state=8, group="SE3", sample_rate_hz=100.0, bio_harmonic_freqs=freqs
    )
    ev = blk.bio_harmonic_eigenvalues()
    # purely imaginary
    assert torch.allclose(ev.real, torch.zeros_like(ev.real), atol=1e-6)
    got = sorted({round(abs(x) / (2 * math.pi), 4) for x in ev.imag.tolist()})
    assert got == sorted(freqs), f"eigenvalue freqs {got} != {sorted(freqs)}"


def _reference_oscillator_energy(
    block: SE3LieAlgebraMambaBlock, norms: torch.Tensor
) -> torch.Tensor:
    """The pre-2026-07-01 iterated-rotation loop, kept PERMANENTLY as the
    ground truth for the cumsum closed form (mirrors the python-reference
    pattern of ``test_continuous_sfc_mamba.py``)."""
    b, length, _ = norms.shape
    n_freqs = block._bio_disc_angles.shape[0]
    drive = norms.mean(dim=-1)

    cos_t = block._bio_eig_real.to(norms.device)
    sin_t = block._bio_eig_imag.to(norms.device)

    sr = torch.zeros(b, n_freqs, device=norms.device, dtype=drive.dtype)
    si = torch.zeros(b, n_freqs, device=norms.device, dtype=drive.dtype)
    out = torch.empty(b, length, n_freqs, device=norms.device, dtype=drive.dtype)
    for t in range(length):
        nsr = cos_t * sr - sin_t * si + drive[:, t : t + 1]
        nsi = sin_t * sr + cos_t * si
        sr, si = nsr, nsi
        out[:, t, :] = sr * sr + si * si
    return out / float(max(length, 1))


def test_oscillator_energy_closed_form_matches_iterated_rotation(
    block: SE3LieAlgebraMambaBlock,
) -> None:
    """PERF (2026-07-01): the unit-magnitude recurrence's cumsum closed form
    must match the old per-token rotation loop at realistic navigator lengths.

    Tolerances widen with L because the *loop* drifts: its fp32-rounded
    ``(cos θ, sin θ)`` eigenvalue has magnitude ≠ 1, and iterating it L times
    compounds that off-circle error (measured: the loop is ~10× FARTHER from
    the fp64 ground truth than the closed form at L=4096 — see the
    arbitration test below)."""
    for length, seed, rtol in ((16, 0, 1e-5), (256, 1, 5e-4), (512, 2, 2e-3)):
        g = torch.Generator().manual_seed(seed)
        norms = torch.rand(2, length, 2, generator=g) + 0.1

        out = block._oscillator_energy(norms)
        ref = _reference_oscillator_energy(block, norms)

        assert out.shape == ref.shape == (2, length, len(DEFAULT_BIO_HARMONIC_FREQS))
        assert torch.allclose(out, ref, rtol=rtol, atol=1e-6), (
            f"L={length}: max abs diff {(out - ref).abs().max().item():.3e}, "
            f"max rel diff "
            f"{((out - ref).abs() / ref.abs().clamp_min(1e-12)).max().item():.3e}"
        )


def test_oscillator_energy_closed_form_beats_loop_against_fp64_truth(
    block: SE3LieAlgebraMambaBlock,
) -> None:
    """Arbitration pin: at long L the closed form is MORE accurate than the
    old loop, not less. Ground truth = the fp64 iterated rotation with the
    eigenvalue recomputed exactly on the unit circle from the stored angle
    (independent construction; its own drift over L=4096 is ~1e-12). The old
    fp32 loop compounds an off-circle eigenvalue; the closed form is exactly
    unit-magnitude by construction. This is why the plan's original
    'abort if it diverges from the loop at large L' guard was inverted."""
    length = 4096
    g = torch.Generator().manual_seed(2)
    norms32 = torch.rand(2, length, 2, generator=g) + 0.1
    norms64 = norms32.double()

    # fp64 ground truth: iterated rotation with exact-unit-circle eigenvalue.
    angles64 = block._bio_disc_angles.double()
    drive64 = norms64.mean(dim=-1)
    cos64, sin64 = torch.cos(angles64), torch.sin(angles64)
    sr = torch.zeros(2, angles64.numel(), dtype=torch.float64)
    si = torch.zeros_like(sr)
    truth = torch.empty(2, length, angles64.numel(), dtype=torch.float64)
    for t in range(length):
        nsr = cos64 * sr - sin64 * si + drive64[:, t : t + 1]
        nsi = sin64 * sr + cos64 * si
        sr, si = nsr, nsi
        truth[:, t, :] = sr * sr + si * si
    truth = truth / float(length)

    err_closed = (block._oscillator_energy(norms32).double() - truth).abs().max()
    err_loop = (
        (_reference_oscillator_energy(block, norms32).double() - truth).abs().max()
    )

    assert err_closed < err_loop, (
        f"closed-form err {err_closed.item():.3e} should beat "
        f"loop err {err_loop.item():.3e} against the fp64 ground truth"
    )


def test_oscillator_energy_closed_form_gradients(
    block: SE3LieAlgebraMambaBlock,
) -> None:
    """Gradients through the closed form must match the iterated loop."""
    g = torch.Generator().manual_seed(3)
    base = torch.rand(1, 64, 2, generator=g) + 0.1
    n_new = base.clone().requires_grad_(True)
    n_ref = base.clone().requires_grad_(True)

    block._oscillator_energy(n_new).sum().backward()
    _reference_oscillator_energy(block, n_ref).sum().backward()

    assert torch.allclose(n_new.grad, n_ref.grad, rtol=1e-4, atol=1e-6)


@requires_cuda_for_mamba
def test_gradient_flow(block: SE3LieAlgebraMambaBlock) -> None:
    block.train()
    xi = torch.randn(2, 12, 6, requires_grad=True)
    loss = block(xi).pow(2).mean()
    loss.backward()
    assert xi.grad is not None and torch.isfinite(xi.grad).all()
    total = sum(
        p.grad.abs().sum().item() for p in block.parameters() if p.grad is not None
    )
    assert total > 0.0, "no parameter gradient flowed through the block"


@pytest.mark.gpu
def test_no_nan_under_amp() -> None:
    """fp16 autocast path must stay finite (fft / GRU run under disabled AMP)."""
    if not torch.cuda.is_available():
        pytest.skip("AMP path requires CUDA")
    blk = SE3LieAlgebraMambaBlock(d_state=8, group="SE3", sample_rate_hz=100.0).cuda()
    xi = torch.randn(2, 12, 6, device="cuda")
    with torch.amp.autocast("cuda"):
        out = blk(xi)
    assert torch.isfinite(out).all()


@requires_cuda_for_mamba
def test_deterministic_seeding() -> None:
    torch.manual_seed(123)
    b1 = SE3LieAlgebraMambaBlock(d_state=8, group="SE3", sample_rate_hz=100.0).eval()
    torch.manual_seed(123)
    b2 = SE3LieAlgebraMambaBlock(d_state=8, group="SE3", sample_rate_hz=100.0).eval()
    xi = torch.randn(2, 10, 6)
    assert torch.allclose(b1(xi), b2(xi), atol=1e-6)


def test_uses_framework_mamba_primitive(block: SE3LieAlgebraMambaBlock) -> None:
    """Integration: the scalar SSM is the framework MambaBlock primitive."""
    from mriforge.models.blocks.mamba_block import MambaBlock

    assert isinstance(block.scalar_ssm, MambaBlock)


@requires_cuda_for_mamba
def test_so3_group_ignores_translation_gate() -> None:
    """group='SO3' passes the translation block through unchanged (no gate)."""
    torch.manual_seed(3)
    blk = SE3LieAlgebraMambaBlock(d_state=8, group="SO3", sample_rate_hz=100.0).eval()
    xi = torch.randn(2, 14, 6)
    out = blk(xi)
    # translation slice (cols 3:6) is identity-passed for SO3.
    assert torch.allclose(out[..., 3:], xi[..., 3:], atol=1e-6)
    # rotation slice is gated (generally changed).
    assert not torch.allclose(out[..., :3], xi[..., :3], atol=1e-6)


@requires_cuda_for_mamba
def test_no_zero_rank_degeneracy(block: SE3LieAlgebraMambaBlock) -> None:
    """Output should not collapse to all-zeros (gate is strictly positive)."""
    torch.manual_seed(4)
    xi = torch.randn(2, 16, 6)
    out = block(xi)
    assert out.abs().sum().item() > 0.0
    # softplus gate is strictly > 0, so a nonzero direction stays nonzero.
    omega = xi[..., :3]
    omega_out = out[..., :3]
    nonzero = torch.linalg.vector_norm(omega, dim=-1) > 1e-3
    assert (torch.linalg.vector_norm(omega_out, dim=-1)[nonzero] > 0).all()


def test_unknown_group_raises() -> None:
    with pytest.raises(ValueError, match="unknown group"):
        SE3LieAlgebraMambaBlock(group="SU2", sample_rate_hz=100.0)  # type: ignore[arg-type]


def test_nyquist_violation_raises() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        SE3LieAlgebraMambaBlock(
            group="SE3", sample_rate_hz=4.0, bio_harmonic_freqs=(2.4,)
        )


def test_nonpositive_frequency_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        SE3LieAlgebraMambaBlock(
            group="SE3", sample_rate_hz=100.0, bio_harmonic_freqs=(0.0, 1.2)
        )


def test_default_freqs_constant() -> None:
    assert DEFAULT_BIO_HARMONIC_FREQS == (0.3, 0.6, 1.2, 2.4)


def test_bad_input_shape_raises(block: SE3LieAlgebraMambaBlock) -> None:
    with pytest.raises(ValueError, match=r"\[B, L, 6\]"):
        block(torch.randn(2, 16, 5))
