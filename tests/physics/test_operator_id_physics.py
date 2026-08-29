"""Physics tests for the operator-ID Magnus/BCH machinery (Proposal 1).

Analytical baselines:

(a) per-generator adjoint identity by the dot-product test;
(b) ``torch.autograd.gradcheck`` on the Magnus action at small N;
(c) matrix-free ``exp(Ω)x`` versus dense ``torch.matrix_exp`` at small N;
(d) BCH partial sum versus sequential composition exp(s_K L_K)…exp(s_1 L_1)
    (metamorphic: order-2 truncation is markedly closer than order-1);
(e) determinism under a fixed seed;
(f) the s ≡ 0 consistency check (operator reduces to the identity).
"""

# ruff: noqa: E402  (pytest.importorskip must precede torch-dependent imports)

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.physics.degradation_generators import (
    build_generators,
    list_generator_modes,
)
from mriforge.infrastructure.physics.magnus_exponential import (
    assemble_omega,
    dense_matrix,
    expm_multiply,
)

DET_MODES = ["motion_rigid", "bias_field_multiplicative", "b0_phase", "gibbs_truncation"]
DT = torch.complex128


def _gens(h=4, w=4, modes=None):
    modes = modes or DET_MODES
    gens, det, _ = build_generators(modes, h, w, dtype=DT)
    return gens, det


def test_all_catalog_modes_listed():
    modes = set(list_generator_modes())
    for m in [*DET_MODES, "rician_noise"]:
        assert m in modes


@pytest.mark.parametrize("mode", DET_MODES)
def test_generator_adjoint_identity(mode):
    """⟨v, L u⟩ == ⟨Lᴴ v, u⟩ to ~1e-5 (the dot-product test)."""
    c, h, w = 1, 4, 4
    gens, _ = _gens(h, w, [mode])
    g = gens[0]
    torch.manual_seed(0)
    u = torch.randn(2, c, h, w, dtype=DT)
    v = torch.randn(2, c, h, w, dtype=DT)
    lhs = (v.conj() * g(u)).sum(dim=(1, 2, 3))
    rhs = (g.adjoint(v).conj() * u).sum(dim=(1, 2, 3))
    assert (lhs - rhs).abs().max() < 1e-5


def test_matrixfree_exp_matches_dense():
    c, h, w = 1, 4, 4
    n = c * h * w
    gens, _ = _gens(h, w)
    torch.manual_seed(1)
    x = torch.randn(1, c, h, w, dtype=DT)
    s = torch.tensor([0.3, 0.2, 0.1, 0.15])
    om = assemble_omega(gens, s, order=2)
    dense = (torch.matrix_exp(dense_matrix(om, (c, h, w), dtype=DT)) @ x.reshape(n)).reshape(
        1, c, h, w
    )
    kry = expm_multiply(om, x, krylov_dim=n, scaling_squaring=False)
    assert (dense - kry).abs().max() < 1e-6


def test_identity_at_zero_severity():
    c, h, w = 1, 4, 4
    n = c * h * w
    gens, _ = _gens(h, w)
    x = torch.randn(1, c, h, w, dtype=DT)
    om = assemble_omega(gens, torch.zeros(len(gens)), order=2)
    y = expm_multiply(om, x, krylov_dim=n, scaling_squaring=False)
    assert (y - x).abs().max() < 1e-10


def test_bch_vs_sequential_composition():
    """Order-2 BCH truncation is markedly closer to the product than order-1."""
    c, h, w = 1, 4, 4
    n = c * h * w
    gens, _ = _gens(h, w)
    x = torch.randn(1, c, h, w, dtype=DT)
    s = torch.tensor([0.05, 0.04, 0.03, 0.02])

    # Sequential product e^{s1 L1} e^{s2 L2} … e^{sK LK} acting on x.
    y_seq = x
    for g, sk in list(zip(gens, s.tolist(), strict=False))[::-1]:
        y_seq = expm_multiply(g.scaled(sk), y_seq, krylov_dim=n, scaling_squaring=False)

    y_o1 = expm_multiply(assemble_omega(gens, s, order=1), x, krylov_dim=n, scaling_squaring=False)
    y_o2 = expm_multiply(assemble_omega(gens, s, order=2), x, krylov_dim=n, scaling_squaring=False)
    err1 = (y_o1 - y_seq).abs().max()
    err2 = (y_o2 - y_seq).abs().max()
    assert err2 < err1
    assert err2 < 1e-3


def test_gradcheck_magnus_action():
    """Autograd gradients of exp(Ω(s))x w.r.t. severities are correct."""
    c, h, w = 1, 3, 3
    n = c * h * w
    # Non-FFT generators keep double-precision exactness for gradcheck.
    gens, _ = _gens(h, w, ["motion_rigid", "bias_field_multiplicative", "b0_phase"])
    x = torch.randn(1, c, h, w, dtype=DT)

    def f(s):
        om = assemble_omega(gens, s, order=2)
        return expm_multiply(om, x, krylov_dim=n, scaling_squaring=False)

    s = torch.rand(3, dtype=torch.float64, requires_grad=True) * 0.3
    assert torch.autograd.gradcheck(f, (s,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_determinism_under_fixed_seed():
    c, h, w = 1, 4, 4
    n = c * h * w
    gens, _ = _gens(h, w)
    s = torch.tensor([0.2, 0.1, 0.15, 0.05])
    om = assemble_omega(gens, s, order=2)

    torch.manual_seed(7)
    x = torch.randn(1, c, h, w, dtype=DT)
    y1 = expm_multiply(om, x, krylov_dim=n, scaling_squaring=False)
    y2 = expm_multiply(om, x, krylov_dim=n, scaling_squaring=False)
    assert torch.equal(y1, y2)


def test_per_sample_severities_broadcast():
    """Per-sample severities [B, K] apply the operator independently per sample."""
    c, h, w = 1, 4, 4
    n = c * h * w
    gens, _ = _gens(h, w)
    x = torch.randn(3, c, h, w, dtype=DT)
    s = torch.rand(3, len(gens)) * 0.2
    om = assemble_omega(gens, s, order=2)
    y = expm_multiply(om, x, krylov_dim=n, scaling_squaring=False)
    # Sample 0 must equal the single-sample operator with s[0].
    om0 = assemble_omega(gens, s[0], order=2)
    y0 = expm_multiply(om0, x[:1], krylov_dim=n, scaling_squaring=False)
    assert (y[:1] - y0).abs().max() < 1e-6
