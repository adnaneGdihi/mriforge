r"""Unit tests for the NR *stability* metric battery (``nr_stability.py``).

CPU-only, fast, deterministic. Each metric is covered by:

* a **monotonicity** test — a graded sweep with the metric moving in the spec'd
  direction (Spearman sign correct);
* an **analytic anchor** — LRJS power iteration converges to a *known* top
  singular value of a controllable matmul reconstructor; FMED :math:`\equiv 0`
  for an identity (perfectly equivariant) reconstructor; PEDU rises with
  acceleration for a null-space-filling reconstructor;
* the **graceful-degradation path** — ``nan`` when the required
  :class:`~mriforge.core.metrics.context.MetricContext` fields are absent.

The synthetic reconstructors:

* ``lrjs`` — a real matmul ``R(y) = reshape(A @ vec(y))`` with an SVD-controlled
  top singular value; and an adjoint zero-fill (``sense_adjoint``).
* ``fmed`` — an identity reconstructor (defect exactly 0) and a hallucinating
  reconstructor (defect > 0).
* ``pedu`` — a generative reconstructor that fills the *null space* (the
  complement of the sampling mask) with fresh random noise each call, so the
  per-sample disagreement grows with the null-space size, i.e. with
  acceleration.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.core.metrics.context import MetricContext
from mriforge.core.metrics.nr_stability import (
    ForwardModelEquivarianceDefect,
    LocalReconstructionJacobianSpectralNorm,
    PosteriorEnsembleDisagreementUncertainty,
)
from mriforge.core.metrics.registry import MetricsRegistry
from mriforge.infrastructure.physics.fft_ops import (
    fft2c,
    ifft2c,
    sense_adjoint,
    sense_forward,
)
from mriforge.infrastructure.physics.sampling import MaskGenerator


def _spearman_sign(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (tiny, dependency-free)."""
    n = len(xs)
    rx = _ranks(xs)
    ry = _ranks(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "key,needs",
    [
        ("lrjs", ("reconstructor", "y_kspace")),
        ("fmed", ("reconstructor",)),
        ("pedu", ("reconstructor", "generator")),
    ],
)
def test_metrics_registered_with_contract(key: str, needs: tuple[str, ...]) -> None:
    assert MetricsRegistry.is_registered(key)
    assert MetricsRegistry.needs(key) == needs
    assert MetricsRegistry.requires_reference(key) is False


@pytest.mark.unit
def test_self_declared_directions_lower_is_better() -> None:
    for cls in (
        LocalReconstructionJacobianSpectralNorm,
        ForwardModelEquivarianceDefect,
        PosteriorEnsembleDisagreementUncertainty,
    ):
        assert cls().higher_is_better is False


# ---------------------------------------------------------------------------
# lrjs — Local Reconstruction-Jacobian Spectral Norm
# ---------------------------------------------------------------------------


def _make_matmul_recon(top_sv: float, n: int, seed: int = 0):
    """Linear ``R(y) = reshape(A @ vec(y))`` with controllable top singular value."""
    g = torch.Generator().manual_seed(seed)
    sv = torch.linspace(top_sv, 0.1, n)
    u, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    v, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    a = u @ torch.diag(sv) @ v.t()
    side = int(math.isqrt(n))

    def recon(y: torch.Tensor) -> torch.Tensor:
        return (a @ y.reshape(-1)).reshape(1, 1, side, side)

    return recon, float(sv[0].item())


@pytest.mark.unit
def test_lrjs_converges_to_known_top_singular_value() -> None:
    """Power iteration recovers the controlled top singular value."""
    side = 4
    n = side * side
    recon, true_sv = _make_matmul_recon(5.0, n)
    y = torch.randn(1, 1, side, side)
    metric = LocalReconstructionJacobianSpectralNorm(num_iters=40)
    val = metric(None, context=MetricContext(reconstructor=recon, y_kspace=y))
    assert math.isfinite(val)
    assert abs(val - true_sv) < 1e-2


@pytest.mark.unit
def test_lrjs_grows_with_conditioning() -> None:
    """LRJS increases monotonically with the reconstructor's top singular value."""
    side = 4
    n = side * side
    y = torch.randn(1, 1, side, side)
    metric = LocalReconstructionJacobianSpectralNorm(num_iters=40)
    top_svs = [1.0, 2.0, 5.0, 10.0, 20.0]
    vals = []
    for top in top_svs:
        recon, _ = _make_matmul_recon(top, n)
        vals.append(metric(None, context=MetricContext(reconstructor=recon, y_kspace=y)))
    # Strictly increasing (LRJS == top singular value here).
    assert _spearman_sign(top_svs, vals) >= 0.6
    for a, b in zip(vals, vals[1:]):
        assert b > a


@pytest.mark.unit
def test_lrjs_adjoint_zero_fill_is_well_posed() -> None:
    """A full-mask single-coil adjoint (== identity) has unit Jacobian norm."""
    side = 16
    clean = torch.randn(1, 1, side, side, dtype=torch.complex64)
    y = sense_forward(clean, smaps=None, mask=None)

    def adj(yk: torch.Tensor) -> torch.Tensor:
        return sense_adjoint(yk, smaps=None, mask=None)

    metric = LocalReconstructionJacobianSpectralNorm(num_iters=20)
    val = metric(clean, context=MetricContext(reconstructor=adj, y_kspace=y))
    # A^H A = I on the full mask -> spectral norm 1.
    assert math.isfinite(val)
    assert abs(val - 1.0) < 1e-2


@pytest.mark.unit
def test_lrjs_nan_without_context() -> None:
    assert math.isnan(LocalReconstructionJacobianSpectralNorm()(torch.randn(1, 1, 8, 8)))


@pytest.mark.unit
def test_lrjs_nan_for_failing_recon() -> None:
    """A reconstructor that raises under the transform degrades to nan, never propagates."""

    def bad(y: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("simulated non-differentiable reconstructor")

    y = torch.randn(1, 1, 6, 6)
    val = LocalReconstructionJacobianSpectralNorm()(
        None, context=MetricContext(reconstructor=bad, y_kspace=y)
    )
    assert math.isnan(val)


# ---------------------------------------------------------------------------
# fmed — Forward-Model Equivariance Defect
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fmed_identity_recon_is_zero() -> None:
    """An identity reconstructor commutes with every rotation -> defect 0."""

    def identity(y: torch.Tensor) -> torch.Tensor:
        return y

    img = torch.randn(1, 1, 24, 24, dtype=torch.complex64)
    metric = ForwardModelEquivarianceDefect(num_rotations=6, group="SO2", seed=0)
    val = metric(img, context=MetricContext(reconstructor=identity, y_kspace=img))
    assert val == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_fmed_rises_with_null_space_hallucination() -> None:
    """A reconstructor that adds a fixed bias breaks equivariance -> defect > 0,

    and the defect grows monotonically with the hallucination amplitude.
    """
    img = torch.randn(1, 1, 24, 24, dtype=torch.complex64)
    bias = torch.randn(1, 1, 24, 24, dtype=torch.complex64)
    metric = ForwardModelEquivarianceDefect(num_rotations=6, group="SO2", seed=0)

    amps = [0.0, 0.5, 1.0, 2.0, 4.0]
    vals = []
    for amp in amps:

        def halluc(y: torch.Tensor, a: float = amp) -> torch.Tensor:
            return y + a * bias

        vals.append(metric(img, context=MetricContext(reconstructor=halluc, y_kspace=img)))
    assert vals[0] == pytest.approx(0.0, abs=1e-6)
    assert vals[-1] > vals[0]
    assert _spearman_sign(amps, vals) >= 0.6


@pytest.mark.unit
def test_fmed_nan_without_reconstructor() -> None:
    assert math.isnan(ForwardModelEquivarianceDefect()(torch.randn(1, 1, 8, 8)))


@pytest.mark.unit
def test_fmed_nan_for_unknown_group_param() -> None:
    """An unsupported group passed via nr_params degrades to nan (no crash)."""

    def identity(y: torch.Tensor) -> torch.Tensor:
        return y

    img = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    ctx = MetricContext(
        reconstructor=identity, y_kspace=img, nr_params={"fmed_group": "SO3"}
    )
    assert math.isnan(ForwardModelEquivarianceDefect()(img, context=ctx))


# ---------------------------------------------------------------------------
# pedu — Posterior/Ensemble Disagreement Uncertainty
# ---------------------------------------------------------------------------


def _null_fill_recon(full_mask: torch.Tensor, amp: float = 0.3):
    """Generative reconstructor: DC on acquired k-space + random null-space fill.

    The stochastic source is the *global* RNG (which the PEDU MC path reseeds
    deterministically from ``ctx.generator``). The variance of the output over
    samples equals the variance of the null-space fill, which scales with the
    null-space size — i.e. with acceleration.
    """
    null = (full_mask == 0).to(torch.float32)

    def recon(yk: torch.Tensor) -> torch.Tensor:
        fill = (torch.randn_like(yk.real) + 1j * torch.randn_like(yk.real)) * amp
        return ifft2c(yk + fill * null)

    # Attach a dummy nn.Module-less marker so PEDU uses the multi-mask path only
    # when no mask is present; here we keep it a plain callable + supply a mask.
    return recon


@pytest.mark.unit
def test_pedu_rises_with_acceleration() -> None:
    """Epistemic disagreement grows in the under-determined regime."""
    side = 32
    # Low-energy structured image so the null-space fill (epistemic) dominates.
    clean = 0.05 * torch.randn(1, 1, side, side, dtype=torch.complex64)
    mg = MaskGenerator(seed=0)
    metric = PosteriorEnsembleDisagreementUncertainty(n_samples=12, submask_keep=1.0)

    accels = [2.0, 4.0, 8.0]
    vals = []
    for r in accels:
        mask = mg.generate_mask("random", (side, side), acceleration_factor=r).reshape(
            1, 1, side, side
        )
        yk = fft2c(clean) * mask
        recon = _null_fill_recon(mask)
        gen = torch.Generator().manual_seed(7)
        ctx = MetricContext(
            reconstructor=recon, y_kspace=yk, mask=mask, generator=gen
        )
        vals.append(metric(clean, context=ctx))
    for v in vals:
        assert math.isfinite(v) and v > 0.0
    # Monotone increase with acceleration.
    assert _spearman_sign(accels, vals) >= 0.6
    assert vals[-1] > vals[0]


@pytest.mark.unit
def test_pedu_mc_dropout_path() -> None:
    """A reconstructor exposing nn.Dropout drives the MC-dropout sampling path."""

    class DropRecon(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.drop = torch.nn.Dropout(0.5)

        def forward(self, yk: torch.Tensor) -> torch.Tensor:
            x = ifft2c(yk)
            return x * self.drop(torch.ones_like(x.real))

    clean = torch.randn(1, 1, 24, 24, dtype=torch.complex64)
    yk = fft2c(clean)
    recon = DropRecon()
    metric = PosteriorEnsembleDisagreementUncertainty(n_samples=6)
    gen = torch.Generator().manual_seed(0)
    ctx = MetricContext(reconstructor=recon, y_kspace=yk, generator=gen)
    val = metric(clean, context=ctx)
    assert math.isfinite(val) and val > 0.0
    assert metric._has_dropout(recon) is True


@pytest.mark.unit
def test_pedu_deterministic_given_generator() -> None:
    """Identical seeds yield bit-identical PEDU (RNG comes from ctx.generator)."""
    side = 16
    clean = torch.randn(1, 1, side, side, dtype=torch.complex64)
    mg = MaskGenerator(seed=0)
    mask = mg.generate_mask("random", (side, side), acceleration_factor=4.0).reshape(
        1, 1, side, side
    )
    yk = fft2c(clean) * mask

    def adj(y: torch.Tensor) -> torch.Tensor:
        return ifft2c(y)

    metric = PosteriorEnsembleDisagreementUncertainty(n_samples=6)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    v1 = metric(clean, context=MetricContext(reconstructor=adj, y_kspace=yk, mask=mask, generator=g1))
    v2 = metric(clean, context=MetricContext(reconstructor=adj, y_kspace=yk, mask=mask, generator=g2))
    assert v1 == pytest.approx(v2, abs=1e-12)


@pytest.mark.unit
def test_pedu_nan_without_context() -> None:
    assert math.isnan(PosteriorEnsembleDisagreementUncertainty()(torch.randn(1, 1, 8, 8)))


@pytest.mark.unit
def test_pedu_nan_without_stochastic_source() -> None:
    """No dropout and no mask -> no stochastic source -> nan (never raises)."""

    def lin(y: torch.Tensor) -> torch.Tensor:
        return y

    yk = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    gen = torch.Generator().manual_seed(0)
    ctx = MetricContext(reconstructor=lin, y_kspace=yk, generator=gen)
    assert math.isnan(PosteriorEnsembleDisagreementUncertainty()(torch.randn(1, 1, 8, 8), context=ctx))
