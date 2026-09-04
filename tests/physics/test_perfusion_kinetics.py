"""Tests for the perfusion tracer-kinetic forward models."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.signal_models.perfusion_kinetics import (
    extended_tofts_forward,
    gamma_variate,
    parker_population_aif,
    spgr_signal_to_concentration,
)


def test_parker_aif_nonneg_and_peaks_early() -> None:
    t = torch.linspace(0, 300, 301)  # 5 min
    aif = parker_population_aif(t)
    assert aif.shape == t.shape
    assert torch.all(aif >= -1e-6)
    # First-pass peak is in the first ~30 s.
    assert aif.argmax().item() < 60


def test_tofts_forward_shape_and_monotone_in_ktrans() -> None:
    t = torch.linspace(0, 300, 151)
    aif = parker_population_aif(t)
    low = extended_tofts_forward(t, aif, ktrans=0.05, ve=0.2, vp=0.02)
    high = extended_tofts_forward(t, aif, ktrans=0.30, ve=0.2, vp=0.02)
    assert low.shape == t.shape
    # Larger Ktrans -> larger total uptake (area under curve).
    assert high.sum() > low.sum()


def test_tofts_residual_zero_at_true_params() -> None:
    t = torch.linspace(0, 300, 151)
    aif = parker_population_aif(t)
    ktrans, ve, vp = 0.12, 0.25, 0.03
    curve = extended_tofts_forward(t, aif, ktrans, ve, vp)
    refit = extended_tofts_forward(
        t, aif, torch.tensor(ktrans), torch.tensor(ve), torch.tensor(vp)
    )
    assert torch.allclose(curve, refit, atol=1e-5)


def test_gamma_variate_is_causal() -> None:
    t = torch.linspace(0, 60, 121)
    c = gamma_variate(t, amplitude=1.0, t0_s=10.0, alpha=3.0, beta_s=2.0)
    # Zero before bolus arrival, positive after.
    assert torch.all(c[t <= 10.0] == 0.0)
    assert c[t > 10.0].max() > 0.0


def test_spgr_signal_to_concentration_monotone() -> None:
    s0 = torch.tensor(100.0)
    low = spgr_signal_to_concentration(torch.tensor(110.0), s0, t10_s=1.4)
    high = spgr_signal_to_concentration(torch.tensor(150.0), s0, t10_s=1.4)
    assert high > low >= 0.0


def test_tofts_bad_shape_raises() -> None:
    t = torch.linspace(0, 10, 11)
    with pytest.raises(ValueError):
        extended_tofts_forward(t, torch.zeros(5), 0.1, 0.2, 0.0)


# ---------------------------------------------------------------------------
# The vectorisation of extended_tofts_forward (2026-07-16).
# ---------------------------------------------------------------------------


def _reference_trapz_loop(
    t_s: torch.Tensor,
    aif: torch.Tensor,
    ktrans: torch.Tensor,
    ve: torch.Tensor,
    vp: torch.Tensor,
) -> torch.Tensor:
    """The per-timestep trapezoid loop the vectorised form replaced.

    Kept here verbatim as an INDEPENDENT reference. Deleting it along with the
    old implementation would have left only a self-consistency check — note that
    ``test_tofts_residual_zero_at_true_params`` above compares the function to
    itself and so cannot detect a wrong convolution at all.
    """
    dt = (t_s[1] - t_s[0]).clamp_min(1e-6)
    kep = ktrans / ve.clamp_min(1e-6)
    tau = t_s - t_s[0]
    kernel = torch.exp(-kep.unsqueeze(-1) * tau)
    conv = torch.zeros(
        (*kernel.shape[:-1], t_s.shape[0]), dtype=t_s.dtype, device=t_s.device
    )
    for i in range(t_s.shape[0]):
        w = kernel[..., : i + 1].flip(-1)
        conv[..., i] = torch.trapz(aif[..., : i + 1] * w, dx=float(dt), dim=-1)
    return ktrans.unsqueeze(-1) * conv + vp.unsqueeze(-1) * aif


@pytest.mark.parametrize("shape", [(), (4,), (2, 3, 5)])
def test_vectorised_tofts_matches_the_reference_loop(shape: tuple[int, ...]) -> None:
    """Exact identity, not an approximation — scalar, batched and per-voxel.

    The trapezoid weights (1/2 at both endpoints) are why a plain causal sum
    would be WRONG here; the endpoint correction is what makes this exact.
    float64 + a 1e-10 tolerance so an algebraic slip cannot hide in fp32 noise.
    """
    torch.manual_seed(0)
    t = torch.linspace(0, 300, 60, dtype=torch.float64)
    aif = parker_population_aif(t)
    ktrans = torch.rand(shape, dtype=torch.float64) * 0.5 + 0.01
    ve = torch.rand(shape, dtype=torch.float64) * 0.4 + 0.05
    vp = torch.rand(shape, dtype=torch.float64) * 0.1

    torch.testing.assert_close(
        extended_tofts_forward(t, aif, ktrans, ve, vp),
        _reference_trapz_loop(t, aif, ktrans, ve, vp),
        rtol=1e-10,
        atol=1e-12,
    )


def test_vectorised_tofts_gradients_match_the_reference_loop() -> None:
    """The refit objective is gradient-driven, so the gradients must match too."""
    t = torch.linspace(0, 300, 40, dtype=torch.float64)
    aif = parker_population_aif(t)

    grads = []
    for fn in (extended_tofts_forward, _reference_trapz_loop):
        ktrans = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        ve = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        vp = torch.tensor(0.05, dtype=torch.float64)
        fn(t, aif, ktrans, ve, vp).sum().backward()
        grads.append((ktrans.grad.clone(), ve.grad.clone()))

    torch.testing.assert_close(grads[0][0], grads[1][0], rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(grads[0][1], grads[1][1], rtol=1e-9, atol=1e-11)


def test_tofts_forward_issues_no_host_sync() -> None:
    """Regression: ``dt.item()`` sat INSIDE a ``for i in range(T)`` loop.

    ToftsResidualLoss calls straight into this every training step, so at T=60
    that was 60 host syncs + 60 kernel launches per loss evaluation, with
    autograd retaining every slice — CLAUDE.md #9 (no ``.item()`` in the loop).

    Asserted on the ast, not on the source text: the source legitimately mentions
    ``.item()`` in a comment and a docstring explaining this very bug, so a
    substring check would fail on the fix's own documentation.
    """
    import ast
    import inspect

    from spectramr.infrastructure.physics.signal_models import perfusion_kinetics

    tree = ast.parse(inspect.getsource(perfusion_kinetics.extended_tofts_forward))
    called_attrs = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "item" not in called_attrs, "extended_tofts_forward re-introduced a host sync"
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.For)], (
        "extended_tofts_forward re-introduced a Python loop over the time axis"
    )


def test_tofts_rejects_a_non_arterial_batched_aif() -> None:
    """A per-voxel AIF is not physical and must raise, not silently broadcast."""
    t = torch.linspace(0, 300, 20)
    batched = parker_population_aif(t).expand(4, 20)
    with pytest.raises(ValueError, match="one curve per acquisition"):
        extended_tofts_forward(t, batched, 0.1, 0.2, 0.0)
