"""Unit tests for non-conformity score functions.

Targets ``spectramr.infrastructure.calibration.scores``. Part of the parallel
test-coverage push (Unit 14). These three score functions
(:func:`absolute_residual`, :func:`quantile_regression`,
:func:`conformalised_quantile_regression`) are pure / stateless, so the
properties we check are mathematical:

- **Permutation invariance.** The *set* of scores produced by a score
  function is invariant under any joint permutation of its inputs (it
  is element-wise).
- **Non-negativity** for ``absolute_residual`` (documented).
- **Identity case.** When prediction equals target, the score is zero.
- **Shape contract.** Loud failure on shape mismatch.
- **CQR alias.** ``conformalised_quantile_regression`` is numerically
  identical to ``quantile_regression`` (Romano-Patterson-Candès 2019).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.calibration.scores import (
    absolute_residual,
    conformalised_quantile_regression,
    quantile_regression,
)


# ---------------------------------------------------------------------------
# absolute_residual
# ---------------------------------------------------------------------------


def test_absolute_residual_basic_values() -> None:
    """``|pred - target|`` element-wise."""
    pred = torch.tensor([1.0, 2.0, 3.0, -4.0])
    target = torch.tensor([0.0, 2.5, 4.0, -1.0])
    out = absolute_residual(pred, target)
    expected = torch.tensor([1.0, 0.5, 1.0, 3.0])
    assert torch.allclose(out, expected)


def test_absolute_residual_is_non_negative() -> None:
    """The absolute residual is always ``≥ 0`` (documented property)."""
    torch.manual_seed(0)
    pred = torch.randn(2048)
    target = torch.randn(2048)
    out = absolute_residual(pred, target)
    assert (out >= 0).all()


def test_absolute_residual_identity_is_zero() -> None:
    """``score(x, x) = 0`` for any tensor ``x``."""
    x = torch.tensor([-3.5, 0.0, 1.0, 17.25])
    out = absolute_residual(x, x)
    assert torch.equal(out, torch.zeros_like(x))


def test_absolute_residual_preserves_shape() -> None:
    """Output shape equals input shape."""
    pred = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    out = absolute_residual(pred, target)
    assert out.shape == pred.shape


def test_absolute_residual_permutation_invariance() -> None:
    """Joint permutation of ``pred`` and ``target`` leaves the score-set unchanged."""
    torch.manual_seed(0)
    pred = torch.randn(256)
    target = torch.randn(256)
    perm = torch.randperm(pred.numel())

    s1 = absolute_residual(pred, target)
    s2 = absolute_residual(pred[perm], target[perm])

    # Same multiset of values (sorted comparison).
    assert torch.allclose(s1.sort().values, s2.sort().values)


def test_absolute_residual_shape_mismatch_raises() -> None:
    """A shape mismatch is a contract violation, not a broadcast."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        absolute_residual(torch.zeros(4), torch.zeros(8))


def test_absolute_residual_is_symmetric_in_arguments() -> None:
    """``score(a, b) == score(b, a)`` — absolute value is symmetric."""
    torch.manual_seed(1)
    a = torch.randn(64)
    b = torch.randn(64)
    assert torch.allclose(absolute_residual(a, b), absolute_residual(b, a))


# ---------------------------------------------------------------------------
# quantile_regression
# ---------------------------------------------------------------------------


def test_quantile_regression_inside_band_is_non_positive() -> None:
    """When the target lies inside ``[lo, hi]`` the score is ``≤ 0``."""
    lo = torch.tensor([-1.0, 0.0, 2.0])
    hi = torch.tensor([1.0, 5.0, 4.0])
    target = torch.tensor([0.0, 3.0, 3.0])  # all strictly inside
    out = quantile_regression(lo, hi, target)
    assert (out <= 0).all()


def test_quantile_regression_outside_band_is_positive() -> None:
    """Target below ``lo`` or above ``hi`` gives a strictly positive score."""
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([1.0, 1.0])
    target = torch.tensor([-1.0, 2.0])  # one below, one above
    out = quantile_regression(lo, hi, target)
    assert (out > 0).all()


def test_quantile_regression_on_boundary_is_zero() -> None:
    """At ``y == lo`` or ``y == hi`` the score is exactly 0."""
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([1.0, 1.0])
    target_low = torch.tensor([0.0, 0.0])
    target_high = torch.tensor([1.0, 1.0])
    assert torch.equal(
        quantile_regression(lo, hi, target_low), torch.zeros_like(lo)
    )
    assert torch.equal(
        quantile_regression(lo, hi, target_high), torch.zeros_like(hi)
    )


def test_quantile_regression_permutation_invariance() -> None:
    """Score-set is invariant under joint permutation of the three inputs."""
    torch.manual_seed(0)
    lo = torch.randn(128)
    hi = lo + torch.rand(128) + 0.01  # ensure hi > lo
    target = torch.randn(128)
    perm = torch.randperm(lo.numel())

    s1 = quantile_regression(lo, hi, target)
    s2 = quantile_regression(lo[perm], hi[perm], target[perm])
    assert torch.allclose(s1.sort().values, s2.sort().values)


def test_quantile_regression_shape_mismatch_raises() -> None:
    """Any shape disagreement among the three inputs raises."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        quantile_regression(torch.zeros(4), torch.zeros(8), torch.zeros(4))
    with pytest.raises(ValueError, match="Shape mismatch"):
        quantile_regression(torch.zeros(4), torch.zeros(4), torch.zeros(8))


def test_quantile_regression_swapped_quantiles_raises() -> None:
    """``pred_lower > pred_upper`` anywhere is a fatal contract violation."""
    lo = torch.tensor([2.0, 0.0])
    hi = torch.tensor([1.0, 1.0])  # first element swapped
    target = torch.zeros(2)
    with pytest.raises(ValueError, match="swapped|pred_lower"):
        quantile_regression(lo, hi, target)


# ---------------------------------------------------------------------------
# conformalised_quantile_regression (alias)
# ---------------------------------------------------------------------------


def test_cqr_is_identical_to_quantile_regression() -> None:
    """``conformalised_quantile_regression`` mirrors ``quantile_regression`` numerically."""
    torch.manual_seed(2)
    lo = torch.randn(32)
    hi = lo + 1.0
    target = torch.randn(32)
    assert torch.equal(
        conformalised_quantile_regression(lo, hi, target),
        quantile_regression(lo, hi, target),
    )


def test_cqr_propagates_swapped_quantile_error() -> None:
    """The alias must keep the same contract checks (no silent fallback)."""
    lo = torch.tensor([1.0])
    hi = torch.tensor([0.0])
    target = torch.zeros(1)
    with pytest.raises(ValueError):
        conformalised_quantile_regression(lo, hi, target)


# ---------------------------------------------------------------------------
# Sanity checks combining scores into a calibration set
# ---------------------------------------------------------------------------


def test_absolute_residual_used_as_calibration_score_yields_finite_quantile() -> None:
    """End-to-end: the score plugs into ``torch.quantile`` without NaN/Inf.

    A regression test for the score's interaction with the calibrator's
    finite-sample correction.
    """
    torch.manual_seed(0)
    pred = torch.randn(1024)
    target = torch.randn(1024)
    scores = absolute_residual(pred, target).flatten().double()
    q = torch.quantile(scores, 0.9)
    assert torch.isfinite(q)
    assert q.item() >= 0.0


# ---------------------------------------------------------------------------
# physics_residual (PR-CC: physics-residual non-conformity score)
#
# The score re-renders the OBSERVED contrast from an estimated tissue-parameter
# map through the physics single-source-of-truth (the Bloch forward) and returns
# the per-voxel |render - observed| residual. Unlike an image-distribution score
# it is anchored to a field-invariant forward model, so its validity needs only
# residual exchangeability, not image-distribution exchangeability.
# ---------------------------------------------------------------------------


def _spin_echo_params() -> torch.Tensor:
    """A ``[1, 3, 2, 2]`` ``(rho, T1_ms, T2_ms)`` map with per-voxel contrast."""
    rho = torch.tensor([[[[1.0, 0.8], [0.6, 0.9]]]])
    t1 = torch.tensor([[[[800.0, 1200.0], [600.0, 1000.0]]]])
    t2 = torch.tensor([[[[80.0, 120.0], [60.0, 100.0]]]])
    return torch.cat([rho, t1, t2], dim=1)


_SE_ACQ = {"TR": 3000.0, "TE": 90.0, "TI": 0.0}


def _render_spin_echo(params: torch.Tensor) -> torch.Tensor:
    """Render the SE contrast directly via the Bloch SSOT (int enum 0 = SE)."""
    from spectramr.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )

    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False)
    tissue = {"rho": params[:, 0:1], "t1": params[:, 1:2], "t2": params[:, 2:3]}
    return layer(tissue, {"TR": 3000.0, "TE": 90.0, "TI": 0.0, "contrast_type": 0})


def test_physics_residual_zero_when_target_is_render_of_params() -> None:
    """When the target IS the Bloch render of the params, the residual is ~0."""
    from spectramr.infrastructure.calibration.scores import physics_residual

    params = _spin_echo_params()
    target = _render_spin_echo(params)
    s = physics_residual(params, target, acquisition=_SE_ACQ, contrast_type="spin_echo")
    assert s.shape == target.shape
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)


def test_physics_residual_positive_for_wrong_params() -> None:
    """A wrong parameter map renders an inconsistent contrast: residual > 0."""
    from spectramr.infrastructure.calibration.scores import physics_residual

    params = _spin_echo_params()
    target = _render_spin_echo(params)
    bad = params.clone()
    bad[:, 2] = bad[:, 2] * 3.0  # triple T2 -> different normalized contrast
    s = physics_residual(bad, target, acquisition=_SE_ACQ, contrast_type="spin_echo")
    assert s.max().item() > 1e-2


def test_physics_residual_requires_acquisition() -> None:
    """No silent fallback: a physics residual without acquisition params raises."""
    from spectramr.infrastructure.calibration.scores import physics_residual

    params = _spin_echo_params()
    target = _render_spin_echo(params)
    with pytest.raises(ValueError, match="acquisition"):
        physics_residual(params, target, acquisition=None, contrast_type="spin_echo")


def test_physics_residual_rejects_unknown_contrast() -> None:
    """Unknown contrast labels raise instead of degrading to a default (pitfall #9)."""
    from spectramr.infrastructure.calibration.scores import physics_residual

    params = _spin_echo_params()
    target = _render_spin_echo(params)
    with pytest.raises(ValueError, match="contrast"):
        physics_residual(
            params, target, acquisition=_SE_ACQ, contrast_type="hyperbolic_secant"
        )


def test_physics_residual_rejects_non_three_channel_prediction() -> None:
    """The prediction must carry the (rho, T1, T2) channels to be renderable."""
    from spectramr.infrastructure.calibration.scores import physics_residual

    params = _spin_echo_params()
    target = _render_spin_echo(params)
    with pytest.raises(ValueError, match="3"):
        physics_residual(target, target, acquisition=_SE_ACQ, contrast_type="spin_echo")
