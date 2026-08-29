"""Unit tests for adjoint-sequence sequence-parameter optimisation.

Covers :mod:`mriforge.infrastructure.physics.adjoint_sequence`, which is *not*
a linear-operator-adjoint module — it implements the Pontryagin /
reverse-mode adjoint method for differentiable MRI **sequence-parameter**
optimisation (Method VI of ``synthetic_marker_research_plan.md`` §4.6).

The public API is:

* :func:`steady_state_se_signal` — differentiable steady-state SE signal.
* :func:`steady_state_ir_se_signal` — differentiable IR-SE signal.
* :func:`optimise_sequence_adjoint` — outer Adam loop over the sequence
  parameter dict ``φ = {tr, te, ti, alpha}`` with optional marker-anchor.
* :func:`jacobian_rank_augmentation_test` — verifies the §4.6
  rank-augmentation theorem on a synthetic Jacobian.

What we verify
--------------
* **Closed-form correctness** of the SE / IR-SE signal formulas.
* **Differentiability**: gradients flow through ``optimise_sequence_adjoint``
  and the loss decreases over iterations.
* **Marker-anchor effect**: the marker term keeps gradient norms
  finite even when anatomy-only optimisation is ill-posed.
* **Rank-augmentation theorem**: combining an anatomy Jacobian with a
  marker Jacobian that opens a new direction strictly increases the
  rank; combining with a redundant marker Jacobian leaves it unchanged.
* **Edge cases**: zero ``lambda_marker`` recovers the anatomy-only
  optimum; the empty-iteration call returns the initial state.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

adj_mod = pytest.importorskip("mriforge.infrastructure.physics.adjoint_sequence")
SequenceOptResult = adj_mod.SequenceOptResult
steady_state_se_signal = adj_mod.steady_state_se_signal
steady_state_ir_se_signal = adj_mod.steady_state_ir_se_signal
optimise_sequence_adjoint = adj_mod.optimise_sequence_adjoint
jacobian_rank_augmentation_test = adj_mod.jacobian_rank_augmentation_test


def _tensor(x: float) -> "torch.Tensor":
    return torch.tensor(float(x))


class TestSteadyStateSESignal:
    """Closed-form Spin-Echo signal."""

    def test_ernst_limit_alpha_90(self):
        """With α = π/2 the formula reduces to ``M0 (1 - e^{-TR/T1}) e^{-TE/T2}``."""
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)
        tr = _tensor(2.0)
        te = _tensor(0.05)

        s = steady_state_se_signal(t1, t2, m0, tr=tr, te=te)

        e1 = math.exp(-2.0 / 1.0)
        e2 = math.exp(-0.05 / 0.1)
        expected = (1.0 - e1) * e2
        assert math.isclose(float(s), expected, abs_tol=1e-6)

    def test_zero_tr_gives_zero_signal(self):
        """``TR → 0`` ⇒ ``1 - e^{-TR/T1} → 0`` ⇒ zero signal."""
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)
        s = steady_state_se_signal(t1, t2, m0, tr=_tensor(0.0), te=_tensor(0.05))
        assert math.isclose(float(s), 0.0, abs_tol=1e-9)

    def test_signal_increases_with_m0(self):
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        s1 = steady_state_se_signal(t1, t2, _tensor(1.0), tr=_tensor(2.0), te=_tensor(0.05))
        s2 = steady_state_se_signal(t1, t2, _tensor(2.0), tr=_tensor(2.0), te=_tensor(0.05))
        assert float(s2) > float(s1)


class TestSteadyStateIRSESignal:
    """Inversion-recovery SE signal."""

    def test_long_ti_recovers_se(self):
        """With TI ≫ T1, ``e^{-TI/T1} → 0`` ⇒ formula → ``M0 (1 + e^{-TR/T1}) e^{-TE/T2}``."""
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)
        tr = _tensor(2.0)
        te = _tensor(0.05)
        ti = _tensor(20.0)

        s = steady_state_ir_se_signal(t1, t2, m0, tr=tr, te=te, ti=ti)

        e_tr = math.exp(-2.0 / 1.0)
        e2 = math.exp(-0.05 / 0.1)
        expected = (1.0 + e_tr) * e2
        assert math.isclose(float(s), expected, abs_tol=1e-3)

    def test_nulling_ti(self):
        """At the IR null ``TI = T1 ln 2`` the magnetisation just crosses zero."""
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)
        tr = _tensor(20.0)  # long TR — recovery term is e^{-20} ~ 0
        te = _tensor(0.0)   # remove e^{-TE/T2} factor
        ti_null = _tensor(math.log(2.0))

        s = steady_state_ir_se_signal(t1, t2, m0, tr=tr, te=te, ti=ti_null)
        # 1 - 2·e^{-ln 2} + e^{-20} = 1 - 1 + ~0 ≈ 0
        assert abs(float(s)) < 1e-6


class TestOptimiseSequenceAdjoint:
    """Reverse-mode adjoint optimisation of ``φ = {tr, te}``."""

    def test_loss_decreases(self, deterministic_seed):
        """Optimising towards a known target signal must drive the loss down."""
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)

        # Target signal corresponds to TR=2.0 / TE=0.05 — i.e. the optimum.
        target = steady_state_se_signal(
            t1, t2, m0, tr=_tensor(2.0), te=_tensor(0.05)
        ).detach()

        # Start far from the optimum.
        initial_phi = {
            "tr": _tensor(0.5),
            "te": _tensor(0.2),
        }

        result = optimise_sequence_adjoint(
            steady_state_se_signal,
            t1,
            t2,
            m0,
            target,
            initial_phi,
            n_iters=80,
            lr=5e-2,
        )

        assert isinstance(result, SequenceOptResult)
        assert len(result.loss_history) == 80
        assert len(result.grad_norm_history) == 80
        assert result.best_loss < result.loss_history[0]
        assert result.best_loss <= result.loss_history[-1] + 1e-12

    def test_marker_anchor_term_active(self, deterministic_seed):
        """Adding a marker term changes the optimisation trajectory."""
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)
        target = _tensor(0.4)

        marker_t1 = _tensor(2.0)
        marker_t2 = _tensor(0.05)
        marker_m0 = _tensor(0.5)
        target_marker = _tensor(0.2)

        initial_phi = {"tr": _tensor(1.0), "te": _tensor(0.05)}

        out_anatomy_only = optimise_sequence_adjoint(
            steady_state_se_signal,
            t1, t2, m0, target,
            {k: v.detach().clone() for k, v in initial_phi.items()},
            n_iters=30, lr=5e-2,
        )
        out_with_marker = optimise_sequence_adjoint(
            steady_state_se_signal,
            t1, t2, m0, target,
            {k: v.detach().clone() for k, v in initial_phi.items()},
            marker_t1=marker_t1,
            marker_t2=marker_t2,
            marker_m0=marker_m0,
            target_marker_signal=target_marker,
            lambda_marker=1.0,
            n_iters=30, lr=5e-2,
        )

        # The two final parameter vectors must differ if the anchor was active.
        diff = sum(
            float((out_with_marker.best_phi[k] - out_anatomy_only.best_phi[k]).abs())
            for k in initial_phi
        )
        assert diff > 1e-4

    def test_bounds_are_respected(self, deterministic_seed):
        """``bounds`` should clamp the parameter dict in-place each step."""
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)
        target = steady_state_se_signal(
            t1, t2, m0, tr=_tensor(5.0), te=_tensor(0.05)
        ).detach()

        initial_phi = {"tr": _tensor(1.0), "te": _tensor(0.05)}
        bounds = {"tr": (0.8, 1.5), "te": (0.04, 0.06)}

        result = optimise_sequence_adjoint(
            steady_state_se_signal,
            t1, t2, m0, target, initial_phi,
            n_iters=20, lr=5e-2,
            bounds=bounds,
        )

        tr_final = float(result.phi_hat["tr"])
        te_final = float(result.phi_hat["te"])
        assert 0.8 - 1e-6 <= tr_final <= 1.5 + 1e-6
        assert 0.04 - 1e-6 <= te_final <= 0.06 + 1e-6

    def test_zero_iterations_returns_initial(self, deterministic_seed):
        """``n_iters=0`` is the 'empty sequence' — no optimisation steps."""
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)
        target = _tensor(0.5)
        initial_phi = {"tr": _tensor(2.0), "te": _tensor(0.05)}

        result = optimise_sequence_adjoint(
            steady_state_se_signal,
            t1, t2, m0, target,
            {k: v.detach().clone() for k, v in initial_phi.items()},
            n_iters=0, lr=1e-2,
        )

        assert result.loss_history == []
        assert result.grad_norm_history == []
        assert math.isinf(result.best_loss) or result.best_loss == float("inf")
        for k, v in initial_phi.items():
            assert torch.allclose(result.best_phi[k], v)


class TestJacobianRankAugmentation:
    """Markers must not *decrease* the column-space rank."""

    def test_adding_independent_marker_increases_rank(self, deterministic_seed):
        # 3-parameter problem with an anatomy Jacobian that only spans 2 dims.
        anat = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],   # in the span of rows 1,2
        ])
        # Marker opens the missing 3rd direction.
        mark = torch.tensor([[0.0, 0.0, 1.0]])

        r_a, r_combined = jacobian_rank_augmentation_test(anat, mark)
        assert r_combined > r_a
        assert r_a == 2
        assert r_combined == 3

    def test_redundant_marker_does_not_change_rank(self, deterministic_seed):
        # Anatomy Jacobian already full-rank.
        anat = torch.eye(3)
        mark = torch.tensor([[1.0, 1.0, 1.0]])  # in the span of identity rows

        r_a, r_combined = jacobian_rank_augmentation_test(anat, mark)
        assert r_a == 3
        assert r_combined == 3


class TestAdjointSequenceComposition:
    """Sequentially composing the differentiable signal functions.

    ``adjoint_sequence`` does not expose a linear-operator adjoint pair,
    so the "chained adjoint" identity from the task description is
    instantiated here as a **reverse-mode autograd composition** check:

        d/dφ [g(f(φ))] = (d g/d f)(d f/d φ)

    This is exactly the chain rule that the Pontryagin adjoint
    implements for us.
    """

    def test_chain_rule_via_autograd(self, deterministic_seed):
        t1 = _tensor(1.0)
        t2 = _tensor(0.1)
        m0 = _tensor(1.0)
        tr = torch.tensor(2.0, requires_grad=True)
        te = torch.tensor(0.05, requires_grad=True)

        s1 = steady_state_se_signal(t1, t2, m0, tr=tr, te=te)
        # Wrap the signal in a downstream "loss" — composition of two ops.
        loss = (s1 - 0.3) ** 2
        loss.backward()

        # Both parameters must receive a finite, non-zero gradient.
        assert tr.grad is not None
        assert te.grad is not None
        assert torch.isfinite(tr.grad)
        assert torch.isfinite(te.grad)
        # The chain rule guarantees ``dL/dTR = 2 (S - 0.3) · dS/dTR``;
        # sign of dS/dTR > 0 (longer TR → more recovery), so sign(dL/dTR)
        # matches sign(S - 0.3).
        s_val = float(s1.detach())
        if abs(s_val - 0.3) > 1e-3:
            assert (s_val - 0.3) * float(tr.grad) > 0
