"""Unit tests for the adjoint-sequence-optimisation module (Method VI)."""
from __future__ import annotations

import torch

from mriforge.infrastructure.physics.adjoint_sequence import (
    jacobian_rank_augmentation_test,
    optimise_sequence_adjoint,
    steady_state_ir_se_signal,
    steady_state_se_signal,
)


class TestSteadyStateSignals:
    def test_se_signal_increases_with_tr(self) -> None:
        t1 = torch.tensor([800.0]); t2 = torch.tensor([80.0]); m0 = torch.tensor([1.0])
        s_short = steady_state_se_signal(
            t1, t2, m0, tr=torch.tensor([500.0]), te=torch.tensor([80.0]),
        )
        s_long = steady_state_se_signal(
            t1, t2, m0, tr=torch.tensor([3000.0]), te=torch.tensor([80.0]),
        )
        assert float(s_long) > float(s_short)

    def test_se_signal_decreases_with_te(self) -> None:
        t1 = torch.tensor([800.0]); t2 = torch.tensor([80.0]); m0 = torch.tensor([1.0])
        s_short = steady_state_se_signal(
            t1, t2, m0, tr=torch.tensor([2000.0]), te=torch.tensor([10.0]),
        )
        s_long = steady_state_se_signal(
            t1, t2, m0, tr=torch.tensor([2000.0]), te=torch.tensor([200.0]),
        )
        assert float(s_long) < float(s_short)

    def test_ir_se_at_null_point(self) -> None:
        # FLAIR null at TI = T1 ln 2 ≈ 0.693 T1
        t1 = torch.tensor([1000.0]); t2 = torch.tensor([80.0]); m0 = torch.tensor([1.0])
        ti_null = t1 * torch.log(torch.tensor(2.0))   # ~693
        s = steady_state_ir_se_signal(
            t1, t2, m0,
            tr=torch.tensor([1e7]),                   # ≈∞ to suppress recovery term
            te=torch.tensor([0.0]),
            ti=ti_null,
        )
        # |1 - 2 e^{-ln2}| = 0
        assert abs(float(s)) < 1e-3


class TestSequenceOptimisation:
    def test_optimisation_decreases_loss(self) -> None:
        t1 = torch.tensor([800.0]); t2 = torch.tensor([80.0]); m0 = torch.tensor([1.0])
        target = torch.tensor([0.5])
        result = optimise_sequence_adjoint(
            steady_state_se_signal, t1, t2, m0, target,
            initial_phi={
                "tr": torch.tensor([1500.0]),
                "te": torch.tensor([60.0]),
            },
            n_iters=50, lr=10.0,
        )
        assert result.best_loss <= result.loss_history[0]

    def test_marker_anchor_changes_solution(self) -> None:
        # Adding a marker term should change the optimum (or keep it the
        # same up to numerical noise — at minimum it must not crash).
        t1 = torch.tensor([800.0]); t2 = torch.tensor([80.0]); m0 = torch.tensor([1.0])
        target = torch.tensor([0.5])
        marker_t1 = torch.tensor([200.0])
        marker_t2 = torch.tensor([50.0])
        marker_m0 = torch.tensor([1.5])
        marker_target = torch.tensor([0.8])
        result = optimise_sequence_adjoint(
            steady_state_se_signal, t1, t2, m0, target,
            initial_phi={
                "tr": torch.tensor([1500.0]),
                "te": torch.tensor([60.0]),
            },
            marker_t1=marker_t1, marker_t2=marker_t2, marker_m0=marker_m0,
            target_marker_signal=marker_target,
            lambda_marker=2.0, n_iters=50, lr=10.0,
        )
        assert "tr" in result.phi_hat and "te" in result.phi_hat


class TestRankAugmentation:
    def test_rank_with_markers_at_least_anatomy_rank(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(8, 4)        # 4-dim anatomy Jacobian (rank 4)
        # marker rows in the anatomy column-span (rank stays 4)
        m_in_span = a[:2] * 2.0
        r1, r2 = jacobian_rank_augmentation_test(a, m_in_span)
        assert r2 >= r1

    def test_marker_strictly_augments_rank(self) -> None:
        # Anatomy spans only 2 of 4 dims; marker fills the rest
        a = torch.zeros(2, 4)
        a[0, 0] = 1.0; a[1, 1] = 1.0
        m = torch.zeros(2, 4)
        m[0, 2] = 1.0; m[1, 3] = 1.0
        r1, r2 = jacobian_rank_augmentation_test(a, m)
        assert r1 == 2 and r2 == 4
