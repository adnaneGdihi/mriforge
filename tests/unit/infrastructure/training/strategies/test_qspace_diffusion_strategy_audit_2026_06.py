"""Regression tests for the 2026-06 audit of QSpaceDiffusionStrategy.

Audit section "### qspace_diffusion_strategy" (TODO/_high_triage_full.txt,
TODO/strategy_audit_2026_05_31_unreviewed_findings.txt). Three
strategy-local defects were fixed in
``infrastructure/training/strategies/qspace_diffusion_strategy.py``:

#1 [high] Bare ``except Exception: return base`` swallowed SH/lstsq failures
   and silently degraded the arm to vanilla reconstruction (no q-space
   regulariser). FIX: removed the catch so any error propagates (pitfall
   #3/#9 — no silent fallbacks).
#2 [low] Dead, type-confused line ``b0_anchor = base.new_zeros(()) if
   hasattr(base, "new_zeros") else ...`` was immediately overwritten by the
   next line. FIX: deleted the dead branch.
#3 [medium] The SH basis ``Y`` (and ``orders``) was built on ``bvecs.device``
   while the lstsq co-locates with ``signal``; a device mismatch was hidden
   by the bare except (#1). FIX: ``Y = Y.to(signal.device)`` /
   ``orders = orders.to(signal.device)`` before the lstsq.

The tests build the strategy via ``object.__new__`` (bypassing the heavy
parent ``__init__``) and patch ``ReconstructionTrainingStrategy._compute_losses_impl``
to return a controlled base dict so they isolate the q-space behavior — the
same idiom as ``test_low_rank_sparse_decompose.py``.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
import torch

from mriforge.infrastructure.training.strategies.qspace_diffusion_strategy import (
    QSpaceDiffusionStrategy,
    build_sh_basis_unit_sphere,
)
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


def _make_strategy() -> QSpaceDiffusionStrategy:
    """A strategy with only the attrs ``_compute_losses_impl`` touches."""
    s = object.__new__(QSpaceDiffusionStrategy)
    # Class-level knobs are inherited via the class; nothing else is needed
    # because the tested path only reads sh_max_order/lambda_* and the batch.
    return s


def _base_dict() -> dict[str, torch.Tensor]:
    """Mimic the parent's return: a recon loss bound to its own graph."""
    total = (torch.zeros(1, requires_grad=True) + 1.0).squeeze(0)
    return {"loss_total": total, "g_total_loss": total}


def _unit_bvecs(n_dir: int) -> torch.Tensor:
    """``n_dir`` reproducible non-degenerate gradient directions on S^2."""
    g = torch.Generator().manual_seed(0)
    v = torch.randn(n_dir, 3, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


class TestNoSilentFallback:
    """#1 — an lstsq/SH failure must propagate, not return the base recon."""

    def test_shape_mismatch_raises_instead_of_returning_base(self):
        s = _make_strategy()
        # signal direction-count (5) deliberately != bvecs count (10) so the
        # lstsq inside fit_sh_coefs raises a RuntimeError. Pre-fix this was
        # swallowed and ``base`` returned silently.
        signal = torch.randn(1, 5, 4, 4)
        bvecs = _unit_bvecs(10)
        batch = {"dwi_signal": signal, "b_vectors": bvecs}

        with patch.object(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            return_value=_base_dict(),
        ):
            with pytest.raises(RuntimeError):
                s._compute_losses_impl(input_batch=batch, epoch=0)

    def test_bare_except_clause_is_gone_from_source(self):
        # The swallow-and-return-base pattern must not be present.
        src = inspect.getsource(QSpaceDiffusionStrategy._compute_losses_impl)
        assert "except Exception" not in src
        assert "return base" in src  # the guarded early-returns remain
        # No bare ``except ...: return base`` directly following the SH build.
        assert "except Exception:\n            return base" not in src


class TestRegulariserRuns:
    """Happy path: with matching shapes on a single device the q-space
    regulariser terms are added (the swallow no longer hides anything, and
    the device co-location keeps lstsq valid)."""

    def test_sh_terms_present_and_added_to_total(self):
        s = _make_strategy()
        n_dir = 15  # enough directions for max_order=4 (K=15 SH coefs)
        signal = torch.randn(1, n_dir, 4, 4, requires_grad=True)
        bvecs = _unit_bvecs(n_dir)
        batch = {"dwi_signal": signal, "b_vectors": bvecs}

        base_before = _base_dict()
        total_before = base_before["loss_total"].detach().clone()
        with patch.object(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            return_value=base_before,
        ):
            out = s._compute_losses_impl(input_batch=batch, epoch=0)

        assert "loss_sh_laplace_beltrami" in out
        # The Laplace-Beltrami term is the reliable witness the SH fit ran.
        assert out["loss_sh_laplace_beltrami"].item() >= 0.0
        # The total absorbed the extra regulariser term.
        assert out["loss_total"].item() != total_before.item()

    def test_b0_anchor_added_when_m0_map_present(self):
        s = _make_strategy()
        n_dir = 15
        signal = torch.randn(1, n_dir, 4, 4)
        bvecs = _unit_bvecs(n_dir)
        batch = {
            "dwi_signal": signal,
            "b_vectors": bvecs,
            "M0_map": torch.randn(1, 1, 4, 4),
        }
        with patch.object(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            return_value=_base_dict(),
        ):
            out = s._compute_losses_impl(input_batch=batch, epoch=0)
        assert "loss_b0_anchor" in out


class TestDeviceColocation:
    """#3 — Y/orders must be moved onto signal.device before the lstsq."""

    def test_build_basis_lives_on_bvecs_device_so_move_is_required(self):
        # Documents WHY the fix is needed: the basis is created on bvecs.device,
        # so when bvecs and signal diverge the move is the only safeguard.
        bvecs = _unit_bvecs(15)
        Y, orders = build_sh_basis_unit_sphere(bvecs, max_order=4)
        assert Y.device == bvecs.device
        assert orders.device == bvecs.device

    def test_source_moves_basis_to_signal_device(self):
        src = inspect.getsource(QSpaceDiffusionStrategy._compute_losses_impl)
        assert "Y = Y.to(signal.device)" in src
        assert "orders = orders.to(signal.device)" in src


class TestDeadBranchRemoved:
    """#2 — the dead ``base.new_zeros`` branch must be gone."""

    def test_new_zeros_branch_absent(self):
        src = inspect.getsource(QSpaceDiffusionStrategy._compute_losses_impl)
        # The dead, type-confused conditional assignment must be gone.
        assert "base.new_zeros(())" not in src
        assert 'hasattr(base, "new_zeros")' not in src
        # The single canonical zero-init line remains, and ``b0_anchor`` is
        # assigned exactly once at init (no double-assignment overwrite).
        assert "b0_anchor = torch.zeros((), device=signal.device)" in src
        init_lines = [
            ln
            for ln in src.splitlines()
            if ln.strip().startswith(
                "b0_anchor = torch.zeros((), device=signal.device)"
            )
        ]
        assert len(init_lines) == 1
