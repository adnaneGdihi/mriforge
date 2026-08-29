"""Regression tests for the 2026-06 strategy-audit HIGH-tier fixes, batch 3.

Covers the 8 strategy files whose audit fixes were applied by hand after the
mechanical fix-workflow stalled: mri_slam, volumetric, virtual_fiducial,
meta_learning, disentangled, pmps, mrf_kspace, pretraining.

Assertions are behavioural where cheap, otherwise positive source-greps for the
*new* construct (negative greps collide with explanatory comments that name the
old construct).
"""
from __future__ import annotations

import inspect
import types

import torch


def test_mri_slam_narrow_except_and_no_self_compare_target() -> None:
    from mriforge.infrastructure.training.strategies.mri_slam_strategy import (
        MRISLAMStrategy,
    )

    src = inspect.getsource(MRISLAMStrategy._photometric_loss)
    # Narrowed: only missing-torchkbnufft falls back to the Cartesian proxy.
    assert "except (ImportError, ModuleNotFoundError)" in src
    # The measurement-independent self-comparison fallback is gone (raises now).
    assert "the trajectory/measurement geometry is inconsistent" in src
    assert "else k_full" not in src


def test_volumetric_metrics_adapter_defaulted() -> None:
    from mriforge.infrastructure.training.strategies.volumetric import (
        TRELLISTrainingStrategy,
    )

    src = inspect.getsource(
        TRELLISTrainingStrategy._setup_strategy_specific_components
    )
    assert "self.metrics_adapter = None" in src


def test_virtual_fiducial_docstring_no_false_fixed_noise_claim() -> None:
    from mriforge.infrastructure.training.strategies.virtual_fiducial_strategy import (
        ConcreteVirtualFiducialStrategy,
    )

    src = inspect.getsource(ConcreteVirtualFiducialStrategy.validation_step)
    # The old false opening docstring is gone; the honest caveat is present.
    assert (
        "Validate with fixed noise for reproducibility and image quality metrics."
        not in src
    )
    assert "does NOT fix the simulator noise" in src


def test_meta_learning_inner_loss_is_deterministic_preferred_key() -> None:
    """``_inner_loss`` must prefer a stable key (``l1``) over insertion order."""
    from mriforge.infrastructure.training.strategies.meta_learning_strategy import (
        MetaLearningTrainingStrategy,
    )

    picked = {}

    def _mk(name):
        def _fn(pred, target):
            picked["name"] = name
            return torch.tensor(0.0)

        return _fn

    s = object.__new__(MetaLearningTrainingStrategy)
    # Insertion order puts "zzz" first; the fix must still pick "l1".
    s.env = types.SimpleNamespace(losses={"zzz": _mk("zzz"), "l1": _mk("l1")})
    s._inner_loss(torch.zeros(1), torch.zeros(1))
    assert picked["name"] == "l1"


def test_disentangled_val_image_errors_logged_not_swallowed() -> None:
    from mriforge.infrastructure.training.strategies.disentangled_strategy import (
        DisentangledTrainingStrategy,
    )

    src = inspect.getsource(DisentangledTrainingStrategy)
    # The bare ``except Exception: pass`` blocks now log.
    assert "val image logging failed" in src
    assert "val image save failed" in src


def test_pmps_fixtures_check_is_anchored() -> None:
    from mriforge.infrastructure.training.strategies import pmps_strategies

    src = inspect.getsource(pmps_strategies)
    # The bare-substring ``"fixtures" in str(...)`` (which a
    # ``/data/fixtures_run/`` path would satisfy) is replaced by the anchored
    # ``tests/fixtures`` path check at every site.
    assert '"tests/fixtures" in str(' in src


def test_mrf_kspace_dead_solver_removed() -> None:
    from mriforge.infrastructure.training.strategies import mrf_kspace_strategies

    # The dead solver import/attr is gone.
    assert not hasattr(mrf_kspace_strategies, "SeparableBeltrami4DSolver")
    src = inspect.getsource(
        mrf_kspace_strategies.SpatiotemporalMRFReconStrategy
    )
    assert "self.solver = " not in src


def test_pretraining_uses_resolve_legacy_batch_and_reraises_val() -> None:
    from mriforge.infrastructure.training.strategies.pretraining import (
        MAEPretrainingStrategy,
    )

    compute_src = inspect.getsource(MAEPretrainingStrategy._compute_losses_impl)
    assert "_resolve_legacy_batch" in compute_src
    val_src = inspect.getsource(MAEPretrainingStrategy.validation_step)
    # The validation-metrics failure now re-raises instead of swallowing.
    assert "raise" in val_src
