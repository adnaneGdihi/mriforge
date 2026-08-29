"""Tests for SparseFrameStrategy (A-6.5 SCO-Frame, component C)."""

from __future__ import annotations

import types

import pytest
import torch

from mriforge.config.schemas.training.sparse_frame import SparseFrameConfig
from mriforge.infrastructure.training.strategies.sparse_frame_strategy import SparseFrameStrategy
from mriforge.models.generators.tight_frame_learner import TightFrameLearner


def _strategy(lam_c: float, lam_p: float = 0.01) -> SparseFrameStrategy:
    strat = object.__new__(SparseFrameStrategy)
    strat.env = types.SimpleNamespace(generator=TightFrameLearner(num_atoms=8, kernel_size=3))
    strat._lambda_coherence = lam_c
    strat._lambda_parseval = lam_p
    from mriforge.models.losses.registry import create_loss

    strat._coherence_loss = create_loss("frame_coherence")
    return strat


def test_adds_coherence_and_parseval_to_g_total_loss(monkeypatch) -> None:
    import mriforge.infrastructure.training.strategies.sparse_frame_strategy as mod

    base = torch.tensor(1.0, requires_grad=True)
    monkeypatch.setattr(
        mod.ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        lambda self, **k: {"g_total_loss": base},
    )
    strat = _strategy(lam_c=0.1, lam_p=0.01)
    out = strat._compute_losses_impl(input_batch=None, target_batch=None, epoch=0)
    assert "frame_coherence" in out and "frame_parseval" in out
    assert out["g_total_loss"].requires_grad
    assert float(out["g_total_loss"]) > 1.0  # penalties added on top of base recon loss


def test_no_coherence_term_when_lambda_zero(monkeypatch) -> None:
    import mriforge.infrastructure.training.strategies.sparse_frame_strategy as mod

    base = torch.tensor(1.0, requires_grad=True)
    monkeypatch.setattr(
        mod.ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        lambda self, **k: {"g_total_loss": base},
    )
    strat = _strategy(lam_c=0.0, lam_p=0.0)  # both off -> control
    out = strat._compute_losses_impl(input_batch=None, target_batch=None, epoch=0)
    assert "frame_coherence" not in out
    assert float(out["g_total_loss"]) == pytest.approx(1.0)


def test_validation_emits_mutual_coherence_witness(monkeypatch) -> None:
    import mriforge.infrastructure.training.strategies.sparse_frame_strategy as mod

    monkeypatch.setattr(
        mod.ReconstructionTrainingStrategy,
        "validation_step",
        lambda self, *a, **k: {"val_psnr": 30.0},
    )
    strat = _strategy(lam_c=0.1)
    out = strat.validation_step(None, None)
    assert out["val_psnr"] == 30.0
    assert 0.0 <= out["val_mutual_coherence"] <= 1.0 + 1e-6


def test_setup_reads_config_block(monkeypatch) -> None:
    import mriforge.infrastructure.training.strategies.sparse_frame_strategy as mod

    monkeypatch.setattr(
        mod.ReconstructionTrainingStrategy, "_setup_strategy_specific_components", lambda self: None
    )
    cfg = SparseFrameConfig(lambda_coherence=0.25, lambda_parseval=0.02)
    strat = object.__new__(SparseFrameStrategy)
    strat.config = types.SimpleNamespace(training=types.SimpleNamespace(sparse_frame=cfg))
    strat.env = types.SimpleNamespace(generator=TightFrameLearner(num_atoms=8, kernel_size=3))
    strat.logging_service = None
    strat._setup_strategy_specific_components()
    assert strat._lambda_coherence == 0.25
    assert strat._lambda_parseval == 0.02


def test_schema_rejects_out_of_bounds_lambda() -> None:
    # A pathological weight must be rejected at config-load, not discovered at divergence.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SparseFrameConfig(lambda_coherence=1e6)
    with pytest.raises(ValidationError):
        SparseFrameConfig(lambda_parseval=-1.0)


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "sparse_frame" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "sparse_frame" in TrainingStrategyConfigSchema.model_fields
