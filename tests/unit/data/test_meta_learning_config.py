"""Phase 4f Amendment J — MetaLearningConfigSchema + dataset refactor."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.data import (
    DataConfigSchema,
    MetaLearningConfigSchema,
)


# ── Schema ──────────────────────────────────────────────────────────────────


def test_meta_learning_defaults() -> None:
    cfg = MetaLearningConfigSchema()
    assert cfg.enabled is False
    assert cfg.support_size == 4
    assert cfg.query_size == 4
    assert cfg.tasks_per_epoch == 100
    assert cfg.task_params == {}


def test_meta_learning_support_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        MetaLearningConfigSchema(support_size=0)


def test_meta_learning_tasks_per_epoch_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        MetaLearningConfigSchema(tasks_per_epoch=0)


def test_meta_learning_accepts_task_params() -> None:
    cfg = MetaLearningConfigSchema(
        enabled=True,
        task_params={"acceleration_factors": [2, 4, 8, 16]},
    )
    assert cfg.task_params["acceleration_factors"] == [2, 4, 8, 16]


def test_data_config_threads_meta_learning_block() -> None:
    cfg = DataConfigSchema(
        meta_learning={
            "enabled": True,
            "support_size": 8,
            "query_size": 16,
            "tasks_per_epoch": 50,
            "task_params": {"acceleration_factors": [2, 4]},
        }
    )
    assert cfg.meta_learning.support_size == 8
    assert cfg.meta_learning.query_size == 16
    assert cfg.meta_learning.tasks_per_epoch == 50
    assert cfg.meta_learning.task_params["acceleration_factors"] == [2, 4]


# ── Dataset refactor (config takes precedence over legacy kwargs) ───────────


def test_meta_learning_dataset_reads_from_config_schema() -> None:
    """Passing ``config=...`` overrides the legacy kwargs."""
    from torch.utils.data import TensorDataset
    import torch

    from mriforge.data.datasets.meta_learning_dataset import MetaLearningDataset

    base = TensorDataset(torch.randn(16, 1, 8, 8))
    cfg = MetaLearningConfigSchema(
        enabled=True,
        support_size=2,
        query_size=3,
        tasks_per_epoch=7,
        task_params={"acceleration_factors": [4, 8]},
    )
    ds = MetaLearningDataset(base_dataset=base, config=cfg)
    assert ds.support_size == 2
    assert ds.query_size == 3
    assert ds.tasks_per_epoch == 7
    assert ds.task_config["acceleration_factors"] == [4, 8]


def test_meta_learning_dataset_backward_compat_with_kwargs() -> None:
    """Legacy callers passing kwargs still work."""
    from torch.utils.data import TensorDataset
    import torch

    from mriforge.data.datasets.meta_learning_dataset import MetaLearningDataset

    base = TensorDataset(torch.randn(8, 1, 8, 8))
    ds = MetaLearningDataset(
        base_dataset=base,
        task_config={"acceleration_factors": [2, 4]},
        support_size=4,
        query_size=4,
        tasks_per_epoch=50,
    )
    assert ds.support_size == 4
    assert ds.tasks_per_epoch == 50
