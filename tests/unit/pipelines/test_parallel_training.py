"""Unit tests for src/pipelines/parallel.py — DP/DDP wrapping utilities."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from spectramr.pipelines.parallel import is_rank_zero, unwrap_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class _SimpleModel(nn.Module):
    """Tiny model for wrapping tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_pipeline(models: dict | None = None) -> MagicMock:
    """Build a mock pipeline with a models dict and device."""
    pipeline = MagicMock()
    pipeline.models = models or {"generator": _SimpleModel()}
    pipeline.device = torch.device("cpu")
    pipeline.data_loaders = {}
    return pipeline


def _make_config(strategy: str = "none", **kwargs) -> MagicMock:
    """Build a mock config with a parallel sub-object."""
    parallel = MagicMock()
    parallel.strategy = strategy
    parallel.device_ids = kwargs.get("device_ids", None)
    parallel.output_device = kwargs.get("output_device", None)
    parallel.sync_batch_norm = kwargs.get("sync_batch_norm", False)
    parallel.find_unused_parameters = kwargs.get("find_unused_parameters", False)
    parallel.gradient_as_bucket_view = kwargs.get("gradient_as_bucket_view", False)
    parallel.static_graph = kwargs.get("static_graph", False)

    config = MagicMock()
    config.parallel = parallel
    return config


# ---------------------------------------------------------------------------
# unwrap_model
# ---------------------------------------------------------------------------
class TestUnwrapModel:
    """Tests for unwrap_model()."""

    def test_plain_model_returns_self(self):
        model = _SimpleModel()
        assert unwrap_model(model) is model

    def test_data_parallel_unwraps(self):
        model = _SimpleModel()
        if torch.cuda.device_count() > 0:
            wrapped = nn.DataParallel(model)
            assert unwrap_model(wrapped) is model
        else:
            pytest.skip("No GPU available for DataParallel test")


# ---------------------------------------------------------------------------
# is_rank_zero
# ---------------------------------------------------------------------------
class TestIsRankZero:
    """Tests for is_rank_zero()."""

    def test_none_strategy_returns_true(self):
        config = _make_config(strategy="none")
        assert is_rank_zero(config) is True

    def test_dp_strategy_returns_true(self):
        config = _make_config(strategy="dp")
        assert is_rank_zero(config) is True

    def test_ddp_without_init_returns_true(self):
        config = _make_config(strategy="ddp")
        # dist is not initialized, so should return True
        assert is_rank_zero(config) is True


# ---------------------------------------------------------------------------
# apply_parallelism — REMOVED
# ---------------------------------------------------------------------------
# The wrapping tests that lived here moved to
# ``tests/unit/infrastructure/distributed/test_strategy_registry.py`` along with
# the code.
#
# ``apply_parallelism`` implemented none/dp/ddp in an if/elif and raised on
# anything else, while FSDP was reached through a separate
# ``parallel.fsdp.enabled`` flag read in ModelBuilder. So ``strategy: 'fsdp'``
# raised ValueError AFTER the whole environment was built, and
# ``strategy: 'none'`` + ``fsdp.enabled: true`` silently sharded. It also wrapped
# everything at one point in the build, which cannot be right for both: FSDP must
# wrap before optimizers exist, DP/DDP after.
# ---------------------------------------------------------------------------


class TestModuleSurfaceAfterTheMove:
    """``pipelines.parallel`` keeps only what still belongs in the pipeline layer."""

    def test_apply_parallelism_is_gone(self) -> None:
        import spectramr.pipelines.parallel as parallel_mod

        assert not hasattr(parallel_mod, "apply_parallelism")

    def test_unwrap_model_and_is_rank_zero_survive(self) -> None:
        import spectramr.pipelines.parallel as parallel_mod

        assert set(parallel_mod.__all__) == {"is_rank_zero", "unwrap_model"}

    def test_unwrap_model_is_the_core_ssot_not_a_second_copy(self) -> None:
        from spectramr.core.module_utils import unwrap_model as canonical

        assert unwrap_model is canonical

    def test_the_strategies_are_reachable_from_the_registry_instead(self) -> None:
        from spectramr.infrastructure.distributed.strategy_registry import (
            list_parallel_strategies,
        )

        assert {"none", "dp", "ddp", "fsdp"} <= set(list_parallel_strategies())
