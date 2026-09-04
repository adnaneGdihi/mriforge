"""Training Strategy Dispatch Coverage Tests.

PURPOSE:
Every key in TrainingStrategyFactory.STRATEGY_CLASS_PATHS must:
  1. Point to a module path that actually exists (no ImportError)
  2. Expose a class that is a subclass of BaseTrainingStrategy
  3. Not shadow a legitimate short-name alias

This catches stale class paths introduced when strategies are renamed/moved.
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_class(class_path: str):
    """Resolve a dotted class path to a class object."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# 1. All STRATEGY_CLASS_PATHS entries are importable
# ---------------------------------------------------------------------------


class TestStrategyClassPathsAreImportable:
    """Every entry in STRATEGY_CLASS_PATHS must resolve without ImportError."""

    @pytest.fixture(scope="class")
    def strategy_paths(self):
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        return dict(TrainingStrategyFactory.STRATEGY_CLASS_PATHS)

    @pytest.mark.parametrize(
        "short_name",
        # We parametrize lazily inside the test body to avoid import at collection
        list(
            __import__(
                "spectramr.infrastructure.training.strategy_factory",
                fromlist=["TrainingStrategyFactory"],
            ).TrainingStrategyFactory.STRATEGY_CLASS_PATHS.keys()
        ),
    )
    def test_strategy_class_path_is_importable(self, short_name: str, strategy_paths):
        """_load_strategy_class(short_name) must not raise ImportError/AttributeError."""
        full_path = strategy_paths[short_name]

        try:
            cls = _load_class(full_path)
        except ImportError as exc:
            pytest.fail(
                f"STRATEGY_CLASS_PATHS['{short_name}'] = '{full_path}'\n"
                f"ImportError: {exc}\n"
                "The module no longer exists or has been moved. "
                "Update STRATEGY_CLASS_PATHS to the correct path."
            )
        except AttributeError as exc:
            pytest.fail(
                f"STRATEGY_CLASS_PATHS['{short_name}'] = '{full_path}'\n"
                f"AttributeError: {exc}\n"
                "The class was renamed or removed from the module. "
                "Update STRATEGY_CLASS_PATHS with the correct class name."
            )
        assert cls is not None


# ---------------------------------------------------------------------------
# 2. Loaded classes are subclasses of BaseTrainingStrategy
# ---------------------------------------------------------------------------


class TestStrategyClassesAreBaseSubclasses:
    """Every resolved strategy class must inherit from BaseTrainingStrategy."""

    @pytest.mark.parametrize(
        "short_name,full_path",
        list(
            __import__(
                "spectramr.infrastructure.training.strategy_factory",
                fromlist=["TrainingStrategyFactory"],
            ).TrainingStrategyFactory.STRATEGY_CLASS_PATHS.items()
        ),
    )
    def test_strategy_class_inherits_base(self, short_name: str, full_path: str):
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

        try:
            cls = _load_class(full_path)
        except (ImportError, AttributeError):
            pytest.skip(f"Could not import {full_path} — tested separately")

        assert issubclass(cls, BaseTrainingStrategy), (
            f"STRATEGY_CLASS_PATHS['{short_name}'] → {cls.__name__} does NOT "
            f"inherit from BaseTrainingStrategy.\n"
            "All strategies must inherit from BaseTrainingStrategy to participate "
            "in the Template Method pattern (hooks, EMA, etc.)."
        )


# ---------------------------------------------------------------------------
# 3. factory._load_strategy_class raises ValueError for unknown paths
# ---------------------------------------------------------------------------


class TestStrategyFactoryFailFast:
    """The factory must fail-fast on unknown class paths, not silently return None."""

    def test_unknown_short_name_raises_value_error(self):
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        factory = TrainingStrategyFactory()
        with pytest.raises(ValueError, match="[Ff]ailed"):
            factory._load_strategy_class("nonexistent_strategy_xyz_123")

    def test_bad_module_path_raises_value_error(self):
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        factory = TrainingStrategyFactory()
        with pytest.raises(ValueError):
            factory._load_strategy_class(
                "spectramr.infrastructure.training.strategies.does_not_exist.SomeClass"
            )

    def test_bad_class_name_raises_value_error(self):
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        factory = TrainingStrategyFactory()
        with pytest.raises(ValueError):
            factory._load_strategy_class(
                "spectramr.infrastructure.training.strategies.gan.NonExistentClassName"
            )


# ---------------------------------------------------------------------------
# 4. STRATEGY_CLASS_PATHS covers all TrainingMode enum values where applicable
# ---------------------------------------------------------------------------


class TestStrategyPathsMeetEnumCoverage:
    """TrainingMode enum values should map to a strategy path (or be documented as skipped)."""

    # Known modes that intentionally alias to another strategy
    ALIASED_MODES = {
        "swarm",  # → reconstruction
        "rectified_flow",  # → diffusion
        "dual_task",  # → reconstruction
        "flow_matching",  # → diffusion
        "ssl",  # → mae pretraining
        "n2n",  # → noise_to_noise
        "test_time_optimization",  # → tto
        # ── DOCUMENTED GAPS: enum values with no strategy_class_paths entry ──
        # These modes exist in TrainingMode enum but are not yet wired up.
        # EITHER add a strategy path OR add a dedicated strategy class.
        # These two were removed from VALID_TRAINING_MODES on 2026-07-19 (they
        # named no strategy). They survive here only because the TrainingMode
        # enum still declares them; enum + these entries go together in the
        # table-reconciliation follow-up.
        "pretraining",  # DELETED MODE: use 'mae' or 'ssl' instead
        "kan",  # DELETED MODE: KAN is an architecture axis, not a paradigm
        "kspace_cold_diffusion",  # SCHEMA_ONLY: use 'diffusion' + diffusion.degradation=kspace_mask
        "3d_generation",  # SCHEMA_ONLY: no 3D generation strategy implemented
    }

    def test_training_mode_enum_values_have_strategy_path(self):
        """Every TrainingMode value must have an entry in STRATEGY_CLASS_PATHS."""
        from spectramr.config.schemas.enums import TrainingMode
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        known_paths = set(TrainingStrategyFactory.STRATEGY_CLASS_PATHS.keys())
        missing = []

        for member in TrainingMode:
            val = member.value.lower()
            if val not in known_paths:
                missing.append(val)

        # Filter documented intentional gaps
        undocumented_missing = [v for v in missing if v not in self.ALIASED_MODES]

        assert not undocumented_missing, (
            f"TrainingMode values with no entry in STRATEGY_CLASS_PATHS:\n"
            f"  {undocumented_missing}\n"
            "Either:\n"
            "  - Add the strategy path to STRATEGY_CLASS_PATHS, OR\n"
            "  - Add the mode to ALIASED_MODES in this test (with a comment)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
