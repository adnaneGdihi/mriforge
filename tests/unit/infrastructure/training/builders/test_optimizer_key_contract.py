"""Contract test: optimizer-dict keys agree with TrainingEnvironment properties.

Locks in the cleanup from ``TODO/backlog_standardize_optimizer_keys.md``.
The single canonical key for the main optimizer is ``"opt_g"``; before the
cleanup, ``OptimizationBuilder`` mixed ``"main"`` (single-optimizer path)
and ``"opt_g"`` (GAN path) which silently disabled adversarial training
when the consumer looked under the wrong name.

This test fails loudly the moment the convention drifts again.
"""

from __future__ import annotations

import inspect

import pytest


def test_environment_opt_g_property_only_reads_canonical_key() -> None:
    """``TrainingEnvironment.opt_g`` looks up ``"opt_g"`` and nothing else.

    Regression guard: prior to the cleanup the property fell back to
    ``self.optimizers.get("main")``; that fallback is what masked the
    builder/consumer key mismatch in the first place.
    """
    from mriforge.infrastructure.training.builders.environment import (
        TrainingEnvironment,
    )

    src = inspect.getsource(TrainingEnvironment.opt_g.fget)
    assert '"opt_g"' in src or "'opt_g'" in src
    # No silent fallback to a legacy alias.
    assert '"main"' not in src
    assert "'main'" not in src


def test_optimization_builder_uses_canonical_key_for_single_optimizer() -> None:
    """``OptimizationBuilder`` writes the single optimizer under ``"opt_g"``.

    Until the cleanup the single-optimizer branch wrote ``"main"``,
    while the GAN branch wrote ``"opt_g"``/``"opt_d"`` — two keys for
    the same role.
    """
    from mriforge.infrastructure.training.builders import optimization_builder

    src = inspect.getsource(optimization_builder)
    # Builder no longer assigns the "main" key anywhere.
    assert 'self._optimizers["main"]' not in src
    assert "self._optimizers['main']" not in src
    # The canonical key is still produced.
    assert 'self._optimizers["opt_g"]' in src


def test_train_pipeline_does_not_fall_back_to_main_key() -> None:
    """``train.py`` reads ``"opt_g"`` directly — no ``or get("main")`` chain."""
    from mriforge.pipelines import train as train_module

    src = inspect.getsource(train_module)
    # The fallback chain is gone; the canonical key is the only access path.
    assert 'pipeline.optimizers.get("opt_g")' in src
    assert 'pipeline.optimizers.get("main")' not in src


def test_training_environment_dataclass_property_set() -> None:
    """The five canonical optimizer-related properties are all defined."""
    from mriforge.infrastructure.training.builders.environment import (
        TrainingEnvironment,
    )

    for name in ("opt_g", "opt_d", "optimizer_g", "optimizer_d"):
        assert hasattr(TrainingEnvironment, name), (
            f"TrainingEnvironment is missing canonical property '{name}'"
        )
