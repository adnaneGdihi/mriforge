"""Regression tests for EMAMixin's empty-marker shape.

Pin the post-audit-05-F3 cleanup: the mixin used to expose
``_update_ema`` with a ``pass`` body and nothing called it. The actual
EMA shadow update happens once per step at
``src/pipelines/train.py`` (``pipeline.ema.update(pipeline.generator)``).
Re-adding a dead method here would put us right back into CLAUDE.md
pitfall #9 (silent fallback / advertised-but-never-fires).
"""

from __future__ import annotations

from mriforge.infrastructure.training.strategies.mixins.ema import EMAMixin


def test_ema_mixin_is_empty_marker() -> None:
    own_methods = {name for name in vars(EMAMixin) if not name.startswith("__")}
    assert not own_methods, (
        f"EMAMixin re-introduced method(s) {sorted(own_methods)}. "
        "EMA updates live in src/pipelines/train.py; the mixin is a marker "
        "only — see TODO/audit/05_strategies_core_mixins_builders.md F3."
    )


def test_base_training_strategy_still_inherits_ema_mixin() -> None:
    """Keep the MRO intact so existing isinstance checks don't trip."""
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy

    assert EMAMixin in BaseTrainingStrategy.__mro__
