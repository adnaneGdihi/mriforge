"""Strategy-factory wiring for X-Diffusion.

Regression test for the registration-audit gap (2026-05-28):
``XDiffusionTrainingStrategy`` is exported from
``infrastructure.training.strategies.__init__`` but had no entry in
``TrainingStrategyFactory.STRATEGY_CLASS_PATHS``. As a result, a YAML
with ``training_mode: x_diffusion`` would silently fall back to
``DiffusionTrainingStrategy`` instead of the dedicated cross-modal
strategy — a textbook CLAUDE.md pitfall #9 / #10 (silent fallback,
passes-with-warnings).

Two keys must resolve to the X-Diffusion strategy:

* ``x_diffusion`` — matches the model_type used by
  ``@register_model(name="x_diffusion", ...)`` in
  ``models/generators/x_diffusion_generator.py``.
* ``cross_modal_diffusion`` — matches one of the modes advertised in
  ``XDiffusionTrainingStrategy.expected_modes``.
"""

from __future__ import annotations

import importlib

from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory


def _resolve(short_name: str) -> type:
    cls_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS[short_name]
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def test_x_diffusion_alias_resolves_to_x_diffusion_strategy() -> None:
    assert _resolve("x_diffusion").__name__ == "XDiffusionTrainingStrategy"


def test_cross_modal_diffusion_alias_resolves_to_x_diffusion_strategy() -> None:
    assert _resolve("cross_modal_diffusion").__name__ == "XDiffusionTrainingStrategy"


def test_x_diffusion_does_not_fall_back_to_plain_diffusion() -> None:
    """The whole point of the wiring: ``x_diffusion`` must NOT resolve
    to ``DiffusionTrainingStrategy``. Without the dedicated entry, the
    dispatcher's Priority-3 path looks up ``training_mode`` directly,
    and the only ``diffusion``-flavoured key is the plain one."""
    assert _resolve("x_diffusion").__name__ != "DiffusionTrainingStrategy"
    assert _resolve("cross_modal_diffusion").__name__ != "DiffusionTrainingStrategy"


def test_both_x_diffusion_aliases_listed() -> None:
    keys = TrainingStrategyFactory.STRATEGY_CLASS_PATHS.keys()
    assert "x_diffusion" in keys
    assert "cross_modal_diffusion" in keys
