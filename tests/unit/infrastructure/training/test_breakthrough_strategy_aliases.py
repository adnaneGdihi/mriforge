"""Strategy-factory registration of the 2026 breakthrough aliases.

The two new aliases — ``sle_compressed_sensing`` and ``spectral_triple`` —
resolve to ``ReconstructionTrainingStrategy`` (per the
``register-and-integrate-per-component`` rule), with the paradigm-specific
behaviour delivered by the YAML's data/mask/loss configuration.
"""

from __future__ import annotations

import importlib

from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory


def test_sle_compressed_sensing_alias_registered() -> None:
    cls_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["sle_compressed_sensing"]
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    assert cls.__name__ == "ReconstructionTrainingStrategy"


def test_spectral_triple_alias_registered() -> None:
    cls_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["spectral_triple"]
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    assert cls.__name__ == "ReconstructionTrainingStrategy"


def test_breakthrough_aliases_are_listed() -> None:
    """The two aliases appear in STRATEGY_CLASS_PATHS so the strict-mode
    audit can mention them in user-facing error messages."""
    keys = TrainingStrategyFactory.STRATEGY_CLASS_PATHS.keys()
    assert "sle_compressed_sensing" in keys
    assert "spectral_triple" in keys
