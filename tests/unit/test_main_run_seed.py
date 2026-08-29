"""Every entry point seeds from ``run.seed``, not the retired ``training.seed``.

``main.py`` computed ``getattr(settings.training, "seed", None) or 42`` in four
places. Phase 4b moved the key to ``run:`` and made ``training.seed`` a
**raise**-posture rename, so the attribute is absent from
``TrainingStrategyConfigSchema`` *and* rejected in YAML -- the ``getattr`` could
never succeed and every path silently used 42.

``pipelines/train.py`` reads ``config.run.seed`` directly, so training was
unaffected. ``predict`` and ``infer_dataset`` route through the inference
pipeline and never reach that reader, so they seeded at 42 regardless of what
the arm declared, and the report bundle stamped ``seed: None`` into provenance.
"""

from __future__ import annotations

import inspect

import pytest

from mriforge.config.schemas.renames import RENAMES
from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
from mriforge.config.settings import TrainingSettings


def _settings(seed: int) -> TrainingSettings:
    return TrainingSettings.settings_from_dict(
        {
            "data": {"train_path": "/tmp/t", "val_path": "/tmp/v"},
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
            "run": {"seed": seed},
        }
    )


@pytest.mark.unit
def test_training_seed_is_a_raise_rename_with_no_attribute_left():
    assert RENAMES["training.seed"].canonical == "run.seed"
    assert RENAMES["training.seed"].posture == "raise"
    assert "seed" not in TrainingStrategyConfigSchema.model_fields


@pytest.mark.unit
def test_run_seed_resolves_and_is_not_the_default():
    settings = _settings(1234)
    assert settings.run.seed == 1234
    # The tell for the old defect: a declared seed that silently became 42.
    assert settings.run.seed != 42


@pytest.mark.unit
def test_no_entry_point_still_reads_the_retired_spelling():
    """Grep the modules that were repointed.

    A negative source assertion is normally weak, but here the retired spelling
    cannot appear in a fixed file even as documentation: every one of these
    modules names the NEW path in its comment, so a match is a real reader.
    """
    from mriforge import main
    from mriforge.pipelines import train

    for mod in (main, train):
        src = inspect.getsource(mod)
        assert 'getattr(settings.training, "seed"' not in src
        assert 'getattr(config, "seed"' not in src

    assert "settings.run.seed" in inspect.getsource(main)
    assert "config.run.seed" in inspect.getsource(train)
