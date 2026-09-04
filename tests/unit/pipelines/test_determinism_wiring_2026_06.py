"""``run_training_pipeline`` must honour ``config.training.deterministic``.

Part of the 2026-06 determinism-SSOT unification: the pipeline previously
hardcoded ``set_global_seed(seed, deterministic=True)``, making the schema
knob a silent no-op (pitfall #15) on the seed-controller mechanism.
"""

from __future__ import annotations

import types

import pytest

from spectramr.pipelines.train import _determinism_from_config

pytestmark = pytest.mark.unit


def _cfg(deterministic: bool | None) -> types.SimpleNamespace:
    training = types.SimpleNamespace()
    if deterministic is not None:
        training.deterministic = deterministic
    return types.SimpleNamespace(training=training)


def test_reads_true() -> None:
    assert _determinism_from_config(_cfg(True)) is True


def test_reads_false() -> None:
    assert _determinism_from_config(_cfg(False)) is False


def test_absent_knob_defaults_true() -> None:
    assert _determinism_from_config(_cfg(None)) is True


def test_none_training_block_defaults_true() -> None:
    assert _determinism_from_config(types.SimpleNamespace(training=None)) is True
