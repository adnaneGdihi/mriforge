"""Regression test for the audit-15-F7 fix.

``_validate_data_availability_at_startup`` used to silently default
``dataset_type`` to ``"reconstruction"`` if ``config.training`` was
None or missing ``strategy_class`` — a classic CLAUDE.md pitfall #9
silent fallback. Configs with broken/missing ``training`` blocks
would skip data validation entirely (or validate against the wrong
dataset_type). The post-fix code raises a ``ValueError`` instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mriforge.bootstrap import _validate_data_availability_at_startup


def _config_without_strategy_class() -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(strategy_class=None, device=None),
        data=SimpleNamespace(data_root="/tmp/does_not_exist"),
    )


def _config_without_training() -> SimpleNamespace:
    return SimpleNamespace(
        training=None,
        data=SimpleNamespace(data_root="/tmp/does_not_exist"),
    )


def test_missing_strategy_class_raises() -> None:
    with pytest.raises(ValueError, match="strategy_class is required"):
        _validate_data_availability_at_startup(_config_without_strategy_class())


def test_missing_training_block_raises() -> None:
    with pytest.raises(ValueError, match="strategy_class is required"):
        _validate_data_availability_at_startup(_config_without_training())
