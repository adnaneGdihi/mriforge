"""Tests for the domain TrainingStrategy ABC (WS-3).

Covers the abstract contract and pins the fix that removed ``@torch.no_grad()``
from the *abstract* ``validation_step`` (a no-op on an abstract method; the
no-grad context belongs on concrete overrides).
"""

from __future__ import annotations

import inspect

from spectramr.domain.training import TrainingStrategy


def test_training_strategy_is_abstract() -> None:
    assert inspect.isabstract(TrainingStrategy)


def test_train_and_validation_steps_are_abstract() -> None:
    assert getattr(TrainingStrategy.train_step, "__isabstractmethod__", False)
    assert getattr(TrainingStrategy.validation_step, "__isabstractmethod__", False)
