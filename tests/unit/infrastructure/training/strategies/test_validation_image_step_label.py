"""Validation renders must be labelled with the TRAINING iteration (#585).

``_log_validation_images_to_tensorboard`` used to derive the saved-image step
from ``validation_step_count``, a counter bumped once per cascade LEVEL. One
validation event at iteration 3000 therefore wrote R2x as ``step003000`` but
R8x/R32x as ``step000001`` / ``step000002``, and once the counter left 0 it
mislabelled every level on every later event. Anything sorting those renders by
step reads the later cascade levels as the oldest files on disk.
"""

import pytest

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


class _LoopState:
    def __init__(self, iteration):
        self.iteration = iteration


class _Strategy:
    """Bare carrier for the step-resolution seam (no DI container needed)."""

    _validation_image_step = DiffusionTrainingStrategy._validation_image_step

    def __init__(self, iteration, validation_step_count):
        self.loop_state = _LoopState(iteration)
        self.validation_step_count = validation_step_count


@pytest.mark.parametrize("validation_step_count", [0, 1, 2, 7, 134])
def test_step_label_is_the_training_iteration_for_every_cascade_level(
    validation_step_count,
):
    """All three cascade levels of one event share the real iteration.

    ``validation_step_count`` is what differs between levels; it must not reach
    the label. Pre-fix, counts 1 and 2 produced labels 1 and 2.
    """
    strategy = _Strategy(iteration=3000, validation_step_count=validation_step_count)
    assert strategy._validation_image_step() == 3000


def test_step_label_survives_a_second_validation_event():
    """After the first event the counter never returns to 0 — the label still holds.

    This is the case the old ``if step == 0`` fallback could not reach at all:
    at iteration 6000 with the counter at 3, it labelled the render ``step3``.
    """
    strategy = _Strategy(iteration=6000, validation_step_count=3)
    assert strategy._validation_image_step() == 6000


def test_step_label_is_an_int():
    """The label is formatted with ``{step:06d}`` downstream, so it must be int."""
    strategy = _Strategy(iteration=3000, validation_step_count=0)
    assert isinstance(strategy._validation_image_step(), int)
