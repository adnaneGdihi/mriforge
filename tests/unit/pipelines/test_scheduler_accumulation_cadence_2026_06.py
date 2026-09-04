"""Regression: LR schedulers must step on the optimizer cadence, not per micro-batch.

The trainer performs an optimizer step only every ``gradient_accumulation_steps``
iterations, but the pipeline stepped the LR schedulers on EVERY iteration — so
under accumulation the schedule advanced N times too fast (e.g. a cosine schedule
hit its minimum at 1/N of the intended horizon). ``_should_step_schedulers``
gates the step to the same boundary the optimizer uses.
"""

from __future__ import annotations

from spectramr.pipelines.train import _should_step_schedulers


def test_no_accumulation_steps_each_iteration_after_warmup():
    start = 0
    # The first two iterations are skipped (PyTorch "step before optimizer.step()").
    assert _should_step_schedulers(0, start, 1) is False
    assert _should_step_schedulers(1, start, 1) is False
    assert _should_step_schedulers(2, start, 1) is True
    assert _should_step_schedulers(3, start, 1) is True


def test_accumulation_steps_only_on_boundary():
    start = 0
    gas = 4
    # Only iterations where (iteration + 1) % 4 == 0 are optimizer-step boundaries.
    stepped = [i for i in range(2, 16) if _should_step_schedulers(i, start, gas)]
    assert stepped == [3, 7, 11, 15]


def test_gas_zero_or_negative_treated_as_one():
    assert _should_step_schedulers(2, 0, 0) is True
    assert _should_step_schedulers(2, 0, -3) is True
