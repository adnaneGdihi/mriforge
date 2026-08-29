"""Regression: unknown gradient_clip_method silently became norm-clipping.

2026-06-11 infrastructure audit. The stepper dispatched ``"value"`` explicitly
and treated everything else (typos, the bogus ``"norm_type"`` template option)
as norm-clipping. It now validates the advertised set and raises (pitfall #9).
"""

from __future__ import annotations

import pytest

from mriforge.infrastructure.training.optimizers.standard_optimizer_stepper import (
    StandardOptimizerStepper,
)


def test_unknown_clip_method_raises():
    with pytest.raises(ValueError, match="gradient_clip_method"):
        StandardOptimizerStepper(gradient_clip_method="norm_type")


def test_valid_clip_methods_construct():
    for method in ("norm", "value"):
        stepper = StandardOptimizerStepper(gradient_clip_method=method)
        assert stepper.gradient_clip_method == method
