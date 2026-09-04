"""The model_output snapshot must state which scale each row lives on (#587).

The base-class fallback snapshot pairs the generator's raw output (network
units) with ``target``/``input`` as they entered ``train_step`` — before any
normalization the strategy applies internally. Adjacent rows of one stats table
can therefore sit on different scales.

Observed on exp_11 attention_none: ``model_output`` abs_max 3.8 against
``target`` abs_max 2401, a 630x gap that reads as a broken model. It was not —
``expm1(4.707) * 22.207 = 2435``; the target was displayed in physical units
while the output stayed log-compressed, and the loss compared like with like.
"""

import pytest
import torch

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

_context = BaseTrainingStrategy._model_output_scale_context


def test_context_names_both_scales():
    ctx = _context(torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4))
    assert "network units" in ctx["model_output_scale"]
    assert "pre-strategy-normalization" in ctx["target_input_scale"]


def test_mismatched_scales_raise_a_warning_note():
    """The exact observed magnitudes must trip the guard."""
    model_output = torch.full((1, 2, 4, 4), 3.80)
    target = torch.full((1, 2, 4, 4), 2401.0)
    ctx = _context(model_output, target)
    assert "scale_warning" in ctx
    assert ctx["abs_max_ratio"] == pytest.approx(2401.0 / 3.80, rel=1e-3)
    assert "NOT on a common scale" in ctx["scale_warning"]
    assert (
        "NOT by itself" in ctx["scale_warning"]
    ), "the note must say a scale gap is not evidence of a model defect"


def test_matched_scales_produce_no_warning():
    """Same space => no note, so the warning stays meaningful."""
    ctx = _context(torch.full((1, 2, 4, 4), 4.2), torch.full((1, 2, 4, 4), 4.7))
    assert "scale_warning" not in ctx
    assert ctx["abs_max_ratio"] == pytest.approx(4.7 / 4.2, rel=1e-3)


def test_ratio_is_orientation_independent():
    """A target SMALLER than the output trips the same guard."""
    ctx = _context(torch.full((1, 2, 4, 4), 2401.0), torch.full((1, 2, 4, 4), 3.80))
    assert "scale_warning" in ctx


def test_absolute_maxima_are_recorded():
    ctx = _context(torch.full((1, 2, 4, 4), 3.80), torch.full((1, 2, 4, 4), 2401.0))
    assert ctx["abs_max_model_output"] == pytest.approx(3.80)
    assert ctx["abs_max_target"] == pytest.approx(2401.0)


def test_all_zero_tensors_do_not_divide_by_zero():
    ctx = _context(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4))
    assert "abs_max_ratio" not in ctx
    assert "scale_warning" not in ctx


def test_non_tensor_input_degrades_to_the_scale_labels():
    """Diagnostics must never break training (the caller is inside a try)."""
    ctx = _context(object(), torch.zeros(1, 2, 4, 4))
    assert "model_output_scale" in ctx
    assert "abs_max_ratio" not in ctx
