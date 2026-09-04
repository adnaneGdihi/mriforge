"""``MultiInferenceStrategy`` and the predict-time data-consistency knob.

``infer_single`` here never calls ``BaseInferenceStrategy.infer``, where the
projection hook runs, and takes no mask or measurement. Accepting the knob would
leave it inert (CLAUDE.md #16), so construction refuses it by name.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.inference.multi_inference_strategy import MultiInferenceStrategy

KNOB = "physics.data_consistency.apply_at_predict"


class _Concrete(MultiInferenceStrategy):
    """``MultiInferenceStrategy`` never implements the base's abstract methods,
    so it cannot be instantiated as shipped (the pipeline's multi branch hits
    the same TypeError). The refusal under test lives in its ``__init__``,
    which this stand-in inherits unchanged."""

    @property
    def training_mode(self):
        from spectramr.config.schemas.enums import TrainingModeTypes

        return TrainingModeTypes.reconstruction

    def preprocess_input(self, input_tensor, **kwargs):
        return input_tensor

    def run_inference(self, input_tensor, **kwargs):
        return input_tensor

    def postprocess_output(self, output_tensor, **kwargs):
        return output_tensor


def _cfg(on: bool):
    return {"physics": {"data_consistency": {"apply_at_predict": on}}, "model": {"model_type": "x"}}


def test_the_knob_is_refused_by_name():
    with pytest.raises(ConfigurationError, match=KNOB.replace(".", r"\.")):
        _Concrete(object(), torch.device("cpu"), _cfg(True))


def test_off_constructs_and_reports_off():
    strategy = _Concrete(object(), torch.device("cpu"), _cfg(False))
    assert strategy.predict_dc is None
    assert strategy.predict_dc_provenance() == {"apply_at_predict": False}


def test_the_bypass_is_real_so_the_refusal_stays_necessary():
    """If ``infer_single`` ever routes through ``infer``, the refusal can go."""
    import inspect

    src = inspect.getsource(MultiInferenceStrategy.infer_single)
    assert "self.infer(" not in src
