"""Unit tests for model-level profiling metrics (param_count, inference_latency_ms)."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from spectramr.core.metrics.profiling_metrics import InferenceLatency, ParameterCount
from spectramr.core.metrics.registry import get_metric


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 3, padding=1))


class TestRegistration:
    @pytest.mark.parametrize(
        "key,canonical",
        [
            ("param_count", "param_count"),
            ("num_params", "param_count"),
            ("inference_latency_ms", "inference_latency_ms"),
            ("latency", "inference_latency_ms"),
        ],
    )
    def test_registered_and_aliased(self, key, canonical):
        assert get_metric(key).name == canonical

    def test_higher_is_better_is_false(self):
        assert ParameterCount().higher_is_better is False
        assert InferenceLatency().higher_is_better is False


class TestParameterCount:
    def test_counts_trainable_params(self):
        model = _tiny_model()
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert ParameterCount()(model=model) == float(expected)
        assert expected > 0

    def test_respects_requires_grad(self):
        model = _tiny_model()
        for p in model[0].parameters():
            p.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert ParameterCount(trainable_only=True)(model=model) == float(trainable)
        assert ParameterCount(trainable_only=False)(model=model) == float(total)
        assert trainable < total

    def test_resolves_module_alias_kwarg(self):
        model = _tiny_model()
        assert ParameterCount()(generator=model) > 0

    def test_nan_without_model(self):
        import math

        assert math.isnan(ParameterCount()(prediction=torch.rand(1, 1, 8, 8)))


class TestInferenceLatency:
    def test_finite_positive_latency(self):
        model = _tiny_model()
        probe = torch.rand(1, 1, 16, 16)
        ms = InferenceLatency(n_warmup=1, n_runs=3)(model=model, input_tensor=probe)
        assert ms > 0.0 and ms < 1e6

    def test_uses_prediction_as_probe(self):
        model = _tiny_model()
        ms = InferenceLatency(n_warmup=1, n_runs=2)(torch.rand(1, 1, 16, 16), model=model)
        assert ms > 0.0

    def test_nan_without_model_or_probe(self):
        import math

        assert math.isnan(InferenceLatency()(model=_tiny_model()))
        assert math.isnan(InferenceLatency()(prediction=torch.rand(1, 1, 8, 8)))
