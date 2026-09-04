"""Model-level profiling metrics (parameter count, inference latency).

Unlike full-reference metrics, these are functions of the *model*, not of a
``(prediction, target)`` image pair, so they are annotated
``MetricType.DOMAIN_SPECIFIC`` in ``scripts/sim2rank/metrics_list.py``. That
keeps them out of both swept buckets (``PER_IMAGE_SPECS`` and ``SUMMARY_SPECS``)
and makes the registry contract harness skip the synthetic-pair forward call.
They are consumed by the reporting / profiling pipeline (which passes the model
via the ``model`` kwarg), not by the per-image validation loop. Both are
referenced as ``metadata.secondary_metrics`` by the HyperMamba VF arm.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from spectramr.core.metrics.registry import register_metric


def _resolve_model(model: object | None, kwargs: dict[str, object]) -> nn.Module | None:
    """Find an ``nn.Module`` among the model kwarg / common aliases."""
    for candidate in (model, kwargs.get("module"), kwargs.get("net"), kwargs.get("generator")):
        if isinstance(candidate, nn.Module):
            return candidate
    return None


@register_metric("param_count", aliases=["params", "parameter_count", "num_params"])
class ParameterCount:
    """Trainable parameter count of a model (summary metric).

    Lower is better at fixed quality: a lighter model is preferable. Returns
    the number of trainable parameters (``requires_grad=True``); pass
    ``trainable_only=False`` to count all parameters.
    """

    def __init__(self, trainable_only: bool = True) -> None:
        self.trainable_only = trainable_only

    @property
    def name(self) -> str:
        return "param_count"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        *,
        model: object | None = None,
        trainable_only: bool | None = None,
        **kwargs: object,
    ) -> float:
        """Count parameters of ``model`` (or a ``module``/``net``/``generator`` kwarg).

        Returns ``nan`` when no model is supplied — this metric is a function of
        the model, not of the image pair, so it is undefined without one.
        """
        net = _resolve_model(model, kwargs)
        if net is None:
            return float("nan")
        only_trainable = self.trainable_only if trainable_only is None else trainable_only
        params = (p for p in net.parameters() if (p.requires_grad or not only_trainable))
        return float(sum(p.numel() for p in params))


@register_metric("inference_latency_ms", aliases=["latency", "latency_ms", "inference_latency"])
class InferenceLatency:
    """Mean forward-pass latency in milliseconds (summary metric).

    Lower is better. Requires both a ``model`` and an ``input_tensor`` (or a
    ``prediction`` tensor to use as a shape probe). CUDA timings synchronise
    around each run; CPU timings use ``time.perf_counter``.
    """

    def __init__(self, n_warmup: int = 2, n_runs: int = 10) -> None:
        self.n_warmup = n_warmup
        self.n_runs = n_runs

    @property
    def name(self) -> str:
        return "inference_latency_ms"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        *,
        model: object | None = None,
        input_tensor: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """Measure mean forward latency (ms) of ``model`` on ``input_tensor``.

        Returns ``nan`` when no model or no input probe is available.
        """
        net = _resolve_model(model, kwargs)
        probe = input_tensor if input_tensor is not None else prediction
        if net is None or probe is None:
            return float("nan")

        device = probe.device
        is_cuda = device.type == "cuda"
        net.eval()
        with torch.no_grad():
            for _ in range(self.n_warmup):
                net(probe)
            if is_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(self.n_runs):
                net(probe)
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
        return float(elapsed / max(self.n_runs, 1) * 1e3)
