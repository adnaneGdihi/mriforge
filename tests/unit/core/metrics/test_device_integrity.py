"""Device-integrity guard for the whole metric registry.

sim2rank is CUDA-canonical, but every metric here was written when the engine
ran on CPU. Nothing asserted that a metric survives a CUDA input, so five of
them (``cw_ssim``, ``pegc``, ``pgle``, ``pcd``, ``ecfd``) shipped a CPU-only
tensor factory and crashed for an entire sweep -- the engine dutifully logged
``Metric X CRASHED`` once per timestep and dropped them from every leaderboard.
A sixth (``task_based_detectability``) had the same defect and was never even
noticed, because the sweep that surfaced the first five didn't reach it.

Two failure shapes, both fixed at the source rather than papered over:

* ``nn.Module`` metric built on CPU, CUDA input
  -> "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)
  should be the same". The registry now honours ``device=`` (it used to filter
  the kwarg away silently for any ctor that didn't declare it -- pitfall #15).
* device-agnostic metric with a hardcoded CPU tensor factory
  (``torch.zeros`` / ``torch.eye`` / ``torch.linspace`` / ``fftfreq`` with no
  ``device=``) -> "Expected all tensors to be on the same device". These now
  derive device from the input, the pattern ``_gaussian_blur`` already used.

The sweep below is the standing guard: it asserts only that a metric does not
raise a *device-mismatch*. Metrics that reject this generic probe for their own
reasons (wrong rank, missing context, absent optional backend) are not device
bugs, so they pass -- which keeps the guard stable as the battery grows.
"""

from __future__ import annotations

import contextlib
import io
import math

import pytest
import torch

from spectramr.core.metrics.registry import MetricsRegistry

# Substrings that identify a device-placement failure (as opposed to a metric
# legitimately rejecting the probe input).
_DEVICE_MISMATCH_SIGNATURES = (
    "same device",
    "Input type",
    "weight type",
    "expected device",
    "cuda:0 and cpu",
    "cpu and cuda",
)

# Metrics excluded from the registry-wide sweep, with the reason. Kept explicit
# rather than silently skipped: an unexplained exclusion is how a broken metric
# hides (pitfall #16).
_SWEEP_EXCLUSIONS = {
    # Fetches pretrained weights on construction; hangs the sweep on an offline
    # or rate-limited host. Device integrity is covered by its own suite.
    "med_fid": "downloads pretrained weights at construction",
}

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="device-integrity guard requires CUDA"
)

# The metrics this change repaired. Pinned by name so a regression names itself
# instead of hiding in the parametrised sweep.
_REGRESSION_METRICS = ("cw_ssim", "pegc", "pgle", "pcd", "ecfd")


def _is_device_mismatch(exc: Exception) -> bool:
    msg = str(exc)
    return any(sig in msg for sig in _DEVICE_MISMATCH_SIGNATURES)


def _is_finite(value: object) -> bool:
    """Finite check accepting either a float or a 0-d tensor return."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return math.isfinite(float(value))  # type: ignore[arg-type]


def _probe(metric, img: torch.Tensor, ref: torch.Tensor) -> None:
    """Call ``metric`` full-reference first, then no-reference.

    Raises the device-mismatch error if one occurs; swallows every other
    exception (the metric rejecting the probe is not this test's concern).
    """
    for args in ((img, ref), (img,)):
        try:
            with (
                contextlib.redirect_stderr(io.StringIO()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                metric(*args)
            return
        except Exception as exc:
            if _is_device_mismatch(exc):
                raise
    return


def _sweep_keys() -> list[str]:
    return sorted(k for k in MetricsRegistry.list_available() if k not in _SWEEP_EXCLUSIONS)


@_CUDA
@pytest.mark.gpu
@pytest.mark.parametrize("key", _sweep_keys())
def test_metric_accepts_cuda_input_without_device_mismatch(key: str) -> None:
    """No registered metric may die of a device mismatch on a CUDA input.

    This is the guard that did not exist: sim2rank went CUDA-canonical while the
    metric battery stayed CPU-only, and the gap was invisible because a crashed
    metric is merely dropped from the leaderboard, not surfaced as a failure.
    """
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    img = torch.rand(1, 1, 64, 64, device=device)
    ref = torch.rand(1, 1, 64, 64, device=device)

    try:
        metric = MetricsRegistry.get(key, device=str(device))
    except TypeError:
        # ctor rejects device= -> the engine's Strategy-2 fallback builds it bare.
        # That path must be device-safe too, so probe it as the engine would.
        try:
            metric = MetricsRegistry.get(key)
        except Exception:
            pytest.skip(f"{key} not constructible in this environment")
        if isinstance(metric, torch.nn.Module):
            metric = metric.to(device)
    except Exception:  # missing optional backend, etc.
        pytest.skip(f"{key} not constructible in this environment")

    try:
        _probe(metric, img, ref)
    except Exception as exc:
        pytest.fail(f"metric {key!r} raised a device mismatch on CUDA input: {exc}")


@_CUDA
@pytest.mark.gpu
@pytest.mark.parametrize("key", _REGRESSION_METRICS)
def test_previously_cpu_only_metrics_return_finite_on_cuda(key: str) -> None:
    """Regression: the five metrics that crashed every sim2rank timestep.

    Asserts a real measurement, not merely the absence of a crash -- a metric
    that returns NaN has measured nothing and the engine classifies it CRASHED
    anyway (``MetricOutcome.from_value``).
    """
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    img = torch.rand(1, 1, 64, 64, device=device)
    ref = torch.rand(1, 1, 64, 64, device=device)

    metric = MetricsRegistry.get(key, device=str(device))
    value = metric(img, ref) if key == "cw_ssim" else metric(img)
    assert _is_finite(value), f"{key} returned non-finite {value!r} on CUDA"


@_CUDA
@pytest.mark.gpu
def test_registry_honours_device_for_module_metric_without_device_ctor() -> None:
    """``get(key, device=...)`` must actually place an nn.Module on that device.

    ``CWSSIM.__init__`` takes no ``device``, so the registry's signature filter
    used to drop the kwarg and hand back CPU parameters -- an advertised knob
    that silently did nothing (pitfall #15), which is exactly how cw_ssim came
    to crash on every CUDA sweep.
    """
    metric = MetricsRegistry.get("cw_ssim", device="cuda:0")
    assert isinstance(metric, torch.nn.Module)
    devices = {p.device.type for p in metric.parameters()}
    assert devices == {"cuda"}, f"cw_ssim parameters landed on {devices}"


@_CUDA
@pytest.mark.gpu
def test_base_metric_to_keeps_device_label_in_sync() -> None:
    """``BaseMetric.to()`` must update ``self.device``, not just the buffers.

    ``compute_metric`` does ``preds.to(self.device)``. When the label lagged the
    real placement, a metric moved to CUDA after construction pulled its CUDA
    inputs back to the CPU and computed there -- right answer, silently on the
    wrong device (the accelerated-run contract forbids exactly this).
    """
    metric = MetricsRegistry.get("vif").to("cuda:0")
    assert metric.device.type == "cuda"
    assert {b.device.type for b in metric.buffers()} == {"cuda"}

    img = torch.rand(1, 1, 64, 64, device="cuda:0")
    out = metric(img, torch.rand(1, 1, 64, 64, device="cuda:0"))
    if isinstance(out, torch.Tensor):
        assert out.device.type == "cuda", "metric silently computed on the CPU"

    # ...and back, so the label is not a one-way latch.
    assert metric.to("cpu").device.type == "cpu"


def test_registry_device_kwarg_is_noop_for_device_agnostic_metric() -> None:
    """A plain (non-Module) metric must not be perturbed by ``device=``.

    Guards the registry fix against over-reach: it may only ``.to()`` an
    nn.Module, never touch a metric that manages its own placement.
    """
    metric = MetricsRegistry.get("pegc", device="cpu")
    assert not isinstance(metric, torch.nn.Module)
    assert metric.name == "pegc"
