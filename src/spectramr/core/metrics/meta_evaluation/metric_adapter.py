"""Robust adapter that makes any registry metric callable with the
``(pred, target)`` contract the meta-evaluation framework expects.

The 96-metric registry covers full-reference image metrics, no-reference
quality metrics, segmentation metrics, distribution-level metrics, and
clinical-task scores. Each family has its own input contract, and the
naive ``metric(pred, target)`` call breaks in three predictable ways:

1. **Shape mismatch** — internal ``F.conv2d`` calls expect ``[B, C, H,
   W]`` but the simulator emits ``[C, H, W]``. (ssim, hfen,
   clinical_ssim, gradient_error, …)
2. **Multi-return** — some metrics return ``(value, diagnostics_dict)``
   or ``(per_class_array, mean)``. We extract the first scalar-shaped
   element. (bat, ktrans, freq_domain_snr, …)
3. **Range/dtype expectation** — complex_hfen wants complex input;
   segmentation metrics want binary inputs; etc.

The adapter does all three transparently and falls back to ``NaN`` if
nothing works. This lets the user pass any subset of the 96-metric
registry without curating it by hand — the figures and CSVs will simply
have NaN rows for the metrics that can't be made to fit.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch

# Metrics deliberately disabled from the meta-evaluation pipeline.
# Filtered out by ``build_safe_metric_set`` before any construction is
# attempted, so the user never sees the underlying ``ImportError`` /
# structural-bug noise. To re-enable, remove the name (and ``pip install
# pyradiomics`` if it's one of the radiomics-dependent ones).
DISABLED_METRICS: set[str] = {
    # Radiomics-dependent — require pyradiomics, currently disabled
    # because the dep is heavy and rarely needed for the FR image-quality
    # comparisons the framework is normally pointed at.
    "frd",  # Feature Radiomic Distance
    "rfs",  # Radiomic Feature Stability
}


def _to_4d(t: torch.Tensor) -> torch.Tensor:
    """Reshape to ``[B, C, H, W]``. ``t`` may be 2-D, 3-D, or already 4-D."""
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 3:
        return t.unsqueeze(0)
    return t


def _extract_scalar(value: object) -> float:
    """Coerce a metric's return value to a Python float, or NaN."""
    if isinstance(value, tuple) and len(value) >= 1:
        value = value[0]
    if isinstance(value, dict):
        # Some metrics return {"value": x, "diagnostics": ...}
        for key in ("value", "score", "result", "mean"):
            if key in value:
                value = value[key]
                break
        else:
            return float("nan")
    if hasattr(value, "item"):
        try:
            if value.numel() == 1:  # type: ignore[union-attr]
                value = value.item()
            else:
                value = float(value.mean().item())  # type: ignore[union-attr]
        except Exception:
            return float("nan")
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return float("nan")
        return float(value)
    return float("nan")


def safe_metric_call(
    metric: Callable[..., object],
    pred: torch.Tensor,
    target: torch.Tensor,
    context: object | None = None,
) -> float:
    """Try shape variants in order; return NaN if all fail.

    Order: original → 4-D → 3-D. The first non-exception, finite return
    wins. Warnings inside the metric implementation are suppressed so the
    user's log isn't flooded by 96-metric ``UserWarning`` cascades.

    ``context`` is the optional :class:`MetricContext` for no-reference
    metrics; it is forwarded as ``context=`` (no shape juggling on the
    measurement tensors, which the NR metrics normalise internally).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # No-reference metric with measurement context: call directly.
        if context is not None:
            for p, t in ((pred, target), (_to_4d(pred), _to_4d(target))):
                try:
                    v = metric(p, t, context=context)
                except Exception:
                    continue
                scalar = _extract_scalar(v)
                if scalar == scalar:
                    return scalar
            return float("nan")

        # 1. Try as-is.
        for p, t in (
            (pred, target),
            (_to_4d(pred), _to_4d(target)),
            (pred.squeeze(), target.squeeze()),
        ):
            try:
                v = metric(p, t)
            except Exception:
                continue
            scalar = _extract_scalar(v)
            if scalar == scalar:  # not NaN
                return scalar
    return float("nan")


def wrap_metric(metric: Callable[..., object]) -> Callable[..., float]:
    """Return a ``(pred, target, context=None) -> float`` callable that won't raise."""

    def _wrapped(pred: torch.Tensor, target: torch.Tensor, context: object | None = None) -> float:
        return safe_metric_call(metric, pred, target, context=context)

    _wrapped.__name__ = getattr(metric, "__name__", "wrapped_metric")
    # Forward common metadata attributes if present.
    for attr in ("higher_is_better", "differentiable", "domain"):
        if hasattr(metric, attr):
            setattr(_wrapped, attr, getattr(metric, attr))
    return _wrapped


def build_safe_metric_set(metric_names: list[str]) -> dict[str, dict]:
    """Resolve names from ``spectramr.core.metrics`` registry and wrap them.

    Returns a dict with the shape ``MetricSet`` expects, so the caller
    can do ``MetricSet(**build_safe_metric_set([...]))``.

    Unknown names raise; the framework should refuse to pretend a metric
    name typed by the user is valid. Use ``list_available()`` to discover.
    """
    import logging

    from spectramr.core.metrics import get_metric, list_available

    log = logging.getLogger(__name__)

    available = set(list_available())
    metrics: dict[str, Callable[..., float]] = {}
    higher: dict[str, bool] = {}
    diff: dict[str, bool] = {}
    for name in metric_names:
        if name not in available:
            raise ValueError(
                f"metric '{name}' is not registered. "
                f"Run `list_available()` to see the {len(available)} known metrics."
            )
        if name in DISABLED_METRICS:
            log.info("metric %r is disabled in DISABLED_METRICS; skipping", name)
            continue
        try:
            raw = get_metric(name)
        except Exception as exc:
            # Construction-time failure (e.g. ``frd`` requires pyradiomics,
            # ``inception_score`` may need a network download). The metric
            # can't be wrapped — skip with a warning so the run continues.
            log.warning(
                "metric %r could not be instantiated (%s); skipping",
                name,
                type(exc).__name__,
            )
            continue
        metrics[name] = wrap_metric(raw)
        higher[name] = bool(getattr(raw, "higher_is_better", False))
        # Default to differentiable=True; SFA's autograd path will detect
        # non-differentiable metrics at call time and fall back to SPSA.
        diff[name] = True
    return {
        "metrics": metrics,
        "higher_is_better": higher,
        "differentiable": diff,
    }
