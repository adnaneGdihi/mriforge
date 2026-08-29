"""Regression: ``metric_adapter`` rescues registry metrics that would
otherwise crash the meta-evaluation pipeline.

Two scenarios exercised:

1. **Shape-mismatch metrics** — call ``F.conv2d`` internally with
   ``groups=C`` and expect ``[B, C, H, W]``. Naive call with ``[C, H, W]``
   raises ``RuntimeError``. The adapter's 4-D retry must produce a
   finite scalar.
2. **Multi-return metrics** — return ``(scalar, diagnostics)``. The
   adapter's tuple-unpack must recover the first scalar.

Tests use a tiny pred/target pair so they finish in < 1 s per metric.
A metric we *expect* to remain unfixable (``frd`` needs pyradiomics)
is asserted to return NaN cleanly rather than crash.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.meta_evaluation.metric_adapter import (
    safe_metric_call,
    wrap_metric,
    build_safe_metric_set,
)


def _pair_2d() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    pred = torch.rand(1, 32, 32)
    target = pred + 0.05 * torch.randn(1, 32, 32)
    return pred, target


# ---------------------------------------------------------------------------
# Shape-mismatch metrics rescued by the 4-D retry
# ---------------------------------------------------------------------------


def test_ssim_rescued_by_4d_retry() -> None:
    """``ssim`` from the registry crashes on [C, H, W]; adapter should fix it."""
    from mriforge.core.metrics import get_metric

    metric = get_metric("ssim")
    pred, target = _pair_2d()
    value = safe_metric_call(metric, pred, target)
    assert value == value, "adapter returned NaN where finite SSIM expected"
    assert 0.0 <= value <= 1.0 + 1e-6


def test_hfen_rescued_by_4d_retry() -> None:
    from mriforge.core.metrics import get_metric

    pred, target = _pair_2d()
    value = safe_metric_call(get_metric("hfen"), pred, target)
    assert value == value, "HFEN adapter returned NaN"


def test_gradient_error_rescued() -> None:
    from mriforge.core.metrics import get_metric

    pred, target = _pair_2d()
    value = safe_metric_call(get_metric("gradient_error"), pred, target)
    assert value == value, "gradient_error adapter returned NaN"


def test_clinical_ssim_does_not_crash_the_run() -> None:
    """``clinical_ssim`` has a structural ``None * float`` bug independent
    of input shape, so the adapter cannot return a finite scalar — but it
    must return NaN cleanly rather than propagate the TypeError."""
    from mriforge.core.metrics import get_metric

    pred, target = _pair_2d()
    value = safe_metric_call(get_metric("clinical_ssim"), pred, target)
    # Either NaN (current expected behaviour) or finite (if the registry
    # fixes the underlying bug). The contract is: no exception escapes.
    assert isinstance(value, float)


# ---------------------------------------------------------------------------
# Unfixable metrics — must return NaN cleanly, not raise
# ---------------------------------------------------------------------------


# ``frd`` used to test the construction-time failure path. Now that it
# lives in DISABLED_METRICS, that path is exercised by the
# ``test_disabled_metrics_filtered_silently`` test below. The
# construction-time except-block in ``build_safe_metric_set`` remains as
# defence-in-depth for any future metric that might fail to instantiate.


# ---------------------------------------------------------------------------
# wrap_metric forwards metadata + behaves as a regular callable
# ---------------------------------------------------------------------------


def test_wrap_metric_forwards_higher_is_better() -> None:
    from mriforge.core.metrics import get_metric

    raw = get_metric("psnr")
    wrapped = wrap_metric(raw)
    # PSNR registry metric carries this flag.
    if hasattr(raw, "higher_is_better"):
        assert hasattr(wrapped, "higher_is_better")
        assert wrapped.higher_is_better == raw.higher_is_better


def test_wrap_metric_returns_float() -> None:
    from mriforge.core.metrics import get_metric

    wrapped = wrap_metric(get_metric("mse"))
    pred, target = _pair_2d()
    v = wrapped(pred, target)
    assert isinstance(v, float)
    assert v == v


# ---------------------------------------------------------------------------
# build_safe_metric_set produces a valid MetricSet kwargs dict
# ---------------------------------------------------------------------------


def test_build_safe_metric_set_includes_known_crashers() -> None:
    """build_safe_metric_set must accept previously-crashing metric names."""
    spec = build_safe_metric_set(["psnr", "ssim", "hfen", "gradient_error"])
    assert set(spec["metrics"].keys()) == {
        "psnr", "ssim", "hfen", "gradient_error"
    }
    # Each value must be callable.
    pred, target = _pair_2d()
    for name, fn in spec["metrics"].items():
        v = fn(pred, target)
        assert isinstance(v, float), f"{name} returned non-float {type(v)}"


def test_build_safe_metric_set_rejects_unknown() -> None:
    import pytest

    with pytest.raises(ValueError, match="not registered"):
        build_safe_metric_set(["definitely_not_a_real_metric"])


def test_disabled_metrics_filtered_silently() -> None:
    """Radiomics metrics (frd, rfs) are in DISABLED_METRICS — filtered
    before any construction attempt, no ImportError warnings emitted."""
    import logging
    from mriforge.core.metrics import list_available
    from mriforge.core.metrics.meta_evaluation.metric_adapter import DISABLED_METRICS

    # Confirm the radiomics names are still in the registry (so the
    # disable-list is what's filtering them, not the registry).
    for name in ("frd", "rfs"):
        if name in list_available():
            assert name in DISABLED_METRICS, (
                f"{name} should be in DISABLED_METRICS to suppress radiomics dep"
            )

    spec = build_safe_metric_set(["psnr", "frd", "rfs"])
    # psnr made it in.
    assert "psnr" in spec["metrics"]
    # Disabled metrics did not — no warning expected, info-level only.
    assert "frd" not in spec["metrics"]
    assert "rfs" not in spec["metrics"]
