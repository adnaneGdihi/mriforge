"""Tests for the Gaussian-NLL calibration metric (B-2.9)."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.quantitative.calibration_nll import GaussianNLLMetric


def test_responds_to_variance() -> None:
    # The metric must CONSUME the predicted variance: a well-matched variance scores
    # lower NLL than a mismatched one (so it measures calibration, not just the mean).
    torch.manual_seed(0)
    target = torch.randn(2, 1, 8, 8)
    m = GaussianNLLMetric()
    # Use a noisy mean (residual std ~0.5) so an honest variance beats an
    # overconfident (near-zero) one.
    noisy_mean = target + 0.5 * torch.randn_like(target)
    honest = torch.cat([noisy_mean, torch.full_like(noisy_mean, -1.4)], dim=1)  # var~0.25
    over = torch.cat([noisy_mean, torch.full_like(noisy_mean, -8.0)], dim=1)  # var~0
    assert m(honest, target) < m(over, target)


def test_higher_is_better_false() -> None:
    assert GaussianNLLMetric().higher_is_better is False


def test_requires_two_channels() -> None:
    with pytest.raises(ValueError):
        GaussianNLLMetric()(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))


def test_registered_via_computer_path_with_device() -> None:
    from spectramr.core.metrics.registry import get_metric

    m = get_metric("gaussian_nll", device=torch.device("cpu"))
    pred = torch.cat([torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)], dim=1)
    val = m(pred, torch.zeros(1, 1, 8, 8), domain="image", data_range=1.0)
    assert val == val  # not NaN


def test_metrics_package_import_does_not_pull_models_losses() -> None:
    """Regression (WS-6): the module-scope ``heteroscedastic_nll`` import made
    the core.metrics walk-discovery trigger the entire losses-package init
    mid-walk, which then re-entered the partially-initialized core.metrics and
    left the loss registry silently half-populated. The import is now lazy —
    importing core.metrics must not drag in models.losses (which is also the
    layer direction: core must not depend on models).
    """
    import os
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import spectramr.core.metrics\n"
        "assert 'spectramr.models.losses' not in sys.modules, "
        "'core.metrics import dragged in models.losses again'\n"
    )
    env = {**os.environ, "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
