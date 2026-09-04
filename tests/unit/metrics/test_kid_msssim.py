"""Unit tests for KID and MS-SSIM metrics and their integration with MetricsMixin.

Tests cover:
- KID small batch accumulation (subset_size=8 default)
- KID finalize flow (stateful metric)
- MSSSIM basic computation
- MetricsMixin training computer includes ms_ssim and kid when flags are set
"""

from unittest.mock import MagicMock

import pytest

from tests.utils.optional_backends import requires_torch_fidelity
import torch

# ─────────────────────────────────────────────
# KID metric
# ─────────────────────────────────────────────


@requires_torch_fidelity
def test_kid_default_subset_size():
    """Verify KID default subset_size was lowered to 8 (not 50)."""
    from spectramr.core.metrics.evaluation_metrics import KID

    kid = KID(device="cpu")
    assert kid.subset_size == 8


@requires_torch_fidelity
def test_kid_compute_metric_returns_zero_during_accumulation():
    """KID.compute_metric() returns 0.0; actual value comes from compute()."""
    from spectramr.core.metrics.evaluation_metrics import KID

    kid = KID(subset_size=4, device="cpu")
    preds = torch.rand(4, 1, 32, 32)
    target = torch.rand(4, 1, 32, 32)
    result = kid.compute_metric(preds, target)
    assert float(result) == pytest.approx(0.0)


@requires_torch_fidelity
def test_kid_summarize_flag():
    """KID instance must have summarize=True (instance attr set in __init__)."""
    from spectramr.core.metrics.evaluation_metrics import KID

    kid = KID(device="cpu")
    # BaseMetric.__init__ sets self.summarize=False; KID.__init__ must override it
    assert (
        kid.summarize is True
    ), "KID.summarize must be True so ValidationMetricsComputer treats it as stateful"


@requires_torch_fidelity
def test_kid_compute_after_accumulation():
    """KID.compute() returns a finite float after accumulating batches."""
    pytest.importorskip("torchmetrics", exc_type=ImportError)
    from spectramr.core.metrics.evaluation_metrics import KID

    kid = KID(subset_size=4, device="cpu")
    # Accumulate 3 batches of 4 samples = 12 samples total (≥ subset_size=4 ×2 real+fake)
    for _ in range(3):
        preds = torch.rand(4, 1, 32, 32)
        target = torch.rand(4, 1, 32, 32)
        kid.update(preds, target)

    value = kid.compute()
    assert torch.isfinite(value), f"KID.compute() returned non-finite: {value}"


@requires_torch_fidelity
def test_kid_returns_zero_on_insufficient_samples():
    """KID returns 0.0 (not an exception) when not enough samples are accumulated."""
    pytest.importorskip("torchmetrics", exc_type=ImportError)
    from spectramr.core.metrics.evaluation_metrics import KID

    # subset_size=50 with only 2 samples — should not raise
    kid = KID(subset_size=50, device="cpu")
    preds = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    kid.update(preds, target)

    value = kid.compute()
    assert float(value) == pytest.approx(0.0)


# ─────────────────────────────────────────────
# MS-SSIM metric
# ─────────────────────────────────────────────


def test_msssim_identical_images():
    """MS-SSIM of identical images should be close to 1.0."""
    pytest.importorskip("torchmetrics", exc_type=ImportError)
    from spectramr.core.metrics.evaluation_metrics import MSSSIM

    metric = MSSSIM(device="cpu")
    # MS-SSIM with 5-level pyramid and kernel_size=11 requires image > 160px
    img = torch.rand(2, 1, 192, 192)
    target = img.clone()
    val = metric(img, target)
    assert float(val) > 0.9, f"Expected MS-SSIM ≈ 1 for identical images, got {val}"


def test_msssim_noise_sensitivity():
    """MS-SSIM decreases when noise is added."""
    pytest.importorskip("torchmetrics", exc_type=ImportError)
    from spectramr.core.metrics.evaluation_metrics import MSSSIM

    metric = MSSSIM(device="cpu")
    # MS-SSIM with 5-level pyramid requires image > 160px
    img = torch.rand(2, 1, 192, 192)
    noisy = (img + 0.5 * torch.randn_like(img)).clamp(0, 1)

    clean_score = metric(img, img.clone())
    noisy_score = metric(noisy, img.clone())
    assert float(noisy_score) < float(
        clean_score
    ), "MS-SSIM should be lower for noisy images"


def test_msssim_registered_in_registry():
    """MS-SSIM is accessible through MetricsRegistry under 'ms_ssim'."""
    from spectramr.core.metrics.registry import MetricsRegistry

    metric = MetricsRegistry.get("ms_ssim", device="cpu")
    assert metric is not None


# ─────────────────────────────────────────────
# MetricsMixin training computer
# ─────────────────────────────────────────────


def _make_metrics_config(**flags):
    """Build a mock metrics config object with compute_* flags."""
    cfg = MagicMock()
    cfg.domain = "image"
    # `metrics.compute` WINS OUTRIGHT over the compute_* flags when truthy
    # (metrics_mixin.py:330). On a MagicMock it auto-vivifies to a truthy Mock
    # that iterates empty, so the resolver took the list branch, found nothing,
    # and every flag below was dead. Declare it empty so these tests exercise
    # the flag fallback they are actually about.
    cfg.compute = []
    default_flags = {
        "compute_psnr": False,
        "compute_ssim": False,
        "compute_mse": False,
        "compute_mae": False,
        "compute_lpips": False,
        "compute_kid": False,
        "compute_ms_ssim": False,
        "compute_hfen": False,
        "compute_nmse": False,
        "compute_fid": False,
    }
    default_flags.update(flags)
    for attr, val in default_flags.items():
        setattr(cfg, attr, val)
    return cfg


def _make_full_config(**metric_flags):
    """Build a minimal top-level config mock for MetricsMixin tests."""
    config = MagicMock()
    config.metrics = _make_metrics_config(**metric_flags)
    config.data.data_range = None
    config.physics = None
    # No validation config → uses training metrics fallback
    config.validation = None
    return config


class _ConcreteMetricsMixin:
    """Minimal concrete class using MetricsMixin for testing."""

    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin

    # Bring methods in without full strategy hierarchy
    _get_training_metrics_computer = MetricsMixin._get_training_metrics_computer
    _extract_metrics_from_config = MetricsMixin._extract_metrics_from_config
    _validate_metrics_domain_consistency = lambda self, cfg: None  # no-op

    def __init__(self, config):
        self.config = config
        self.device = "cpu"
        self._training_computer = None

    @property
    def training_metrics_computer(self):
        return self._get_training_metrics_computer(self.config)


def test_training_computer_includes_ms_ssim_when_flag_set():
    """MetricsMixin training computer includes ms_ssim when compute_ms_ssim=True."""
    cfg = _make_full_config(compute_ms_ssim=True, compute_psnr=True)
    instance = _ConcreteMetricsMixin(cfg)
    computer = instance.training_metrics_computer
    metric_names = computer.get_metric_names()
    assert "ms_ssim" in metric_names, f"Expected 'ms_ssim' in {metric_names}"


@requires_torch_fidelity
def test_training_computer_includes_kid_when_flag_set():
    """MetricsMixin training computer includes kid when compute_kid=True."""
    cfg = _make_full_config(compute_kid=True, compute_psnr=True)
    instance = _ConcreteMetricsMixin(cfg)
    computer = instance.training_metrics_computer
    metric_names = computer.get_metric_names()
    assert "kid" in metric_names, f"Expected 'kid' in {metric_names}"


def test_training_computer_excludes_ms_ssim_when_flag_false():
    """MetricsMixin training computer excludes ms_ssim when compute_ms_ssim=False."""
    cfg = _make_full_config(compute_ms_ssim=False, compute_psnr=True)
    instance = _ConcreteMetricsMixin(cfg)
    computer = instance.training_metrics_computer
    metric_names = computer.get_metric_names()
    assert "ms_ssim" not in metric_names, f"Unexpected 'ms_ssim' in {metric_names}"


@requires_torch_fidelity
def test_training_computer_excludes_kid_when_flag_false():
    """MetricsMixin training computer excludes kid when compute_kid=False."""
    cfg = _make_full_config(compute_kid=False, compute_psnr=True)
    instance = _ConcreteMetricsMixin(cfg)
    computer = instance.training_metrics_computer
    metric_names = computer.get_metric_names()
    assert "kid" not in metric_names, f"Unexpected 'kid' in {metric_names}"


def test_training_computer_fallback_when_no_flags():
    """MetricsMixin training computer has psnr+ssim fallback when no flags enabled."""
    config = MagicMock()
    config.metrics = None
    config.data.data_range = None
    config.physics = None
    config.validation = None

    instance = _ConcreteMetricsMixin(config)
    computer = instance.training_metrics_computer
    metric_names = computer.get_metric_names()
    assert "psnr" in metric_names or "ssim" in metric_names


# ─────────────────────────────────────────────
# ValidationMetricsConfig list path (KID enabled via validation.metrics)
# ─────────────────────────────────────────────


@requires_torch_fidelity
def test_validation_metrics_config_from_list_includes_kid():
    """ValidationMetricsConfig.from_list(['kid']) creates enabled KID spec."""
    from spectramr.core.metrics.types import ValidationMetricsConfig

    cfg = ValidationMetricsConfig.from_list(["psnr", "kid", "hfen"])
    names = [s.name for s in cfg.metrics if s.enabled]
    assert "kid" in names


@requires_torch_fidelity
def test_create_validation_metrics_computer_with_kid():
    """create_validation_metrics_computer handles 'kid' in list config."""
    from spectramr.core.metrics.computer import create_validation_metrics_computer

    computer = create_validation_metrics_computer(config=["psnr", "kid"], device="cpu")
    assert "kid" in computer.get_metric_names()
