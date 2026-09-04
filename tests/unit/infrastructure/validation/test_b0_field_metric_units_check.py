"""Tier-1 audit: the static half of the b0_field_rmse units lock.

If a config requests the absolute-Hz ``b0_field_rmse`` metric, the chosen model
must declare ``output_field_units == "Hz"``. This is the static twin of the
runtime guard (``field_comparability``): together they stop grading a non-Hz /
image output as a B0 field — the bssfp_peak_finder facade (pitfall #16).
"""
from __future__ import annotations

from types import SimpleNamespace

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
from tests.utils.config_block_stub import block_stub


def _cfg(metrics, model_type):
    return SimpleNamespace(
        validation=block_stub("validation", metrics=metrics),
        model=SimpleNamespace(model_type=model_type),
    )


def test_b0_metric_with_hz_model_passes():
    chk = ConfigHealthChecker()
    r = chk.check_b0_field_metric_requires_hz_model(
        _cfg(["field_mae", "b0_field_rmse"], "bssfp_b0_regressor")
    )
    assert r.passed


def test_b0_metric_with_non_hz_model_errors():
    # resnet_unwrap does not declare output_field_units → the facade guard fires.
    chk = ConfigHealthChecker()
    r = chk.check_b0_field_metric_requires_hz_model(
        _cfg(["b0_field_rmse"], "resnet_unwrap")
    )
    assert not r.passed
    assert r.severity == "error"
    assert "Hz" in (r.fix_hint or "") or "Hz" in r.message


def test_no_b0_metric_is_not_applicable():
    chk = ConfigHealthChecker()
    r = chk.check_b0_field_metric_requires_hz_model(_cfg(["psnr"], "resnet_unwrap"))
    assert r.passed


def test_alias_b0_rmse_hz_also_triggers_the_lock():
    chk = ConfigHealthChecker()
    r = chk.check_b0_field_metric_requires_hz_model(
        _cfg(["b0_rmse_hz"], "resnet_unwrap")
    )
    assert not r.passed
    assert r.severity == "error"
