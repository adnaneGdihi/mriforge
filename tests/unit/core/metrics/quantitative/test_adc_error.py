"""Tests for the ADC MAE metric (audit 2026-07 I2)."""
from __future__ import annotations

import math

import torch

from mriforge.core.metrics.quantitative.adc_error import ADCMeanAbsoluteError


def test_zero_error_for_identical_maps() -> None:
    m = ADCMeanAbsoluteError()
    adc = torch.rand(1, 1, 4, 4) * 3e-3
    assert m(adc, adc.clone()) == 0.0


def test_mae_value() -> None:
    m = ADCMeanAbsoluteError()
    pred = torch.full((1, 1, 2, 2), 1.5e-3)
    ref = torch.full((1, 1, 2, 2), 1.0e-3)
    assert abs(m(pred, ref) - 0.5e-3) < 1e-9


def test_complex_input_returns_nan_not_a_grade() -> None:
    m = ADCMeanAbsoluteError()
    cplx = torch.complex(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4))
    assert math.isnan(m(cplx, torch.rand(1, 1, 4, 4)))


def test_shape_mismatch_returns_nan() -> None:
    m = ADCMeanAbsoluteError()
    assert math.isnan(m(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 8, 8)))


def test_lower_is_better() -> None:
    assert ADCMeanAbsoluteError().higher_is_better is False


def test_registered() -> None:
    from mriforge.core.metrics.registry import MetricsRegistry

    assert MetricsRegistry.is_registered("adc_mae")
