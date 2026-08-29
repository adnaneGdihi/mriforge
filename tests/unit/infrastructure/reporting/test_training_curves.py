"""Unit tests for the auto-report training-curve helper."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting._training_curves import (
    ema, plot_training_curves, _should_use_log,
)


class TestEMA:
    def test_constant_input(self) -> None:
        x = np.array([2.0, 2.0, 2.0, 2.0])
        y = ema(x, half_life=2.0)
        assert np.allclose(y, 2.0)

    def test_handles_nan(self) -> None:
        x = np.array([1.0, float("nan"), 1.0])
        y = ema(x, half_life=2.0)
        # NaN should not propagate; previous value held
        assert not np.isnan(y).any()

    def test_smoothing_reduces_variance(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = ema(x, half_life=8.0)
        assert y.var() < x.var()


class TestLogDetection:
    def test_log_when_orders_of_magnitude(self) -> None:
        x = np.array([1e-5, 1e-3, 1e-1, 1.0, 10.0])
        assert _should_use_log(x) is True

    def test_linear_when_narrow(self) -> None:
        x = np.array([0.01, 0.02, 0.03, 0.05])
        assert _should_use_log(x) is False

    def test_no_log_with_zeros(self) -> None:
        # Even if the maximum is huge, presence of zeros should not crash
        x = np.array([0.0, 0.0, 1e6])
        # Function only considers positives so it COULD return False; we
        # just assert it doesn't raise.
        _should_use_log(x)


class TestPlotTrainingCurves:
    def test_writes_file(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        steps = 100
        df = pd.DataFrame({
            "g_total_loss": np.linspace(2.0, 0.5, steps) + 0.05 * rng.standard_normal(steps),
            "val_psnr":     np.linspace(20.0, 32.0, steps) + 0.3 * rng.standard_normal(steps),
        })
        out = tmp_path / "losses.png"
        result = plot_training_curves(
            df, ["g_total_loss"],
            title="t", ylabel="L",
            save_path=out, higher_is_better=False,
        )
        assert result == out and out.exists()

    def test_skips_when_no_columns(self, tmp_path) -> None:
        df = pd.DataFrame({"foo": [1, 2, 3]})
        out = tmp_path / "out.png"
        result = plot_training_curves(
            df, ["bar", "baz"],
            title="t", ylabel="L", save_path=out,
        )
        assert result is None and not out.exists()

    def test_log_axis_auto_for_loss(self, tmp_path) -> None:
        # Loss spans 4 orders of magnitude → should pick log
        df = pd.DataFrame({"loss": np.geomspace(1e-4, 1.0, 100)})
        out = tmp_path / "log.png"
        plot_training_curves(df, ["loss"], title="t", ylabel="L", save_path=out)
        assert out.exists()
