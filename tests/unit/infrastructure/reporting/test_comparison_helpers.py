"""Unit tests for the shared reporting comparison helpers.

Pin three contracts:

1. ``l1`` / ``psnr`` / ``ssim_simple`` return correct numerics on
   simple cases.
2. ``add_residual_panel`` returns a non-zero ``abs_max`` for a
   non-degenerate residual and writes a colorbar to the figure.
3. ``add_residual_histogram`` annotation includes L1 / bias / σ.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mriforge.infrastructure.reporting.plotters._comparison_helpers import (
    add_image_panel,
    add_residual_histogram,
    add_residual_panel,
    _format_metric_str,
    l1,
    psnr,
    ssim_simple,
)


class TestNumericHelpers:
    def test_l1_zero_for_identical(self) -> None:
        x = np.random.RandomState(0).rand(8, 8)
        assert l1(x, x) == 0.0

    def test_psnr_inf_for_identical(self) -> None:
        x = np.random.RandomState(0).rand(8, 8)
        assert psnr(x, x) == float("inf")

    def test_psnr_decreases_with_noise(self) -> None:
        rng = np.random.RandomState(0)
        ref = rng.rand(16, 16).astype(np.float32)
        a = ref + 0.01 * rng.randn(16, 16)
        b = ref + 0.10 * rng.randn(16, 16)
        assert psnr(a, ref) > psnr(b, ref)

    def test_ssim_one_for_identical(self) -> None:
        x = np.random.RandomState(0).rand(8, 8)
        assert abs(ssim_simple(x, x) - 1.0) < 1e-6


class TestMetricFormatter:
    def test_multiline_default(self) -> None:
        x = np.random.RandomState(0).rand(8, 8)
        s = _format_metric_str(x + 0.1, x)
        assert "\n" in s
        assert "PSNR" in s and "SSIM" in s and "L1" in s

    def test_singleline(self) -> None:
        x = np.random.RandomState(0).rand(8, 8)
        s = _format_metric_str(x + 0.1, x, multiline=False)
        assert "\n" not in s
        assert "PSNR" in s and "SSIM" in s and "L1" in s

    def test_handles_none(self) -> None:
        assert _format_metric_str(None, None) == ""

    def test_only_l1(self) -> None:
        x = np.random.RandomState(0).rand(8, 8)
        s = _format_metric_str(x + 0.1, x, show=("l1",))
        assert s.startswith("L1")
        assert "PSNR" not in s and "SSIM" not in s


class TestPanelBuilders:
    def test_residual_panel_returns_abs_max(self) -> None:
        rng = np.random.RandomState(0)
        ref = rng.rand(16, 16).astype(np.float32)
        pred = ref + 0.05 * rng.randn(16, 16)
        fig, ax = plt.subplots(figsize=(2, 2))
        abs_max, residual = add_residual_panel(
            ax, pred, ref, title="t", fig=fig,
        )
        assert abs_max > 0
        assert residual.shape == ref.shape
        plt.close(fig)

    def test_residual_panel_handles_zero_residual(self) -> None:
        # When pred == ref the residual is identically zero
        x = np.zeros((8, 8))
        fig, ax = plt.subplots(figsize=(2, 2))
        abs_max, residual = add_residual_panel(ax, x, x, fig=fig)
        # Should not raise; abs_max defaults to a tiny floor
        assert abs_max > 0
        assert (residual == 0).all()
        plt.close(fig)

    def test_residual_histogram_renders(self) -> None:
        rng = np.random.RandomState(0)
        residual = rng.randn(32, 32) * 0.1
        fig, ax = plt.subplots(figsize=(2, 2))
        add_residual_histogram(ax, residual, title="hist")
        # The annotation should be present in the axes children
        text_objs = [t for t in ax.texts]
        assert len(text_objs) == 1
        annotation = text_objs[0].get_text()
        assert "L1" in annotation and "bias" in annotation and "σ" in annotation
        plt.close(fig)

    def test_image_panel_with_metric(self) -> None:
        rng = np.random.RandomState(0)
        x = rng.rand(8, 8)
        fig, ax = plt.subplots(figsize=(2, 2))
        add_image_panel(ax, x, title="img", metric_str="PSNR 30")
        text_objs = [t for t in ax.texts]
        assert any("PSNR 30" in t.get_text() for t in text_objs)
        plt.close(fig)


class TestPlottersImportable:
    def test_sr_synthesis_kspace_failure_importable(self) -> None:
        from mriforge.infrastructure.reporting.plotters.failure_gallery import make as fg
        from mriforge.infrastructure.reporting.plotters.mri_specific.kspace_recon import make as ksp
        from mriforge.infrastructure.reporting.plotters.mri_specific.sr_triptych import make as sr
        from mriforge.infrastructure.reporting.plotters.mri_specific.synthesis_c2c import make as c2c
        for fn in (sr, c2c, ksp, fg):
            assert callable(fn)

    def test_sr_renders_on_synthetic(self, tmp_path) -> None:
        import pandas as pd
        from mriforge.infrastructure.reporting.plotters.mri_specific.sr_triptych import make
        rng = np.random.RandomState(0)
        cases = []
        for i in range(2):
            ref = rng.rand(32, 32).astype(np.float32)
            pred = ref + 0.05 * rng.randn(32, 32).astype(np.float32)
            lr = ref + 0.10 * rng.randn(32, 32).astype(np.float32)
            cases.append({"name": f"sub-{i}", "lr": lr, "pred": pred, "hr": ref})
        out = tmp_path / "sr.png"
        result = make(pd.DataFrame(), out, cases=cases)
        assert result is not None and result.exists()
