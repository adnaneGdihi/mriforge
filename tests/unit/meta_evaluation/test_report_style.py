"""Unit tests for the package-local figure-style helpers.

``_report_style`` is the meta-evaluation package's own styling SSOT — it
exists because the layer-direction rule forbids ``src/`` importing the
``scripts/sim2rank/style.py`` module. It centralizes the palette, the
constrained-layout figure factory (which removes the tight_layout +
colorbar warning class), and the shared annotation / sizing / labelling
helpers that were previously inlined or duplicated.
"""

from __future__ import annotations

import warnings

from mriforge.core.metrics.meta_evaluation import _report_style as rs

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


def test_palette_constants_present() -> None:
    for name in ("ACCENT", "BAR", "DIM_GREY", "DEFENSIBLE_GREEN", "FAIL_RED"):
        value = getattr(rs, name)
        assert isinstance(value, str) and value.startswith("#")


# ---------------------------------------------------------------------------
# top_bottom_indices
# ---------------------------------------------------------------------------


def test_top_bottom_indices_basic() -> None:
    top, bottom = rs.top_bottom_indices([0.1, 0.9, 0.5, 0.3, 0.7], k=2)
    assert top == [1, 4]
    assert bottom == [0, 3]


def test_top_bottom_indices_nan_safe() -> None:
    top, bottom = rs.top_bottom_indices([float("nan"), 0.5, 0.1], k=2)
    assert 0 not in top
    assert 0 not in bottom


# ---------------------------------------------------------------------------
# barh_figsize
# ---------------------------------------------------------------------------


def test_barh_figsize_respects_minimum() -> None:
    w, h = rs.barh_figsize(1)
    assert h >= 2.0
    assert w > 0


def test_barh_figsize_grows_with_items() -> None:
    _, h_small = rs.barh_figsize(5)
    _, h_big = rs.barh_figsize(80)
    assert h_big > h_small


# ---------------------------------------------------------------------------
# human_label
# ---------------------------------------------------------------------------


def test_human_label_known_acronyms() -> None:
    assert rs.human_label("psnr") == "PSNR"
    assert rs.human_label("ms_ssim") == "MS-SSIM"
    assert rs.human_label("lpips") == "LPIPS"


def test_human_label_unknown_falls_back_to_titleish() -> None:
    # Unknown keys are made presentable, never dropped.
    out = rs.human_label("some_new_metric")
    assert "some" in out.lower()
    assert "_" not in out  # underscores cleaned up


# ---------------------------------------------------------------------------
# figure() / save_figure()
# ---------------------------------------------------------------------------


def test_figure_returns_fig_and_ax() -> None:
    fig, ax = rs.figure(figsize=(4, 3))
    try:
        assert hasattr(fig, "savefig")
        assert hasattr(ax, "plot")
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_save_figure_writes_nontrivial_png(tmp_path) -> None:
    fig, ax = rs.figure(figsize=(4, 3))
    ax.plot([0, 1, 2], [0, 1, 4])
    path = rs.save_figure(fig, tmp_path / "x.png")
    assert path.exists()
    assert path.stat().st_size > 1024


def test_figure_with_colorbar_is_warning_free(tmp_path) -> None:
    # The whole point of the constrained-layout factory: a figure with a
    # colorbar must save WITHOUT the "not compatible with tight_layout"
    # warning class that the old code triggered.
    import numpy as np

    fig, ax = rs.figure(figsize=(4, 3))
    im = ax.imshow(np.arange(9).reshape(3, 3))
    fig.colorbar(im, ax=ax)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rs.save_figure(fig, tmp_path / "cbar.png")  # must not raise


def test_median_via_top_bottom_handles_all_nan() -> None:
    top, bottom = rs.top_bottom_indices([float("nan"), float("nan")], k=2)
    assert top == []
    assert bottom == []


# ---------------------------------------------------------------------------
# matplotlib optional-dependency guard (6.1). matplotlib is REQUIRED in
# pyproject; the guard keeps the module importable in a degraded env and
# raises a clear error only when a figure is actually built.
# ---------------------------------------------------------------------------


def test_module_exposes_matplotlib_flag() -> None:
    assert isinstance(rs.MATPLOTLIB_AVAILABLE, bool)


def test_figure_raises_clear_error_without_matplotlib(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(rs, "MATPLOTLIB_AVAILABLE", False)
    with pytest.raises(ImportError, match="matplotlib"):
        rs.figure()
    with pytest.raises(ImportError, match="matplotlib"):
        rs.figure_grid(1, 1, figsize=(3, 3))
