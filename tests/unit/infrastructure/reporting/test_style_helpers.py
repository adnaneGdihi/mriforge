"""Tests for the beautify-backlog Phase 2 style helpers.

Locks ``TODO/backlog_beautify_logging_and_reporting.md`` items 2.5 and
2.6: every plotter has a single canonical call for attaching a
colorbar and for emitting both a PNG and a PDF.

The matplotlib backend is forced to ``Agg`` so these tests pass on
headless CI without a display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture()
def sample_axes_image() -> tuple:
    """Build a tiny figure with one ``imshow`` for the colorbar tests."""
    fig, ax = plt.subplots(figsize=(2, 2))
    img = ax.imshow(np.random.rand(8, 8), vmin=-0.1, vmax=1.2)
    yield fig, ax, img
    plt.close(fig)


# ---------------------------------------------------------------------
# 2.5 — attach_colorbar
# ---------------------------------------------------------------------


def test_attach_colorbar_returns_colorbar_instance(sample_axes_image) -> None:
    from matplotlib.colorbar import Colorbar

    from mriforge.infrastructure.reporting.style import attach_colorbar

    fig, ax, im = sample_axes_image
    cbar = attach_colorbar(im, ax, label="PSNR", units="dB")
    assert isinstance(cbar, Colorbar)


def test_attach_colorbar_label_includes_units(sample_axes_image) -> None:
    from mriforge.infrastructure.reporting.style import attach_colorbar

    fig, ax, im = sample_axes_image
    cbar = attach_colorbar(im, ax, label="SSIM")
    assert cbar.ax.get_ylabel() == "SSIM"

    cbar2 = attach_colorbar(im, ax, label="PSNR", units="dB")
    assert "[dB]" in cbar2.ax.get_ylabel()


def test_attach_colorbar_zero_floor_clamps_vmin(sample_axes_image) -> None:
    """``vmin_zero_floor=True`` rewrites a negative ``vmin`` to zero."""
    from mriforge.infrastructure.reporting.style import attach_colorbar

    fig, ax, im = sample_axes_image
    # Sample image was constructed with vmin=-0.1.
    attach_colorbar(im, ax, label="SSIM", vmin_zero_floor=True)
    vmin, _ = im.get_clim()
    assert vmin == 0.0


def test_attach_colorbar_attaches_a_locator(sample_axes_image) -> None:
    """``num_ticks`` configures a ``MaxNLocator`` (exact tick count is data-dependent)."""
    import matplotlib.ticker as mticker

    from mriforge.infrastructure.reporting.style import attach_colorbar

    fig, ax, im = sample_axes_image
    cbar = attach_colorbar(im, ax, label="value", num_ticks=3)
    # The locator must be a MaxNLocator (matplotlib chooses round
    # numbers, so the exact tick count varies by data range).
    assert isinstance(cbar.locator, mticker.MaxNLocator)


# ---------------------------------------------------------------------
# 2.6 — save_figure (PNG + PDF dual emit)
# ---------------------------------------------------------------------


def test_save_figure_writes_both_png_and_pdf(tmp_path: Path) -> None:
    from mriforge.infrastructure.reporting.style import save_figure

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1, 2], [0, 1, 4])
    paths = save_figure(fig, tmp_path, stem="line_demo")
    assert "png" in paths and "pdf" in paths
    assert paths["png"].exists()
    assert paths["pdf"].exists()
    assert paths["png"].name == "line_demo.png"
    assert paths["pdf"].name == "line_demo.pdf"


def test_save_figure_closes_figure_by_default(tmp_path: Path) -> None:
    """After ``save_figure`` the figure object should be closed (released)."""
    from mriforge.infrastructure.reporting.style import save_figure

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    fignum = fig.number
    save_figure(fig, tmp_path, stem="x")
    # Closed figures are no longer registered with pyplot.
    assert fignum not in plt.get_fignums()


def test_save_figure_can_skip_pdf(tmp_path: Path) -> None:
    from mriforge.infrastructure.reporting.style import save_figure

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [1, 0])
    paths = save_figure(fig, tmp_path, stem="png_only", formats=("png",))
    assert "png" in paths
    assert "pdf" not in paths
    assert not (tmp_path / "png_only.pdf").exists()


def test_save_figure_can_skip_png(tmp_path: Path) -> None:
    from mriforge.infrastructure.reporting.style import save_figure

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [1, 0])
    paths = save_figure(fig, tmp_path, stem="pdf_only", formats=("pdf",))
    assert "pdf" in paths
    assert "png" not in paths


def test_save_figure_creates_nested_directories(tmp_path: Path) -> None:
    from mriforge.infrastructure.reporting.style import save_figure

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [1, 0])
    target = tmp_path / "deep" / "nested" / "subdir"
    save_figure(fig, target, stem="x")
    assert target.is_dir()


# ---------------------------------------------------------------------
# Control-character sanitisation (regression: "Glyph 127/128 missing
# from font(s) DejaVu Sans" UserWarning raised by ``fig.savefig`` when a
# data-derived label carries C0/C1 control bytes — e.g. a corrupt arm
# name flowing into ``suptitle``). See style.save_figure.
# ---------------------------------------------------------------------


def test_sanitize_text_for_font_strips_control_chars() -> None:
    from mriforge.infrastructure.reporting.style import sanitize_text_for_font

    # U+007F (DEL) and U+0080 (a C1 control) are the exact glyphs from the
    # reported warning; other C0 controls must go too.
    assert sanitize_text_for_font("val\x7f_ssim\x80") == "val_ssim"
    assert sanitize_text_for_font("a\x00\x01\x1f\x9fb") == "ab"


def test_sanitize_text_for_font_keeps_newlines_tabs_and_unicode() -> None:
    from mriforge.infrastructure.reporting.style import sanitize_text_for_font

    # Legitimate layout whitespace and renderable unicode (µ, ±, en-dash)
    # must survive untouched.
    assert sanitize_text_for_font("line1\nline2\tcol") == "line1\nline2\tcol"
    assert sanitize_text_for_font("mean ± σ – µV") == "mean ± σ – µV"  # noqa: RUF001


def test_sanitize_figure_text_rewrites_all_text_artists() -> None:
    from matplotlib.text import Text

    from mriforge.infrastructure.reporting.style import sanitize_figure_text

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_title("bad\x7ftitle")
    ax.set_xlabel("x\x80label")
    fig.suptitle("arm\x7f\x80name")
    changed = sanitize_figure_text(fig)
    assert changed >= 3
    for t in fig.findobj(Text):
        assert not any(ord(ch) in range(0x7F, 0xA0) for ch in t.get_text())
    plt.close(fig)


def test_save_figure_does_not_warn_on_control_char_titles(tmp_path: Path) -> None:
    """Reproduces the reported warning: a control-char ``suptitle`` must
    not trigger matplotlib's "missing from font" ``UserWarning`` at save."""
    import warnings

    from matplotlib.text import Text

    from mriforge.infrastructure.reporting.style import save_figure

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1, 2], [0, 1, 4])
    fig.suptitle("mrixfields_b17\x7f\x80")  # DEL + C1 control, as in the bug
    ax.set_title("val\x7f_ssim")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_figure(fig, tmp_path, stem="ctrl", formats=("pdf", "png"), close=False)
    font_warnings = [w for w in caught if "missing from font" in str(w.message)]
    assert not font_warnings, [str(w.message) for w in font_warnings]
    # And the artists themselves were normalised (deterministic anchor).
    for t in fig.findobj(Text):
        assert "\x7f" not in t.get_text() and "\x80" not in t.get_text()
    plt.close(fig)
