"""Tests for the v6.3 reporting style helpers: styled_figure, caption_for.

The ProvenanceManifest and ReaderStudyReport cases that used to live here went
with their modules: `reporting/provenance_manifest.py` and
`reporting/reader_study.py` had no caller anywhere outside these tests, so they
were surfaces kept alive by their own coverage (#710)."""

from __future__ import annotations

from mriforge.infrastructure.reporting.style import (
    add_metadata_footer,
    caption_for,
    styled_figure,
)


def test_styled_figure_decorator_calls_use_default_style(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        "mriforge.infrastructure.reporting.style.use_default_style",
        lambda *a, **k: called.append(True),
    )

    @styled_figure
    def make_figure():
        return "ok"

    assert make_figure() == "ok"
    assert called == [True]


def test_caption_for_known_template() -> None:
    s = caption_for("headline_pareto", metric="PSNR", cost="params (M)", n_test=200)
    assert "PSNR" in s
    assert "200" in s


def test_caption_for_unknown_id_returns_empty() -> None:
    assert caption_for("not_a_real_id") == ""


def test_caption_for_missing_kwarg_does_not_crash() -> None:
    s = caption_for("headline_pareto")  # missing metric / cost / n_test
    assert "missing key" in s
