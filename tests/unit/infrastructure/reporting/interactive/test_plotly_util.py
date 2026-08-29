"""Tests for the interactive-layer plotly helpers.

Targets ``mriforge.infrastructure.reporting.interactive.plotly_util``: the plotly
probe, the inline-once library script (offline self-containment), the div
serialiser, and the explainability caption.
"""

from __future__ import annotations

import importlib.util

from mriforge.infrastructure.reporting.interactive import plotly_util as pu

_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def test_has_plotly_matches_environment():
    assert pu.has_plotly() is _HAS_PLOTLY


def test_plotlyjs_script_inlines_library_or_none():
    js = pu.plotlyjs_script()
    if _HAS_PLOTLY:
        assert js is not None and js.startswith("<script")
        # It is the actual library, not a CDN <script src=...> fetch.
        assert "src=" not in js.split(">", 1)[0]
        assert "Plotly" in js
    else:
        assert js is None


def test_caption_escapes_and_wraps():
    html = pu.caption("compare <a> & <b>")
    assert "How to read this" in html
    assert "&lt;a&gt;" in html and "&amp;" in html
    assert "class='howto'" in html


def test_to_div_has_no_external_plotly_fetch():
    if not _HAS_PLOTLY:
        return
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=[0, 1], y=[1, 0]))
    div = pu.to_div(fig, div_id="unit-div")
    assert "unit-div" in div
    # include_plotlyjs=False -> the div must not fetch plotly from a CDN.
    assert 'src="https://cdn' not in div
    assert "cdn.plot.ly/plotly" not in div


def test_flag_colour_is_okabe_ito_vermillion():
    assert pu.flag_colour() == "#D55E00"
