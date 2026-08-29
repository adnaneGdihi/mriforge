r"""Shared helpers for the interactive (plotly) reporting layer.

Single source of truth for: probing whether plotly is installed, inlining the
plotly.js library **once** per report (so the HTML is self-contained and works
offline — unlike the old ``include_plotlyjs="cdn"`` path), the dark theme that
matches the QC report CSS, the image colour scales, and the "how to read this"
caption block that makes every interactive panel explainable.

All plotly imports are guarded and happen inside functions, so importing this
module never fails when plotly is absent — the builders simply return ``None``
and the HTML assembler falls back to the static PNG.
"""

from __future__ import annotations

import html

# Report panel background (matches ``_CSS`` --panel in html_report.py).
_PANEL_BG = "#1c2029"
_FG = "#e6e9ef"
_FLAG = "#D55E00"

# Greyscale for magnitude images (black background -> white signal, the MRI
# display convention); Magma for error maps (dark = low error).
GRAY_SCALE = [[0.0, "#000000"], [1.0, "#ffffff"]]
ERROR_SCALE = "Magma"


def has_plotly() -> bool:
    """Return True when plotly is importable (the interactive layer is available)."""
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        return False
    return True


def plotlyjs_script() -> str | None:
    """Return a ``<script>`` tag with the plotly.js library inlined, or None.

    Emitted **once** at the top of the report; every div is then rendered with
    ``include_plotlyjs=False`` (see :func:`to_div`), so the whole report needs
    no network access to be interactive.
    """
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        return None
    return f"<script type='text/javascript'>{get_plotlyjs()}</script>"


def to_div(fig, *, div_id: str | None = None) -> str:
    """Serialise a plotly figure to an inline ``<div>`` (no embedded plotly.js)."""
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id)


def apply_dark(fig, *, title: str | None = None, height: int | None = None) -> None:
    """Apply the report's dark theme to a plotly figure in place."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_PANEL_BG,
        plot_bgcolor=_PANEL_BG,
        font={"color": _FG, "size": 12},
        margin={"l": 70, "r": 20, "t": 44 if title else 20, "b": 40},
    )
    if title is not None:
        fig.update_layout(title={"text": title, "font": {"size": 14}})
    if height is not None:
        fig.update_layout(height=height)


def image_axes(fig, *, n_cols: int = 1) -> None:
    """Configure axes for image heatmaps: square pixels, row 0 at the top."""
    for i in range(1, n_cols + 1):
        xa = "xaxis" if i == 1 else f"xaxis{i}"
        ya = "yaxis" if i == 1 else f"yaxis{i}"
        fig.update_layout(
            {
                xa: {
                    "visible": False,
                    "scaleanchor": ya.replace("axis", ""),
                    "constrain": "domain",
                },
                ya: {"visible": False, "autorange": "reversed", "constrain": "domain"},
            }
        )


def caption(text: str) -> str:
    """Render a 'How to read this' explainability caption as an HTML block."""
    return f"<div class='howto'><b>How to read this.</b> {html.escape(text)}</div>"


def flag_colour() -> str:
    """Okabe-Ito vermillion used to flag outliers / departures (report SSOT)."""
    return _FLAG
