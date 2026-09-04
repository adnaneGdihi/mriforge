r"""Assemble a self-contained QC HTML report.

Bundles the rendered figures (base64-embedded PNGs, so the file is portable),
an interactive group-IQM section, per-subject reportlets and table links into a
single ``qc_report.html``. Interactive group plots use :mod:`plotly` when it
is installed (``pip install spectramr[viz]``) and fall back to the static
``qc_group_strip`` PNG otherwise — no hard dependency on plotly.

Pure-Python assembly (``html`` + ``base64``): no template engine, so nothing can
drift out of sync with the code and the report stays a single file.
"""

from __future__ import annotations

import base64
import html
import logging
from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata

logger = logging.getLogger(__name__)

# QC figures are surfaced first, each with a human caption.
_FEATURED = [
    ("qc_group_strip", "Group IQM distribution"),
    ("qc_subject_mosaic", "Per-subject QC mosaic"),
    ("qc_carpet", "Carpet + spike diagnostics"),
]

_CSS = """
:root { --bg:#12151c; --panel:#1c2029; --fg:#e6e9ef; --muted:#8a93a6;
        --accent:#56B4E9; --flag:#D55E00; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
header { padding:20px 28px; background:var(--panel); border-bottom:2px solid #000; }
header h1 { margin:0 0 6px; font-size:20px; }
.meta { color:var(--muted); font-size:12px; }
main { max-width:1100px; margin:0 auto; padding:20px 28px 60px; }
section { background:var(--panel); border-radius:8px; padding:16px 20px;
          margin:18px 0; }
section h2 { margin:0 0 10px; font-size:15px; border-bottom:1px solid #2a2f3a;
             padding-bottom:6px; }
img.fig { max-width:100%; height:auto; background:#fff; border-radius:4px;
          display:block; margin:6px 0; }
figure { margin:0 0 14px; }
figcaption { color:var(--muted); font-size:11px; margin-top:4px; }
.howto { color:var(--muted); font-size:12px; line-height:1.5; margin:8px 0 2px;
         border-left:3px solid var(--accent); padding:4px 10px; background:#191d26; }
.howto b { color:var(--fg); }
a { color:var(--accent); }
table.tbl { border-collapse:collapse; font-size:12px; width:100%; }
table.tbl th, table.tbl td { border:1px solid #2a2f3a; padding:4px 8px;
                             text-align:right; }
table.tbl th { background:#232834; }
.empty { color:var(--muted); font-style:italic; }
footer { color:var(--muted); font-size:11px; text-align:center; padding:20px; }
"""


def _b64_png(path: str | Path) -> str | None:
    """Return a base64 ``<img>`` payload for a figure's PNG twin, or None."""
    p = Path(path)
    png = p if p.suffix == ".png" else p.with_suffix(".png")
    if not png.exists():
        return None
    data = base64.b64encode(png.read_bytes()).decode("ascii")
    return f'<img class="fig" alt="{html.escape(png.stem)}" src="data:image/png;base64,{data}" />'


def _section(title: str, inner: str) -> str:
    return f"<section><h2>{html.escape(title)}</h2>{inner}</section>"


def _static_fallback(title: str, fig_id: str, figures: dict, *, note: bool = True) -> str | None:
    """Embed a figure's static PNG as a fallback for an interactive section."""
    p = figures.get(fig_id)
    img = _b64_png(p) if p is not None else None
    if not img:
        return None
    note_html = (
        (
            "<div class='meta'>Install <code>plotly</code> "
            "(<code>pip install spectramr[viz]</code>) for the interactive "
            "version.</div>"
        )
        if note
        else ""
    )
    return _section(title, f"<figure>{img}</figure>{note_html}")


def _header(metadata: RunMetadata | None, method: str, task: str) -> str:
    bits = [f"task: {html.escape(str(task))}"]
    if metadata is not None:
        dirty = "*" if metadata.git_dirty else ""
        bits += [
            f"git: {html.escape(metadata.git_commit[:10])}{dirty}",
            f"seed: {metadata.seed}",
            f"data: {html.escape(str(metadata.dataset_version))}",
            f"generated: {html.escape(str(metadata.timestamp_utc))}",
        ]
    return (
        f"<header><h1>QC report — {html.escape(method)}</h1>"
        f"<div class='meta'>{' · '.join(bits)}</div></header>"
    )


def _tables_section(tables: dict | None, out_dir: Path) -> str:
    if not tables:
        return ""
    rows = []
    for tid, group in tables.items():
        if not group:
            continue
        links = " · ".join(
            f"<a href='{html.escape(str(Path(p).relative_to(out_dir)))}'>"
            f"{html.escape(Path(p).name)}</a>"
            for p in group.values()
        )
        rows.append(f"<li><b>{html.escape(tid)}</b>: {links}</li>")
    if not rows:
        return ""
    return "<section><h2>Tables</h2><ul>" + "".join(rows) + "</ul></section>"


def _interactive_sections(
    *,
    per_call_df,
    predictions_df,
    aggregated_df,
    cases,
    figures,
    shown,
) -> tuple[list[str], bool]:
    """Build the plotly interactive sections (with static-PNG fallbacks).

    Returns ``(sections, used_plotly)``. Each entry is a full ``<section>``;
    ``used_plotly`` is True when at least one genuinely interactive div was
    emitted (so the assembler injects plotly.js once).
    """
    from . import interactive as itv

    sections: list[str] = []
    used = False

    def add(title: str, div: str | None, fallback_fid: str | None) -> None:
        nonlocal used
        if div is not None:
            sections.append(_section(title, div))
            used = True
            if fallback_fid:
                shown.add(fallback_fid)
            return
        if fallback_fid:
            fb = _static_fallback(title, fallback_fid, figures)
            if fb is not None:
                sections.append(fb)
                shown.add(fallback_fid)

    add(
        "Group IQM distribution",
        itv.group_iqm_div(per_call_df=per_call_df, predictions_df=predictions_df),
        "qc_group_strip",
    )
    add("Per-subject viewer (2-D)", itv.subject_viewer_2d_div(cases), "qc_subject_mosaic")
    add("Volume viewer (3-D)", itv.volume_viewer_3d_div(cases), None)
    add("Training dynamics", itv.learning_curve_div(aggregated_df), "fig_1_2_learning_curves")
    add(
        "Per-case metric distribution",
        itv.metric_distribution_div(per_call_df=per_call_df, predictions_df=predictions_df),
        "metric_distribution",
    )
    add(
        "Method agreement (Bland-Altman)",
        itv.comparison_div(predictions_df=predictions_df),
        "bland_altman",
    )
    return sections, used


def build_html_report(
    out_dir: str | Path,
    *,
    figures: dict,
    tables: dict | None = None,
    metadata: RunMetadata | None = None,
    per_call_df: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    aggregated_df: pd.DataFrame | None = None,
    cases: list | None = None,
    interactive: bool = True,
    method: str = "report",
    task: str = "default",
    filename: str = "qc_report.html",
) -> Path | None:
    """Write ``<out_dir>/<filename>`` and return its path (None if nothing to show).

    Args:
        figures: ``{fig_id: Path | None}`` from the plotter dispatch.
        tables: ``{table_id: {ext: Path}}`` (optional; rendered as links).
        metadata: run provenance for the header.
        per_call_df: per-case metric frame (interactive group / distribution plots).
        predictions_df: per-case long frame (distribution / Bland-Altman).
        aggregated_df: tidy metric frame (interactive learning curves).
        cases: recorded image cases (2-D + optional ``*_volume`` for the viewers).
        interactive: when True and plotly is installed, emit interactive plotly
            sections; otherwise every section falls back to the embedded PNG.

    The report is a single self-contained file: static figures are base64 PNGs
    and plotly.js is inlined once, so it renders fully offline (no CDN).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = figures or {}
    sections: list[str] = []  # real content sections only (drives the skip guard)
    shown = {"qc_group_strip"}
    used_plotly = False

    if interactive:
        interactive_sections, used_plotly = _interactive_sections(
            per_call_df=per_call_df,
            predictions_df=predictions_df,
            aggregated_df=aggregated_df,
            cases=cases,
            figures=figures,
            shown=shown,
        )
        sections.extend(interactive_sections)
    else:
        # No interactive layer requested: still surface the group strip statically.
        fb = _static_fallback("Group IQM distribution", "qc_group_strip", figures, note=False)
        if fb is not None:
            sections.append(fb)

    # Featured static QC reportlets not already shown (carpet, and mosaic when
    # the 2-D viewer fell back / was skipped).
    for fid, cap in _FEATURED:
        if fid in shown:
            continue
        p = figures.get(fid)
        img = _b64_png(p) if p is not None else None
        if img:
            sections.append(_section(cap, f"<figure>{img}</figure>"))
        shown.add(fid)

    # Remaining figures gallery.
    gallery = []
    for fid, p in figures.items():
        if fid in shown or p is None:
            continue
        img = _b64_png(p)
        if img:
            gallery.append(f"<figure>{img}<figcaption>{html.escape(fid)}</figcaption></figure>")
    if gallery:
        sections.append("<section><h2>Figures</h2>" + "".join(gallery) + "</section>")

    tables_html = _tables_section(tables, out_dir)
    if tables_html:
        sections.append(tables_html)

    if not sections:
        logger.info("html report: no figures to embed; skipping")
        return None

    # Inline plotly.js once (self-contained, offline) when interactive divs exist.
    js = ""
    if used_plotly:
        from .interactive import plotlyjs_script

        js = plotlyjs_script() or ""

    body = (
        "<main>" + js + "".join(sections) + "</main><footer>Generated by spectramr report · "
        "QC</footer>"
    )
    doc = (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>QC report — {html.escape(method)}</title>"
        f"<style>{_CSS}</style></head><body>"
        + _header(metadata, method, task)
        + body
        + "</body></html>"
    )
    html_path = out_dir / filename
    html_path.write_text(doc, encoding="utf-8")
    logger.info("html report: wrote %s", html_path)
    return html_path
