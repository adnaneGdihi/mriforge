r"""Table builders for the reporting pipeline.

Each builder consumes the long-format aggregator DataFrame and emits
both Markdown and LaTeX (IEEE-formatted) renderings.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

TableFn = Callable[..., dict[str, Path] | None]
TABLES: dict[str, TableFn] = {}


def register(table_id: str, fn: TableFn) -> None:
    if table_id in TABLES:
        raise ValueError(f"table '{table_id}' is already registered")
    TABLES[table_id] = fn


def get(table_id: str) -> TableFn:
    if table_id not in TABLES:
        raise KeyError(f"unknown table '{table_id}'. Available: {sorted(TABLES)}")
    return TABLES[table_id]


def list_available() -> list[str]:
    return sorted(TABLES)


from . import (  # noqa: E402
    ablation_table,
    dataset_descriptor,
    hyperparameter_table,
    main_results,
    run_summary,
)

register("tab_2_1_main_results", main_results.make)
register("tab_2_2_ablation", ablation_table.make)
register("tab_2_3_hyperparameters", hyperparameter_table.make)
register("tab_2_4_dataset_descriptor", dataset_descriptor.make)
# The tables-only floor: the one table a single training run can always fill.
# Every table above needs either several methods or a cohort descriptor, so all
# of them return None on a plain run — which is why the default report used to
# emit nothing at all from a run dir full of metrics.
register("tab_run_summary", run_summary.make)


def dispatch(
    df: pd.DataFrame, table_ids: list[str], out_dir: str | Path, **kwargs: Any
) -> dict[str, dict[str, Path] | None]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Path] | None] = {}
    for tid in table_ids:
        try:
            results[tid] = get(tid)(df, out_dir, **kwargs)
        except KeyError:
            results[tid] = None
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("table '%s' failed: %s", tid, exc)
            results[tid] = None
    return results


def _render_markdown_table(df: pd.DataFrame, *, floatfmt: str = ".3f", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Avoids the optional ``tabulate`` dep that ``df.to_markdown`` requires —
    this repo's cluster Docker image does not ship it and triple_emit
    must work offline.
    """
    work = df.reset_index(drop=not index) if not index else df.reset_index()
    cols = list(work.columns)

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:{floatfmt}}"
        return str(v)

    rows = [[_fmt(v) for v in row] for row in work.itertuples(index=False, name=None)]
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, separator, body]) + "\n"


def triple_emit(
    df: pd.DataFrame,
    out_dir: str | Path,
    stem: str,
    *,
    caption: str | None = None,
    label: str | None = None,
    floatfmt: str = ".3f",
    index: bool = False,
) -> dict[str, Path]:
    """Emit a DataFrame as ``.tex`` + ``.md`` + ``.csv`` in one call.

    The central Phase 3.1 deliverable from
    ``TODO/backlog_beautify_logging_and_reporting.md``. Each format
    serves a distinct audience:

    - **LaTeX** (``.tex``) — publication path; booktabs-style with
      ``\\caption{}`` / ``\\label{}`` already wrapped.
    - **Markdown** (``.md``) — renders natively in PR reviews and
      GitHub previews.
    - **CSV** (``.csv``) — machine-readable for follow-up analysis
      without re-parsing LaTeX.

    Args:
        df: Long-format DataFrame to render.
        out_dir: Directory for the three output files.
        stem: Filename stem (extensions appended). E.g. ``"main_results"``
            produces ``main_results.{tex,md,csv}``.
        caption: Optional LaTeX caption (wrapped in ``\\caption{...}``).
        label: Optional LaTeX label (e.g. ``"tab:main_results"``).
        floatfmt: Format spec for numeric cells in the markdown
            rendering. Passed through to ``df.to_markdown``.
        index: Whether to include the DataFrame index in the output.

    Returns:
        Mapping ``{"tex": Path, "md": Path, "csv": Path}`` with the
        three written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=index)

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(_render_markdown_table(df, floatfmt=floatfmt, index=index))

    tex_path = out_dir / f"{stem}.tex"
    tex_lines: list[str] = [r"\begin{table}[t]\centering"]
    if caption is not None:
        tex_lines.append(rf"\caption{{{caption}}}")
    if label is not None:
        tex_lines.append(rf"\label{{{label}}}")
    tex_lines.append(df.to_latex(index=index, float_format=lambda v: f"{v:{floatfmt}}").strip())
    tex_lines.append(r"\end{table}")
    tex_path.write_text("\n".join(tex_lines) + "\n")

    return {"tex": tex_path, "md": md_path, "csv": csv_path}


def emit_table_csv(
    df: pd.DataFrame,
    out_dir: str | Path,
    stem: str,
    *,
    index: bool = False,
) -> Path:
    """Write the machine-readable CSV sink for a bespoke-formatted table.

    The publication tables (``main_results``, ``ablation_table``,
    ``hyperparameter_table``) render rich Markdown + LaTeX in place
    (bold best-marking, mean±std cells, significance superscripts) that
    :func:`triple_emit` cannot reproduce from a plain DataFrame. They
    were therefore missing the third audience sink: a **machine-readable
    CSV** of raw values for downstream analysis without re-parsing the
    LaTeX. This helper supplies exactly that one sink so every table
    emits the full ``{tex, md, csv}`` triple (Phase 3.1 intent of
    ``backlog_beautify_logging_and_reporting.md``) without duplicating
    the rich-formatting logic.

    Args:
        df: Raw-value DataFrame (no markup) to serialise.
        out_dir: Output directory (created if absent).
        stem: Filename stem; ``.csv`` is appended.
        index: Whether to write the DataFrame index.

    Returns:
        Path to the written ``.csv`` file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=index)
    return csv_path
