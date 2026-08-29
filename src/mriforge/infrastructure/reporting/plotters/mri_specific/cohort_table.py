r"""Figure A.11 — Cohort descriptor visual + statement.

Rendered as a one-row figure containing a bar of split sizes and a
text panel with summary statistics. The structured *table* form is in
``mriforge.infrastructure.reporting.tables.dataset_descriptor``; this is the
graphical companion.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mriforge.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from mriforge.infrastructure.reporting.style import save_figure, use_default_style


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cohort: dict | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the cohort visual.

    Args:
        cohort: dict containing at least ``{train_n, val_n, test_n}``;
            optional ``age_mean``, ``age_std``, ``sex_split``, ``scanners``.
    """
    if not cohort:
        return None
    out_path = Path(out_path)

    use_default_style()
    fig, (ax_bar, ax_txt) = plt.subplots(
        1, 2, figsize=(5.4, 1.8), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    splits = [
        ("train", cohort.get("train_n", 0)),
        ("val", cohort.get("val_n", 0)),
        ("test", cohort.get("test_n", 0)),
    ]
    ax_bar.bar(
        [s for s, _ in splits],
        [n for _, n in splits],
        color=["#0072B2", "#56B4E9", "#D55E00"],
        alpha=0.85,
    )
    for x, (_, n) in enumerate(splits):
        ax_bar.text(x, n, str(n), ha="center", va="bottom", fontsize=7)
    ax_bar.set_ylabel("subjects (n)")
    ax_bar.set_title("split sizes (subject-level)", fontsize=8)

    lines = [
        f"Total: {sum(n for _, n in splits)}",
        f"Age: {cohort.get('age_mean', '?'):.1f} ± {cohort.get('age_std', '?'):.1f}"
        if isinstance(cohort.get("age_mean"), (int, float))
        else "Age: ?",
        f"Sex: {cohort.get('sex_split', '?')}",
        f"Scanners: {cohort.get('scanners', '?')}",
        f"Pathology: {cohort.get('pathology', '?')}",
        f"Split rule: {cohort.get('split_rule', 'subject-level (no slice leakage)')}",
    ]
    ax_txt.set_axis_off()
    ax_txt.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=7, family="monospace")

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
