"""Cross-dataset overview figures into ``results/eda/_overview/``.

These are the figures that make "neglect no dataset" verifiable: the landscape map plots
every discovered dataset, and the coverage matrix lists each one with the figures it
produced (or the reason it was skipped).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROLE_MARKER = {"RAW": "o", "MAP": "s", "img": "^", "SIM": "D"}
_STATUS_COLOR = {"downloaded": "#1b7837", "not_downloaded": "#999999", "defective": "#b2182b"}


def _save(fig, out_dir: Path, fname: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def render_overview(entries: list[dict], out_dir: Path) -> list[Path]:
    produced: list[Path] = []

    # 00 landscape: field strength (log) by anatomy, marker = role, color = status
    fig, ax = plt.subplots(figsize=(11, 6))
    anatomies = sorted({(e.get("anatomy") or "?") for e in entries})
    ay = {a: i for i, a in enumerate(anatomies)}
    for e in entries:
        for ft in e.get("field_T") or [None]:
            if ft:
                ax.scatter(
                    ft,
                    ay[e.get("anatomy") or "?"],
                    marker=_ROLE_MARKER.get(e.get("role") or "", "x"),
                    c=_STATUS_COLOR.get(e.get("status") or "", "#555555"),
                    s=60,
                    alpha=0.8,
                )
    ax.set_xscale("log")
    ax.set_xlabel("field strength [T] (log)")
    ax.set_yticks(range(len(anatomies)))
    ax.set_yticklabels(anatomies)
    ax.set_title("Dataset landscape (marker = role, color = status)")
    produced.append(_save(fig, out_dir, "00_landscape.png"))

    # 01 domain shift: normalized intensity overlays for present datasets
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for e in entries:
        vals = e.get("norm_intensity") or []
        if len(vals):
            ax.hist(np.asarray(vals), bins=60, histtype="step", density=True, label=e["dataset_id"])
            plotted += 1
    ax.set_title("Normalized intensity — cross-dataset domain shift")
    ax.set_xlabel("normalized intensity")
    if plotted:
        ax.legend(fontsize=6, ncol=2)
    produced.append(_save(fig, out_dir, "01_domain_shift.png"))

    # 02 sample counts
    fig, ax = plt.subplots(figsize=(11, max(3, len(entries) * 0.18)))
    ids = [e["dataset_id"] for e in entries]
    counts = [max(e.get("n_records", 0), 0) for e in entries]
    ax.barh(ids, np.array(counts, dtype=float) + 0.1, color="#4393c3")
    ax.set_xscale("log")
    ax.set_xlabel("records (log)")
    ax.axvline(20, color="red", ls="--", label="n<20 → heavy aug")
    ax.legend()
    ax.set_title("Per-dataset sample counts")
    produced.append(_save(fig, out_dir, "02_sample_counts.png"))

    # 04 coverage matrix (the receipt)
    fig, ax = plt.subplots(figsize=(10, max(3, len(entries) * 0.22)))
    ax.axis("off")
    rows = [
        [
            e["dataset_id"],
            e["modality"],
            e["tier"],
            str(len(e.get("figures", []))),
            (e.get("error") or "ok")[:40],
        ]
        for e in entries
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["dataset", "modality", "tier", "#figs", "status"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    ax.set_title("Coverage matrix — every dataset accounted for")
    produced.append(_save(fig, out_dir, "04_coverage_matrix.png"))
    return produced
