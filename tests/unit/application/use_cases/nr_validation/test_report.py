"""Tests for the §8.8 NR-validation reporting surface."""

from __future__ import annotations

import csv
from pathlib import Path

from spectramr.application.use_cases.nr_validation.report import (
    render_cards_markdown,
    write_cards_table,
    write_nr_validation_report,
)


def _result() -> dict:
    return {
        "tiers_run": ["concordance", "fr_proxy"],
        "cards": {
            "gri": {
                "overall_pass": True,
                "tiers": {
                    "concordance": {"passed": True, "score": 0.81},
                    "fr_proxy": {"passed": True, "score": 0.74},
                },
            },
            "ndcr": {
                "overall_pass": False,
                "tiers": {
                    "concordance": {"passed": False, "score": 0.10},
                    "fr_proxy": {"passed": False, "score": float("nan")},
                },
            },
        },
        "redundancy": {"corr": {"gri|ndcr": 0.2}, "kept": ["gri", "ndcr"], "pruned": {}},
        "aggregator": {"backend": "ridge_closed_form"},
    }


def test_write_cards_table_csv_columns(tmp_path: Path) -> None:
    path = write_cards_table(_result(), tmp_path / "cards.csv")
    assert path.exists()
    rows = list(csv.reader(path.open()))
    header = rows[0]
    assert header == [
        "metric", "overall_pass",
        "concordance_pass", "concordance_score",
        "fr_proxy_pass", "fr_proxy_score",
    ]
    # sorted by metric: gri then ndcr
    gri = rows[1]
    assert gri[0] == "gri" and gri[1] == "True"
    assert gri[2] == "True" and gri[3] == "0.8100"
    ndcr = rows[2]
    assert ndcr[0] == "ndcr" and ndcr[1] == "False"
    # NaN score serialises to empty string, never "nan"
    assert ndcr[5] == ""


def test_render_cards_markdown_table() -> None:
    md = render_cards_markdown(_result())
    assert "| metric | overall | concordance | fr_proxy |" in md
    assert "PASS" in md and "fail" in md
    # 4 lines (header + separator + 2 metric rows) joined by "\n" => 3 newlines
    assert md.count("\n") == 3
    assert len(md.splitlines()) == 4


def test_write_nr_validation_report_emits_csv_and_md(tmp_path: Path) -> None:
    paths = write_nr_validation_report(_result(), tmp_path, figures=False)
    names = {p.name for p in paths}
    assert "nr_cards.csv" in names
    assert "nr_cards.md" in names
    assert all(p.exists() for p in paths)


def test_report_handles_empty_cards(tmp_path: Path) -> None:
    # No crash on a degenerate result (no tiers, no cards).
    empty = {"tiers_run": [], "cards": {}}
    path = write_cards_table(empty, tmp_path / "empty.csv")
    rows = list(csv.reader(path.open()))
    assert rows == [["metric", "overall_pass"]]
    assert render_cards_markdown(empty).startswith("| metric | overall |")
